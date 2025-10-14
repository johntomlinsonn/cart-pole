import math
import random as rand

import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        #nueral network with 2 hidden layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #outputs the q value for each action, based on how well the action will do
        self.fc3 = nn.Linear(hidden_size, action_size)

    #How the data flows through the network
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            np.array(state, dtype=np.float32),
            action,
            np.float32(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = rand.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

state_dim = 5  # normalized x, x_dot, sin(theta), cos(theta), theta_dot
action_dim = 2  # left, right

q_network = QNetwork(state_dim, action_dim)
#target network learns more slowly to stabilize training
target_network = QNetwork(state_dim, action_dim)
#copying the weights from the q network to the target network
target_network.load_state_dict(q_network.state_dict())
#moving the model to eval mode
target_network.eval()

# device for training/inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
q_network.to(device)
target_network.to(device)

optimizer = optim.Adam(q_network.parameters(), lr=5e-4)
#storing the experiences
memory = ReplayBuffer(10000)

steps_done = 0

# environment limits (from XML) and reward shaping coefficients
X_LIMIT = 1.0
THETA_LIMIT_RADIANS = math.radians(12)
FORCE_MAG = 1.0
MAX_STEPS_PER_EPISODE = 1000
RENDER_TRAINING = False

def select_action(state, epsilon):
    global steps_done
    sample = rand.random()
    steps_done += 1
    if sample > epsilon:
        with torch.no_grad():
            state_t = state_to_tensor(state)
            q_values = q_network(state_t)
            return q_values.max(1)[1].item()
    else:
        return rand.randrange(action_dim) 

def state_to_tensor(state):
    state_np = np.array(state, dtype=np.float32)
    return torch.from_numpy(state_np).unsqueeze(0).to(device)


def prepare_batch(batch, device='cpu'):
    state, action, reward, next_state, done = batch
    states_t = torch.from_numpy(state).float().to(device)
    actions_t = torch.from_numpy(action).long().to(device)
    rewards_t = torch.from_numpy(reward).float().to(device)
    next_states_t = torch.from_numpy(next_state).float().to(device)
    dones_t = torch.from_numpy(done.astype(np.float32)).float().to(device)
    return states_t, actions_t, rewards_t, next_states_t, dones_t


# Loading the cartpole 3d model for  the enviorment
model = mujoco.MjModel.from_xml_path("cartpole.xml")
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data) if RENDER_TRAINING else None

#Randomizing where the pole is starting 
def Intial_state():
    theta = rand.uniform(-math.radians(6), math.radians(6))
    xpos = rand.uniform(-0.05, 0.05)
    xdot = rand.uniform(-0.5, 0.5)
    theta_dot = 0
    return xpos, xdot, theta, theta_dot

#Gathering the state of the cartpole
def gather_raw_state(data):
    # x, x_dot, theta, theta_dot
    return np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]], dtype=np.float32)


def process_state(raw_state):
    x, x_dot, theta, theta_dot = raw_state
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    norm_x = np.clip(x / X_LIMIT, -1.0, 1.0)
    norm_x_dot = np.tanh(x_dot)
    norm_theta_dot = np.tanh(theta_dot)
    return np.array([norm_x, norm_x_dot, sin_theta, cos_theta, norm_theta_dot], dtype=np.float32)

#Pushing the cart left or right with a certain force
def right(data, force=FORCE_MAG):
    data.ctrl[0] = np.clip(force, -1.0, 1.0)


def left(data, force=FORCE_MAG):
    data.ctrl[0] = np.clip(-force, -1.0, 1.0)

#Reward function to see how well the cartpole is doing
def compute_reward(raw_state):
    x, x_dot, theta, theta_dot = raw_state
    angle_error = abs(theta)
    position_error = abs(x)
    reward = 1.0
    reward -= 2.0 * (angle_error / THETA_LIMIT_RADIANS)
    reward -= 0.5 * (position_error / X_LIMIT)
    reward -= 0.01 * (abs(x_dot) + abs(theta_dot))
    return max(reward, -2.0)


def is_terminal(raw_state):
    x, _, theta, _ = raw_state
    return abs(theta) > THETA_LIMIT_RADIANS or abs(x) > X_LIMIT


num_episodes = 300
batch_size = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 500

# keep track of total reward per episode for plotting later
episode_rewards = []

# DQN hyperparameters
GAMMA = 0.99
TARGET_UPDATE = 1000


def optimize_model(device='cpu'):
    """Sample a batch and perform a single DQN optimization step."""
    if len(memory) < batch_size:
        return
    batch = memory.sample(batch_size)
    states_t, actions_t, rewards_t, next_states_t, dones_t = prepare_batch(batch, device)

    # Compute current Q values
    q_values = q_network(states_t)
    state_action_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # Compute target Q values
    with torch.no_grad():
        next_q_values = target_network(next_states_t).max(1)[0]
        expected_q_values = rewards_t + (1.0 - dones_t) * GAMMA * next_q_values

    loss = F.mse_loss(state_action_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1.0)
    optimizer.step()


def save_reward_plot(rewards, filename='total_reward.png'):
    """Save a matplotlib plot of total reward per episode to filename."""
    plt.figure(figsize=(8, 5))
    plt.plot(rewards, label='Total reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_actions(actions, filename="actions_taken.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(actions, label='action_taken')
    plt.xlabel('Step')
    plt.ylabel('Total Reward')
    plt.title('ACtions taken')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()



for episode in range(num_episodes):
    data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1] = Intial_state()
    data.ctrl[:] = 0
    total_reward = 0
    raw_state = gather_raw_state(data)
    state = process_state(raw_state)
    actions = []
    for _ in range(MAX_STEPS_PER_EPISODE):
        
        if RENDER_TRAINING:
            viewer.render()

        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                  np.exp(-1. * steps_done / EPSILON_DECAY)
        action = select_action(state, epsilon)
        actions.append(action)
        if action == 0:
            left(data)
        else:
            right(data)

        mujoco.mj_step(model, data)
        next_raw_state = gather_raw_state(data)
        next_state = process_state(next_raw_state)

        r = compute_reward(next_raw_state)
        done = is_terminal(next_raw_state)

        memory.push(state, action, r, next_state, done)
        optimize_model(device)

        state = next_state
        raw_state = next_raw_state
        total_reward += r

        if steps_done % TARGET_UPDATE == 0:
            target_network.load_state_dict(q_network.state_dict())

        if done:
            break

    episode_rewards.append(total_reward)
    print(f"Episode {episode+1}/{num_episodes} finished - total_reward={total_reward:.2f}")
    save_reward_plot(episode_rewards, 'total_reward.png')
    if episode % 10 == 0:
        plot_actions(actions,f"actions_over_time_{episode}.png")



if viewer is not None:
    viewer.close()



    