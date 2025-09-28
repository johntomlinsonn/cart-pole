import mujoco
import mujoco_viewer
import time
import random as rand
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
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
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = rand.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

state_dim = 4  # x, x_dot, theta, theta_dot
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

optimizer = optim.Adam(q_network.parameters(), lr=0.001)
#storing the experiences
memory = ReplayBuffer(10000)

steps_done = 0

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

def state_to_tensor(state, device='cpu'):
    state_np = np.array(state, dtype=np.float32)
    return torch.from_numpy(state_np).unsqueeze(0).to(device)


def to_numpy_state(state):
    return np.array(state, dtype=np.float32)


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
viewer = mujoco_viewer.MujocoViewer(model, data)

#Randomizing where the pole is starting 
def Intial_state():
    upper_bound = math.pi/8 +2* math.pi
    lower_bound =  3*(math.pi)/2 + 3* math.pi/8
    xpos = 0  # x
    xdot = 0  # x_dot
    theta = rand.uniform(lower_bound, upper_bound)  # theta
    #theta = 0
    theta_dot = 0  # theta_dot
    return xpos, xdot, theta, theta_dot

#Gathering the state of the cartpole
def gatherState(data):
    # x, x_dot, theta, theta_dot
    return [data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]]

#Pushing the cart left or right with a certain force
def right(data, force=1, delta_time = 0.05):
    accel = force
    data.qvel[1] =  data.qvel[1] + accel * delta_time

def left(data, force=1, delta_time = 0.05):
    accel = -force
    data.qvel[1] =  data.qvel[1] + accel * delta_time

#Reward function to see how well the cartpole is doing
def close_to_0_degrees(theta):
    d = abs(theta - 2 * math.pi)
    closeness_percent = 100 * (1 - d / (math.pi / 4))
    return max(0, closeness_percent)

def reward(state, num_steps):
    x, x_dot, theta, theta_dot = state
    x_reward = (1 - abs(0-x))*100
    theta_reward = close_to_0_degrees(theta)
    if theta_reward <= 0:
        return -10000
    return x_reward + theta_reward


num_episodes = 5000
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

for episode in range(num_episodes):
    data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1] = Intial_state()
    total_reward = 0
    while 10_000:
        viewer.render()
        state = gatherState(data)
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                  np.exp(-1. * steps_done / EPSILON_DECAY)
        action = select_action(state, epsilon)
        if action == 0:
            left(data)
        else:
            right(data)
        mujoco.mj_step(model, data)
        next_state = gatherState(data)
        r = reward(next_state, steps_done)
        done = (close_to_0_degrees(next_state[2]) == 0)
        memory.push(to_numpy_state(state), action, r, to_numpy_state(next_state), done)
        # train step: try optimizing the model
        optimize_model(device)

        state = next_state
        total_reward += r

        # periodically update the target network
        if steps_done % TARGET_UPDATE == 0:
            target_network.load_state_dict(q_network.state_dict())

        if done:
            break
    # end of episode: record total reward
    episode_rewards.append(total_reward)
    print(f"Episode {episode+1}/{num_episodes} finished - total_reward={total_reward:.2f}")

# training finished — save the rewards plot
save_reward_plot(episode_rewards, 'total_reward.png')



    