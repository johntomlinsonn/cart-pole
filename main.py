import mujoco
import mujoco_viewer
import time
import random as rand
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = rand.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

# Loading the cartpole 3d model for  the enviorment
model = mujoco.MjModel.from_xml_path("cartpole.xml")
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

#Randomizing where the pole is starting 
def Intial_state():
    upper_bound = math.pi/4 +2* math.pi
    lower_bound =  (7 * math.pi)/4
    xpos = rand.random()  # x
    xdot = 0  # x_dot
    theta = rand.uniform(lower_bound, upper_bound)  # theta
    theta_dot = 0  # theta_dot
    return xpos, xdot, theta, theta_dot

#Gathering the state of the cartpole
def gatherState(data):
    # x, x_dot, theta, theta_dot
    return [data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]]

#Pushing the cart left or right with a certain force
def right(data, force=1, delta_time = 0.01):
    accel = force
    data.qvel[1] =  data.qvel[1] + accel * delta_time

def left(data, force=1, delta_time = 0.01):
    accel = -force
    data.qvel[1] =  data.qvel[1] + accel * delta_time

#Reward function to see how well the cartpole is doing
def close_to_0_degrees(theta):
    d = abs(theta - 2 * math.pi)
    closeness_percent = 100 * (1 - d / (math.pi / 4))
    return max(0, closeness_percent)

def reward(state):
    x, x_dot, theta, theta_dot = state
    x_reward = (1 - abs(0-x))*100
    theta_reward = close_to_0_degrees(theta)

    return x_reward + theta_reward

while True:
    mujoco.mj_step(model, data)
    viewer.render()
    
    if second:
        time.sleep(3)
        second = False
    if first:
        data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1] = Intial_state()
        viewer.render()
        first = False
        second = True
    