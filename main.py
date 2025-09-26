import mujoco
import mujoco_viewer
import time
import random as rand
import math
model = mujoco.MjModel.from_xml_path("cartpole.xml")
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)


def Intial_state():
    upper_bound = math.pi/4 +2* math.pi
    lower_bound =  (7 * math.pi)/4
    xpos = rand.random()  # x
    xdot = 0  # x_dot
    theta = rand.uniform(lower_bound, upper_bound)  # theta
    theta_dot = 0  # theta_dot
    return xpos, xdot, theta, theta_dot

def gatherState(data):
    # x, x_dot, theta, theta_dot
    return [data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]]


#actions
def right(data, force=1, delta_time = 0.01):
    accel = force
    data.qvel[1] =  data.qvel[1] + accel * delta_time

def left(data, force=1, delta_time = 0.01):
    accel = -force
    data.qvel[1] =  data.qvel[1] + accel * delta_time

def close_to_0_degrees(theta):
    d = abs(theta - 2 * math.pi)
    closeness_percent = 100 * (1 - d / (math.pi / 4))
    return max(0, closeness_percent)

def reward(state):
    x, x_dot, theta, theta_dot = state
    x_reward = (1 - abs(0-x))*100
    theta_reward = close_to_0_degrees(theta)

    return x_reward + theta_reward


# 4 state variables, 2 actions
# x, x_dot, theta, theta_dot
# left -1, right +1
first = True
second = False
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
    