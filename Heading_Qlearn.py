import pyrealsense2.pyrealsense2 as rs
import numpy as np
import math as m
from adafruit_servokit import ServoKit
# import multiprocessing
import time
from datetime import datetime as dt
import matplotlib.pyplot as plt
import pickle
import xlsxwriter as xl
# from termios import

steps = 37
neg_vel = -3
pos_vel = 3
neg_ang = -180
pos_ang = 180
neg_omega = -3
pos_omega = 3
motor_cw = 1
motor_stop = .15
stop_index = 6
motor_ccw = -1
action_step = 13
episodes = 5
iterations = 50
interval = .5
ang_lim = 21
omega_lim = 4
alpha = .2
gamma = .9
epsilon = 1
vel_space = np.linspace(neg_vel, pos_vel, steps)
heading_space = np.linspace(neg_ang, pos_ang, steps)
omega_space = np.linspace(neg_omega, pos_omega, steps)
# actions = np.linspace(motor_ccw, motor_cw, action_step)
actions = (-1, -0.81, -.62, -0.43, -0.23, -0.04, 0.15, 0.29, 0.43, 0.58, 0.72, 0.86, 1)
vel_factor = 5
heading_factor = .067
omega_factor = .25
action_factor = .1
action_ratio = abs(actions[len(actions) - 1] - motor_stop) / abs(actions[0] - motor_stop)
neg_reward = -5
channel = 16
time_step = 0.5
startup = 10

states = []
for v in range(len(vel_space)):
    for h in range(len(heading_space)):
        for o in range(len(omega_space)):
            states.append((v,h,o))

x_data = []; y_data = []; xvel_data = []; time_now = []; heading_data = []; omega_data = []; yvel_data = []
action_taken = []

velocity = []; yaw = []; yaw_dot = []; track_reward = []
velocity_reward = []; heading_reward = []; omega_reward = []; action_reward = []

def get_obs(pipe):
    # Wait for the next set of frames from the camera
    # Fetch pose frame
    frames = pipe.wait_for_frames()
    pose = frames.get_pose_frame()
    if pose:
        # Print some of the pose data to the terminal
        data = pose.get_pose_data()
        w = data.rotation.w
        x = -data.rotation.z
        y = data.rotation.x
        z = -data.rotation.y
        
        
        # pitch = -np.arcsin(2.0 * (x * z - w * y)) * 180 / (np.pi)
        # roll  =  np.arctan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z) * 180.0 / (np.pi)
        heading   =  np.arctan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z) * 180.0 / (np.pi)
        x_pos = -data.translation.x
        y_pos = data.translation.y
        xvel = -data.velocity.x
        yvel = data.velocity.y
        omega = data.angular_velocity.y
        print('X position: {0:.7f}, Y position: {1:.7f}'.format(x_pos, y_pos))
        # print("Frame #{}".format(pose.frame_number))
        # print("RPY [deg]: Roll: {0:.7f}, Pitch: {1:.7f}, Yaw: {2:.7f}".format(roll, pitch, yaw))
        # print("Velocity [m/s]: {0:.7f}" .format(xvel))
        x_data.append(x_pos)
        y_data.append(y_pos)
        xvel_data.append(xvel)
        yvel_data.append(yvel)
        time_now.append(dt.now().strftime("%Y_%m_%d_%H_%M_%S"))
        heading_data.append(heading)
        omega_data.append(omega)
    return xvel, heading, omega

def get_state(xvel,heading, omega):
    xvel = int(np.digitize(xvel, vel_space)) - 1
    heading = int(np.digitize(heading, heading_space)) - 1
    omega = int(np.digitize(omega, omega_space)) - 1
    return xvel, heading, omega

def get_act(state):
    possibilities = np.array([q_value[(state, a)] for a in actions])
    index = np.argmax(possibilities)
    act = actions[index]
    return act

def get_reward(state_n, act):
    (xvel, heading, omega) = state_n
    xvel_reward = xvel * vel_factor
    if abs(heading) < ang_lim:
        heading_reward = (ang_lim - abs(heading)) * heading_factor
    else:
        heading_reward = neg_reward
    if act == motor_stop:
        action_reward = 0.5
    elif act < motor_stop:
        action_reward = -abs(act - motor_stop)
        action_reward = action_reward * action_ratio
    else:
        action_reward = -abs(act - motor_stop)
    return heading_reward

def export_data(episode, time_now, xvel_data, yvel_data, heading_data, omega_data, action_taken, heading_reward):
    wb = xl.Workbook("Episode: " + str(episode) + dt.now().strftime("%Y_%m_%d_%H_%M_%S"))
    ws = wb.add_worksheet("Logged data")
    ws.write(0, 0, "Time Stamp")
    ws.write(0, 1, "Velocity [m/s]")
    ws.write(0, 2, "Velocity Reward")
    ws.write(0, 3, "Heading [Deg]")
    ws.write(0, 4, "Heading Reward")
    ws.write(0, 5, "Omega [rad/s]")
    ws.write(0, 6, "Omega Reward")
    ws.write(0, 7, "Action")
    ws.write(0, 8, "Action Reward")
    ws.write(0, 9, "Reward Sum")
    for i, time in enumerate(time_now):
        ws.write(i+1, 0, time)
    for i, vel in enumerate(xvel_data):
        ws.write(i+1, 1, vel)
    for i, vel_rew in enumerate(yvel_data):
        ws.write(i+1, 2, vel_rew)
    for i, head in enumerate(heading_data):
        ws.write(i+1, 3, head)
    for i, head_rew in enumerate(heading_reward):
        ws.write(i+1, 4, head_rew)
    for i, ome in enumerate(omega_data):
        ws.write(i+1, 5, ome)
    for i, ome_rew in enumerate(omega_reward):
        ws.write(i+1, 6, ome_rew)
    for i, act in enumerate(action_taken):
        ws.write(i+1, 7, act)
    for i, act_rew in enumerate(action_reward):
        ws.write(i+1, 8, act_rew)
    for i, rew in enumerate(track_reward):
        ws.write(i+1, 9, rew)
    wb.close()

servo = ServoKit(channels = channel).continuous_servo[0]
servo.throttle = motor_stop

print("Do you want to load a Q-Table or create a new one, y/n? ")
if input() == 'y':
    print("What file path should I follow to find your Q-Table?")
    q_file = open(input(), 'rb')
    q_value = pickle.load(q_file)
    q_file.close()
else:
    print("Randomly Initializing Q Table.")
    q_value = {}
    for state in states:
        for act in actions:
            q_value[(state, act)] = np.random.random()    
total_reward = np.zeros(episodes)

for e in range(episodes):
    act = motor_stop
    servo.throttle = act
    ep_reward = 0
    pipe = rs.pipeline()
    # Build config object and request pose data
    cfg = rs.config()
    cfg.enable_stream(rs.stream.pose)

    # Start streaming with requested config
    pipe.start(cfg)
    print("Realsense initializing.")
    time.sleep(startup)
    print('Starting Episode: ' + str(e))
    xvel, heading, omega = get_obs(pipe)
    for i in range(iterations):
        # if sym == False:
        #     state = get_state(xvel, heading, omega)
        # if sym == True:
        #     sym_states = get_state(xvel, heading, omega)
        #     state = (sym_states[0], sym_states[1], sym_states[2])
        #     sym_state = (sym_states[0], sym_states[3], sym_states[4])
        state = get_state(xvel, heading, omega)
        compare = np.random.random()
        if compare > epsilon:
            act = get_act(state)
        else:
            rand = np.random.randint(low= 0, high= len(actions))
            act = actions[rand]
        servo.throttle = act
        time.sleep(time_step)
        xvel_n, heading_n, omega_n = get_obs(pipe)
        # if sym == False:
        #     state_n = get_state(xvel_n, heading_n, omega_n)
        # if sym == True:
        #     sym_states_n = get_state(xvel_n, heading_n, omega_n)
        #     state_n = (sym_states_n[0], sym_states_n[1], sym_states_n[2])
        #     sym_state_n = (sym_states_n[0], sym_states_n[3], sym_states_n[4])
        state_n = get_state(xvel_n, heading_n, omega_n)
        rewards = get_reward(state_n, act)
        reward = rewards[0]
        track_reward.append(rewards[0]); velocity_reward.append(rewards[1]); heading_reward.append(rewards[2]); omega_reward.append(rewards[3]); action_reward.append(rewards[4])
        ep_reward += reward
        act_n = get_act(state_n)
        # if sym == False:
        #     q_value[state, act] = q_value[state, act] + alpha * (reward + gamma * q_value[state_n, act_n] - q_value[state, act])
        # if sym == True:
        #     q_value[state, act] = q_value[state, act] + alpha * (reward + gamma * q_value[state_n, act_n] - q_value[state, act])
        #     q_value[sym_state, -act] = q_value[sym_state, -act] + alpha * (reward + gamma * q_value[sym_state_n, -act_n] - q_value[sym_state, -act])
        q_value[(state, act)] = q_value[(state, act)] + alpha * (reward + gamma * q_value[(state_n, act_n)] - q_value[(state, act)])
        xvel, heading, omega = xvel_n, heading_n, omega_n
    if epsilon > 2 / episodes:
        epsilon -= 2 / episodes
    else:
        epsilon = 0  
    servo.throttle = motor_stop
    total_reward[e] = ep_reward
    pipe.stop()
    export_data(e, time_now, velocity, yaw, yaw_dot, action_taken, velocity_reward, heading_reward, omega_reward, action_reward, track_reward)
    time_stamp = dt.now().strftime("%H_%M_%S")
    # if sym == False:
    #     file_loc = open('D:/UNCC/Research/Summer 23/Q_Tables/qtable_' + time_stamp + 'episode:' + str(e) + 'no_symmetry.file', 'wb')
    # if sym == True:
    #     file_loc = open('D:/UNCC/Research/Summer 23/Q_Tables/qtable_' + time_stamp + 'episode:' + str(e) + 'symmetry.file', 'wb')
    file_loc = open('/home/pi/OUR2022Research/onPi/qtable_' + time_stamp + 'episode:' + str(e) + '.file', 'wb')
    pickle.dump(q_value, file_loc)
    file_loc.close()