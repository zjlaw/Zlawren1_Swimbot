import pyrealsense2.pyrealsense2 as rs
import numpy as np
import math as m
from adafruit_servokit import ServoKit
import time
from datetime import datetime as dt
import matplotlib.pyplot as plt
import pickle
import xlsxwriter as xl

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
episodes = 20
iterations = 50
interval = .5
ang_lim = 21
omega_lim = 4
alpha = .2
gamma = .9
epsilon_decay = 200
vel_space = np.linspace(neg_vel, pos_vel, steps)
heading_space = np.linspace(neg_ang, pos_ang, steps)
omega_space = np.linspace(neg_omega, pos_omega, steps)
actions = (-1, -0.81, -.62, -0.43, -0.23, -0.04, 0.15, 0.29, 0.43, 0.58, 0.72, 0.86, 1)
vel_factor = 5
heading_factor = .067
omega_factor = .25
action_factor = .1
action_ratio = abs(actions[len(actions) - 1] - motor_stop) / abs(actions[0] - motor_stop)
neg_reward = -5
channel = 16
time_step = 0.5
startup = 5
colors = np.linspace(0, iterations, iterations + 1)
output = 5
pause = 20

states = []
for v in range(len(vel_space)):
    for h in range(len(heading_space)):
        for o in range(len(omega_space)):
            states.append((v,h,o))

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
        x_pos = data.translation.z
        y_pos = data.translation.x
        xvel = data.velocity.z
        yvel = data.velocity.x
        omega = data.angular_velocity.y
        print('X position: {0:.7f}, Y position: {1:.7f}'.format(x_pos, y_pos))
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
    if abs(heading) < ang_lim:
        heading_reward = (ang_lim - abs(heading)) * heading_factor
    else:
        heading_reward = neg_reward
    return heading_reward

def export_data(episode, time_now, x_data, xvel_data, y_data, yvel_data, heading_data, heading_reward, omega_data, action_taken):
    wb = xl.Workbook('/home/pi/Zlawren1_Swimbot/Logs/Episode:_' + str(episode) + '_' + dt.now().strftime("%Y_%m_%d_%H_%M_%S") + '.xlsx')
    ws = wb.add_worksheet("Logged data")
    ws.write(0, 0, "Time Stamp")
    ws.write(0, 1, "X Position [m]")
    ws.write(0, 2, "X Velocity [m/s]")
    ws.write(0, 3, "Y Position [m]")
    ws.write(0, 4, "Y Velocity [m/s]")
    ws.write(0, 5, "Heading [Deg]")
    ws.write(0, 6, "Heading Rewards")
    ws.write(0, 7, "Omega [rad/s]")
    ws.write(0, 8, "Action")
    for i, time in enumerate(time_now):
        ws.write(i+1, 0, time)
    for i, x_pos in enumerate(x_data):
        ws.write(i+1, 1, x_pos)
    for i, x_vel in enumerate(xvel_data):
        ws.write(i+1, 2, x_vel)
    for i, y_pos in enumerate(y_data):
        ws.write(i+1, 3, y_pos)
    for i, y_vel in enumerate(yvel_data):
        ws.write(i+1, 4, y_vel)
    for i, head in enumerate(heading_data):
        ws.write(i+1, 5, head)
    for i, head_rew in enumerate(heading_reward):
        ws.write(i+1, 6, head_rew)
    for i, ome in enumerate(omega_data):
        ws.write(i+1, 7, ome)
    for i, act in enumerate(action_taken):
        ws.write(i+1, 8, act)
    wb.close()

servo = ServoKit(channels = channel).continuous_servo[0]
servo.throttle = motor_stop

print("Do you want to load a Q-Table y/n? ")
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

print("Do you want to load an epsilon value y/n? ")
if input() == 'y':
    print("What file path should I follow to find your epsilon?")
    e_file = open(input(), 'rb')
    epsilon = pickle.load(e_file)
    e_file.close()
else:
    print("Epsilon will be set to 1 and decay linearly over the session.")
    epsilon = 1

total_reward = np.zeros(episodes)

pipe = rs.pipeline()
# Build config object and request pose data
cfg = rs.config()
profile = cfg.resolve(pipe)
dev = profile.get_device()
tm2 = dev.as_tm2()

if(tm2):
    # tm2.first_wheel_odometer()?
    pose_sensor = tm2.first_pose_sensor()
    wheel_odometer = pose_sensor.as_wheel_odometer()

    # calibration to list of uint8
    f = open("/home/pi/Zlawren1_Swimbot/calibration_odometry.json")
    chars = []
    for line in f:
        for c in line:
            chars.append(ord(c))  # char to uint8

    # load/configure wheel odometer
    wheel_odometer.load_wheel_odometery_config(chars)

print("Realsense initializing.")
time.sleep(startup)

# Start streaming with requested config
pipe.start(cfg)


for e in range(episodes):

    x_data = []; y_data = []; xvel_data = []; yvel_data = []; time_now = []; heading_data = []; omega_data = [] 
    action_taken = []; track_reward = []

    act = motor_stop
    servo.throttle = act
    action_taken.append(act)
    ep_reward = 0
    print('Starting Episode: ' + str(e))
    xvel, heading, omega = get_obs(pipe)
    for i in range(iterations):
        state = get_state(xvel, heading, omega)
        compare = np.random.random()
        if compare > epsilon:
            act = get_act(state)
        else:
            rand = np.random.randint(low= 0, high= len(actions))
            act = actions[rand]
        servo.throttle = act
        action_taken.append(act)
        time.sleep(time_step)
        xvel_n, heading_n, omega_n = get_obs(pipe)
        state_n = get_state(xvel_n, heading_n, omega_n)
        reward = get_reward(state_n, act)
        track_reward.append(reward)
        ep_reward += reward
        act_n = get_act(state_n)
        q_value[(state, act)] = q_value[(state, act)] + alpha * (reward + gamma * q_value[(state_n, act_n)] - q_value[(state, act)])
        xvel, heading, omega = xvel_n, heading_n, omega_n
    if epsilon - 2 / epsilon_decay > 0:
        epsilon -= 2 / epsilon_decay
    else:
        epsilon = 0

    time_stamp = dt.now().strftime("%Y_%m_%d_%H_%M_%S")
    servo.throttle = motor_stop
    total_reward[e] = ep_reward

    plt.scatter(x_data, y_data, c = colors, cmap = 'plasma', alpha= .7, edgecolors= 'black')
    plt.xlabel('X position [m]')
    plt.ylabel('Y position [m]')
    plt.title('Postion over time')
    plt.grid(True)
    cbar = plt.colorbar()
    cbar.set_label('Time Progression')
    plt.savefig('/home/pi/Zlawren1_Swimbot/Plots/' + time_stamp + 'episode:' + str(e) + '.png')
    plt.close()

    heading_reward = track_reward
    export_data(e, time_now, x_data, xvel_data, y_data, yvel_data, heading_data, heading_reward, omega_data, action_taken)
    if (e+1) % output == 0:
        file_loc = open('/home/pi/Zlawren1_Swimbot/Q_Tables/' + time_stamp + '_episode:' + str(e) + '.file', 'wb')
        pickle.dump(q_value, file_loc)
        file_loc.close()

        e_loc = open('/home/pi/Zlawren1_Swimbot/Epsilon/' + time_stamp + '_episode:' + str(e) + '.file', 'wb')
        pickle.dump(epsilon, e_loc)
        e_loc.close()
    print('Are you ready for the next episode: y/n?')
    if input() == 'y':
        pass
    else:
        print('You have 20 seconds before the next episode begins')
        time.sleep(pause)
    
pipe.stop()