import pyrealsense2.pyrealsense2 as rs
import numpy as np
import time
from datetime import datetime as dt
import matplotlib.pyplot as plt

iterations = 50
colors = np.linspace(0, iterations, iterations)
x_data = []; y_data = []

## Basic ##
# Declare RealSense pipeline, encapsulating the actual device and sensors
# pipe = rs.pipeline()

# # Build config object and request pose data
# cfg = rs.config()
# cfg.enable_stream(rs.stream.pose)

# # Start streaming with requested config
# pipe.start(cfg)

# Adjusted calibration ##
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

# Start streaming with requested config
    pipe.start(cfg)

for i in range(iterations):
    frames = pipe.wait_for_frames()
    pose = frames.get_pose_frame()
    data = pose.get_pose_data()
    w = data.rotation.w
    x = -data.rotation.z
    y = data.rotation.x
    z = -data.rotation.y

    x_pos = data.translation.z
    y_pos = data.translation.x
    x_data.append(x_pos)
    y_data.append(y_pos)
    print('X position: {0:.7f}, Y position: {1:.7f}'.format(x_pos, y_pos))
    time.sleep(0.5)

plt.scatter(x_data, y_data, c = colors, cmap = 'plasma', alpha= .7, edgecolors= 'black')
plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.title('Postion over time')
plt.grid(True)
cbar = plt.colorbar()
cbar.set_label('Time Progression')
time_stamp = dt.now().strftime("%Y_%m_%d_%H_%M_%S")
plt.savefig('/home/pi/Zlawren1_Swimbot/RealSense_Test_Graphs/Test_Run_' + time_stamp + '.png')
plt.show()
pipe.stop()