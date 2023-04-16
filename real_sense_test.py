import pyrealsense2.pyrealsense2 as rs
import numpy as np
import time
import matplotlib.pyplot as plt

iterations = 50
colors = np.linspace(0, iterations, iterations)
x_data = []; y_data = []
# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()

# Build config object and request pose data
cfg = rs.config()
cfg.enable_stream(rs.stream.pose)

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

    x_pos = -data.translation.x
    y_pos = data.translation.y
    x_data.append(x_pos)
    y_data.append(y_pos)
    # print('X position: {0:.7f}, Y position: {1:.7f}'.format(x_pos, y_pos))
    time.sleep(0.5)

plt.scatter(x_data, y_data, c = colors, cmap = 'plasma', alpha= .7, edgecolors= 'black')
plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.title('Postion over time')
plt.grid(True)
cbar = plt.colorbar()
cbar.set_label('Time Progression')    
plt.show()
pipe.stop()