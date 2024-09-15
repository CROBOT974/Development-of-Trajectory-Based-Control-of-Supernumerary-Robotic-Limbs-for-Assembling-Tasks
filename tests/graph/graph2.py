import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

folder_path = 'data'
file_path1 = os.path.join(folder_path, 'desire.npy')
desire = np.load(file_path1)
desire = np.degrees(desire)
file_path2 = os.path.join(folder_path, 'sensor.npy')
sensor = np.load(file_path2)
sensor = np.degrees(sensor)
time = np.arange(14000)

for i in range(7):
    plt.figure()  # Create a new figure for each plot
    plt.plot(time, desire[:, i], label='desired angles')
    plt.plot(time, sensor[:, i], label='real angles')
    plt.axvline(x=4000, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=6000, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=10000, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=12000, color='black', linestyle='--', linewidth=1)
    plt.title(f'Actuator {i + 1}')
    plt.xlabel('Time Step')
    plt.ylabel('Position / Angles (degree)')
    plt.legend()
    plt.show()  #


