import os
import numpy as np
import matplotlib.pyplot as plt

folder_path = 'data'
file_path2 = os.path.join(folder_path, 'rhand_tras.npy')
rhand = np.load(file_path2)
rhand[:, 0] +=1
file_path3 = os.path.join(folder_path, 'lhand_tras.npy')
lhand = np.load(file_path3)
lhand[:, 0] +=1

file_path4 = os.path.join(folder_path, 'hand_tras1.npy')
rhandtest = np.load(file_path4)
rhandtest[:, 0] +=1
file_path5 = os.path.join(folder_path, 'hand_tras2.npy')
lhandtest = np.load(file_path5)
lhandtest[:, 0] +=1


file_path6 = os.path.join(folder_path, 'robot_tras1.npy')
robot1 = np.load(file_path6)
file_path7 = os.path.join(folder_path, 'robot_tras2.npy')
robot2 = np.load(file_path7)
file_path8 = os.path.join(folder_path, 'robot_tras3.npy')
robot3 = np.load(file_path8)

'''2D图'''
robot1_2d = robot1[:, 1:]
robot3_2d = robot3[:, 1:]
rhand_2d = rhand[:, 1:]
lhand_2d = lhand[:, 1:]

plt.figure()

distances1 = np.linalg.norm(robot1_2d - rhand_2d, axis=1)
distances2 = np.linalg.norm(robot3_2d - lhand_2d, axis=1)

# Find the maximum distance and the corresponding index
max_distanc1 = np.max(distances1)
max_index1 = np.argmax(distances1)
max_distance2 = np.max(distances2)
max_index2 = np.argmax(distances2)

# Plot points from the 'robot' array using y and z coordinates
k = -1
plt.plot(robot1[:k, 1], robot1[:k, 2], c='blue', label='Robot Trajectory')
plt.plot(robot2[:, 1], robot2[:, 2], c='blue')
plt.plot(robot3[:, 1], robot3[:, 2], c='blue')

# Plot points from the 'rhand' array using y and z coordinates
plt.plot(rhand[:k, 1], rhand[:k, 2], c='red', label='R_Hand Trajectory')


# Plot points from the 'lhand' array using y and z coordinates
plt.plot(lhand[:, 1], lhand[:, 2], c='green', label='L_Hand Trajectory')
plt.scatter(robot1[max_index1, 1], robot1[max_index1, 2], edgecolor='black', facecolor='none', s=100, label='Largest Error')
plt.scatter(robot1[-1, 1], robot1[-1, 2], edgecolor='red', facecolor='none', s=100, label='End of tracing 1')
plt.scatter(robot3[0, 1], robot3[0, 2], edgecolor='green', facecolor='none', s=100, label='Start of tracing 2')

# Add labels to the axes
plt.xlabel('y-axis(m)')
plt.ylabel('z-axis(m)')

# Add a legend
plt.legend()

# Show the plot
plt.show()

# # Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot lines connecting points from the 'robot' array
k = -1
ax.plot(robot1[:, 0], robot1[:, 1], robot1[:, 2], c='blue', label='Robot Trajectory')
ax.plot(robot2[:, 0], robot2[:, 1], robot2[:, 2], c='blue')
ax.plot(robot3[:, 0], robot3[:, 1], robot3[:, 2], c='blue')


# Plot lines connecting points from the 'rhand' array
ax.plot(rhand[:, 0], rhand[:, 1], rhand[:, 2], c='red', label='R_Hand Trajectory')

# Plot lines connecting points from the 'lhand' array
ax.plot(lhand[:, 0], lhand[:, 1], lhand[:, 2], c='green', label='L_Hand Trajectory')


# # Plot lines connecting points from the 'rhand' array
# ax.plot(rhandtest[:, 0], rhandtest[:, 1], rhandtest[:, 2], c='red', label='R_Hand Trajectory')
#
# ax.plot(lhandtest[:, 0], lhandtest[:, 1], lhandtest[:, 2], c='green', label='L_Hand Trajectory')
#
#
ax.scatter(robot1[-1, 0], robot1[-1, 1], robot1[-1, 2], edgecolor='red', facecolor='none', s=10, label='End of tracing 1')
ax.scatter(robot3[0, 0], robot3[0, 1], robot3[0, 2], edgecolor='green', facecolor='none', s=10, label='Start of tracing 2')


# Add labels
ax.set_xlabel('X(m)')
ax.set_ylabel('Y(m)')
ax.set_zlabel('Z(m)')

# Add a legend
ax.legend()

# Show the plot
plt.show()