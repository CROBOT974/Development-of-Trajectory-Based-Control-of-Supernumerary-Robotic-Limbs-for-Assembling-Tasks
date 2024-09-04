from unittest import TestCase
import numpy as np
import mujoco.viewer
import os

import mujoco
from pid_control.src.controller import PIDController
import mediapy as media

model = mujoco.MjModel.from_xml_path("D:/dissertation/Mujoco/model/kortex/scene4.xml")
data = mujoco.MjData(model)

# 定义PID参数
kp, ki, kd = 1.0, 0.0, 0.0

position_controllers = [PIDController(kp, ki, kd, ts=model.opt.timestep) for _ in range(20)]
velocity_controllers = [PIDController(kp, ki, kd, ts=model.opt.timestep) for _ in range(20)]

kp_p_v = [[2,2,2,2, 30, 40, 100, 75, 50, 50, 50, 50, 30, 40, 100, 75, 50, 50, 50, 50],  # position kp-value
          [1,1,1,1, 45, 90, 25, 22, 20, 10, 10, 10, 45, 90, 25, 22, 20, 10, 10, 10, ]]  # velocity kp-value
ki_p_v = [[0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [2,2,2,2, 20, 70, 240, 110, 110, 22, 22, 22, 20, 70, 240, 110, 110, 22, 22, 22]]
kd_p_v = [[0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

for i in range(20):
    position_controllers[i].set_parameter(kp_p_v[0][i], ki_p_v[0][i], kd_p_v[0][i])
    velocity_controllers[i].set_parameter(kp_p_v[1][i], ki_p_v[1][i], kd_p_v[1][i])

time_num = 0
folder_path = './data'

file_path1 = os.path.join(folder_path, 'desired_poses.npy')
desired_poses = np.load(file_path1)
# real_pos
file_path2 = os.path.join(folder_path, 'real_pos.npy')
real_pos = np.load(file_path2)
# total_time
file_path3 = os.path.join(folder_path, 'total_time.npy')
total_time = np.load(file_path3)
# total_time_step_num
file_path4 = os.path.join(folder_path, 'total_time_step_num.npy')
total_time_step_num = np.load(file_path4)
# mark
file_path5 = os.path.join(folder_path, 'mark.npy')
mark = np.load(file_path5)
# mark_r
file_path6 = os.path.join(folder_path, 'mark_r.npy')
mark_r = np.load(file_path6)

file_path7 = os.path.join(folder_path, 'hand tras.npy')
hand_trans = np.load(file_path7)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # while True:
    while viewer.is_running() and data.time <= total_time:
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],
            pos=[mark[0] - 0.4, mark[1]+0.2, mark[2] + 0.],
            mat=np.eye(3).flatten(),
            rgba=[1, 0.5, 0.5, 1]
        )
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],
            pos=[mark_r[0], mark_r[1], mark_r[2] + 0.],
            mat=np.eye(3).flatten(),
            rgba=[0.5, 1, 0.5, 1]
        )
        viewer.user_scn.ngeom = 2
        time_num += 1

        if time_num >= total_time_step_num:
            break

        desired_pos = desired_poses[time_num, :].copy()
        sensor_data = data.sensordata.copy()
        real_pos_prev = real_pos.copy()
        real_pos = sensor_data.copy()
        real_vel = (real_pos - real_pos_prev) / model.opt.timestep

        error_pos = desired_pos - sensor_data

        pos_out = [position_controllers[i].control(error_pos[i]) for i in range(20)]
        for i in range(20):
            data.ctrl[i] = velocity_controllers[i].control(pos_out[i] - real_vel[i])

        mujoco.mj_step(model, data)
        viewer.sync()

# mujoco.mj_resetDataKeyframe(model, data, 0)
# mujoco.mj_forward(model, data)
# # Recording a video
# n_frames = int(20 * total_time)+1
# height = 200
# width = 300
# frames = []
#
# with mujoco.Renderer(model, height, width) as renderer:
#   for i in range(n_frames):
#     while data.time < i/20.0:
#         desired_pos = desired_poses[time_num, :].copy()
#         sensor_data = data.sensordata.copy()
#         real_pos_prev = real_pos.copy()
#         real_pos = sensor_data.copy()
#         real_vel = (real_pos - real_pos_prev) / model.opt.timestep
#
#         error_pos = desired_pos - sensor_data
#
#         pos_out = [position_controllers[j].control(error_pos[j]) for j in range(model.nv - 16)]
#         for j in range(model.nv - 16):
#             data.ctrl[j] = velocity_controllers[j].control(pos_out[j] - real_vel[j])
#
#         mujoco.mj_step(model, data)
#
#     renderer.update_scene(data)
#     frame = renderer.render()
#     frames.append(frame)
#
# media.write_video('./data/render.mp4', frames, fps=20)

