import os
import numpy as np
import mujoco.viewer
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from src.controller import PIDController
from src.motion_planning.RRTProcess import TrajectoryProcess
from src.controller.imp_controller.impedance import Imp_Controller


def ik_tracing(chain, init_pose, init_ori, degree, trajectory):
    trajectory_pos = []
    tar_matrix = np.eye(4)
    pose = init_pose

    # determined the continuous orientation by Slerp
    init_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    init_rpy = init_orientation.as_euler('xyz')
    rpy_array = np.array([[init_rpy[0], init_rpy[1], init_rpy[2]],
                          [init_rpy[0] - degree / 2, init_rpy[1], init_rpy[2]],
                          [init_rpy[0] + degree, init_rpy[1], init_rpy[2]]]
                         )
    quaternions = [Rotation.from_euler('xyz', rpy_array[0]),
                   Rotation.from_euler('xyz', rpy_array[1]),
                   Rotation.from_euler('xyz', rpy_array[2])]
    key_rots = Rotation.random(len(quaternions))
    for i in range(quaternions.__len__()):
        key_rots[i] = quaternions[i]
    key_times = [0, 2, 4]
    slerp = Slerp(key_times, key_rots)
    times = np.linspace(0, 4, len(trajectory))
    interp_rots = slerp(times)

    for j in range(len(trajectory)):
        target_pos = trajectory[j]
        end_orientation = np.dot(init_ori, interp_rots[j].as_matrix())
        tar_matrix[:3, 3] = target_pos
        tar_matrix[:3, :3] = end_orientation

        pose = chain.inverse_kinematics_frame(tar_matrix, initial_position=np.append(np.array(0), pose),
                                              orientation_mode='all')
        pose = pose[1:]
        trajectory_pos.append(pose)
    output_ori = np.dot(init_ori, interp_rots[-1].as_matrix())
    return trajectory_pos, output_ori


model = mujoco.MjModel.from_xml_path("model/scene3.xml")
data = mujoco.MjData(model)

# ID of all sites
squat_p = int(model.site('knee_site').id)  # 0
stand_p = int(model.site('stand_site').id)  # 1
rhand_p = int(model.site('hand_site_r').id)  # 2
lhand_p = int(model.site('hand_site_l').id)  # 3
main_p = int(model.site('main_site').id)  # 4
m_tip_p = int(model.site('m_tip_site').id)  # 5
slave_p = int(model.site('slave_site').id)  # 6
s_tip_p = int(model.site('s_tip_site').id)  # 7

mujoco.mj_resetDataKeyframe(model, data, 3)  # squat, hand down, to get the init_pos of tube
mujoco.mj_forward(model, data)
armr_sq_down_xpos = data.site_xpos[rhand_p].copy()
mujoco.mj_resetDataKeyframe(model, data, 2)  # stand, hold the tube
mujoco.mj_forward(model, data)
init_lhand_pos = data.site_xpos[lhand_p].copy()  # left hand
armr_hold_qpos = data.qpos[5:8].copy()
arml_hold_qpos = data.qpos[8:11].copy()
mujoco.mj_resetDataKeyframe(model, data, 1)  # stand, arm up, to calculate the trajectory of hand
mujoco.mj_forward(model, data)
init_base_stand = data.site_xpos[main_p].copy()
init_base_r_stand = data.site_xpos[slave_p].copy()
armr_up_qpos = data.qpos[5:8].copy()
arml_up_qpos = data.qpos[8:11].copy()

"""------------------------------------------PID Controll---------------------------------------------"""
# determine the parameters for PID
kp, ki, kd = 1.0, 0.0, 0.0

position_controllers = [PIDController(kp, ki, kd, ts=model.opt.timestep) for _ in range(23)]
velocity_controllers = [PIDController(kp, ki, kd, ts=model.opt.timestep) for _ in range(23)]

kp_p_v = [[2, 2, 2, 2, 2, 2, 2, 30, 40, 100, 75, 50, 50, 50, 50, 30, 40, 100, 75, 50, 50, 50, 50],  # position kp-value
          [1, 1, 1, 1, 1, 1, 1, 45, 90, 25, 22, 20, 10, 10, 10, 45, 90, 25, 22, 20, 10, 10, 10, ]]  # velocity kp-value
ki_p_v = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [2, 2, 2, 2, 2, 2, 2, 20, 70, 240, 110, 110, 22, 22, 22, 20, 70, 240, 110, 110, 22, 22, 22]]
kd_p_v = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

for i in range(23):
    position_controllers[i].set_parameter(kp_p_v[0][i], ki_p_v[0][i], kd_p_v[0][i])
    velocity_controllers[i].set_parameter(kp_p_v[1][i], ki_p_v[1][i], kd_p_v[1][i])
"""---------------------------------------------------------------------------------------------------"""

"""------------------------------------------IMP Controll---------------------------------------------"""
mass_matrix = np.zeros((model.nv, model.nv), dtype=np.float64)
mujoco.mj_fullM(model, mass_matrix, data.qM)
mass_matrix_arm_1 = mass_matrix[11:18,11:18].copy()
mass_matrix_arm_2 = mass_matrix[22:29,22:29].copy()
mass_matrix_arm_1[-1, -1] *= 5.0
mass_matrix_arm_2[-1, -1] *= 5.0

B = 20.0 * np.ones(7)
K = 100.0 * np.ones(7)
Impedence1 = Imp_Controller(K ,B ,mass_matrix_arm_1)
Impedence2 = Imp_Controller(K ,B ,mass_matrix_arm_2)
"""---------------------------------------------------------------------------------------------------"""

# chain info of main and secondary robot arms
my_chain = Chain.from_urdf_file("model/45degree/GEN3_URDF_main.urdf",
                                active_links_mask=[False, True, True, True, True, True, True, True, False])
my_chain_r = Chain.from_urdf_file("model/45degree/GEN3_URDF_slave.urdf",
                                  active_links_mask=[False, True, True, True, True, True, True, True, False])

mujoco.mj_resetDataKeyframe(model, data, 3)  # squat, hand down
mujoco.mj_forward(model, data)

init_pos = data.xpos[27]
init_pos_r = data.xpos[-2]  # to calculate the link_info, the main arm needs offset
init_matrix = data.xmat[27].reshape(3, 3).copy()  # init ori of end effector
init_matrix_r = data.xmat[-2].reshape(3, 3).copy()
init_qpos = data.qpos[11:18]
init_qpos_r = data.qpos[22:29]

mark = init_pos.copy()
mark_r = init_pos_r.copy()
obstacle_squat_p = data.site_xpos[squat_p].copy()  # safety area when squat
obstacle_squat_size = [0.25, 0.22, 0.44]
obstacle_stand_p = data.site_xpos[stand_p].copy()  # safety area when stand
obstacle_stand_size = [0.19, 0.22, 0.75]
init_base = data.site_xpos[main_p].copy()
init_base_r = data.site_xpos[slave_p].copy()
print("init_base", init_base)
print("init_base_r", init_base_r)

# since the init base of robot chain is [0,0,0], we need to minus the current init base
init_pos = init_pos - init_base
init_pos_r = init_pos_r - init_base_r
print("ini_pos", init_pos, init_pos_r)

# determine the end position for first tra_planning (above the tube)
# hand_point = np.load('data/hand tras.npy')
end_pos = np.array([armr_sq_down_xpos[0] - 1, armr_sq_down_xpos[1], armr_sq_down_xpos[2] + 0.1])
end_pos = end_pos - init_base
end_pos_r = init_pos_r
end_ori = [0, -90, 35]
end_ori_r = [0, 0, 0]

# RRT_Connect method for the first trajectory planning
print("Trajectory Planning ......")
RRT = TrajectoryProcess(my_chain, my_chain_r,
                        obstacle_squat_p, obstacle_squat_size,
                        init_pos, init_pos_r,  # init position of end effector in Cartesian Space
                        end_pos, end_pos_r,  # target position of end effector in Cartesian Space
                        init_matrix, init_matrix_r,  # init orientation of end effector
                        end_ori, end_ori_r,  # target orientation of end effector
                        init_qpos, init_qpos_r)  # init joint angles
control1, control2, tar_mat, _ = RRT.rrt_connect()  # control1,2 has 8 elements, include 1 for gripper

# determine the end position for first tra_planning (grab the tube)
# tar_mat[:3, 3] = end_pos
tar_mat[2, 3] -= 0.1
ik_joint = my_chain.inverse_kinematics_frame(tar_mat, initial_position=np.append(np.array(0), control1[-1]),
                                             orientation_mode='all')
control3 = ik_joint[1:]

# Trajectory Tracing
trajectory1 = np.load('data/hand_tras1.npy')
trajectory1 = trajectory1 - init_base_stand
trajectory2 = np.load('data/hand_tras2.npy')
trajectory2 = trajectory2 - init_base_stand
tracing, out_ori = ik_tracing(my_chain, control3, tar_mat[:3, :3], (np.pi * 7 / 9), trajectory1)
tracing2, out_ori2 = ik_tracing(my_chain, tracing[-1], out_ori, 0, trajectory2)
tracing = np.vstack((tracing, tracing[-1]))
tracing2 = np.vstack((tracing2, tracing2[-1]))
f1 = tracing[-1]
f2 = tracing2[-1]

mujoco.mj_resetDataKeyframe(model, data, 2)  # stand, hold the tube
mujoco.mj_forward(model, data)

init_pos_r2 = data.xpos[-2]
init_matrix2 = out_ori2.copy()
init_matrix_r2 = data.xmat[-2].reshape(3, 3).copy()
init_qpos2 = tracing2[-1][:-1]
init_qpos_r2 = data.qpos[22:29]

init_base2 = data.site_xpos[main_p].copy()
init_base_r2 = data.site_xpos[slave_p].copy()

init_pos2 = trajectory1[-1]
init_pos_r2 = init_pos_r2 - init_base_r_stand

end_pos2 = init_pos2
end_pos_r2 = np.array([init_lhand_pos[0] + 0.14, init_lhand_pos[1] + 0.09, init_lhand_pos[2] + 0.09])
end_pos_r2 = end_pos_r2 - init_base_r_stand

end_ori = [0, 0, 0]
end_ori_r = [0, -90, -90]

# RRT_Connect method for the second trajectory planning
print("Trajectory Planning ......")
RRT2 = TrajectoryProcess(my_chain, my_chain_r,
                         obstacle_stand_p, obstacle_stand_size,
                         init_pos2, init_pos_r2,
                         end_pos2, end_pos_r2,
                         init_matrix2, init_matrix_r2,
                         end_ori, end_ori_r,
                         init_qpos2, init_qpos_r2)
control2_1, control2_2, _, tar_mat2 = RRT2.rrt_connect()

# 最后的抓取
end_pos_r3 = np.array([init_lhand_pos[0] + 0.19, init_lhand_pos[1], init_lhand_pos[2]])
end_pos_r3 = end_pos_r3 - init_base_r_stand
tar_mat2[:3, 3] = end_pos_r3
ik_joint2 = my_chain_r.inverse_kinematics_frame(tar_mat2, initial_position=np.append(np.array(0), control2_2[-1]),
                                                orientation_mode='all')
control4 = ik_joint2[1:]

# set up the time sequence
steps = 11
time_step_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
time = np.zeros(steps)

time[0] = 10
time_step_num[0] = round(time[0] / model.opt.timestep) + 1
time[1] = 10
time_step_num[1] = round(time[1] / model.opt.timestep) + 1
time_step_num[2] = len(control1)
time[2] = model.opt.timestep * (time_step_num[2] - 1)
time[3] = 5
time_step_num[3] = round(time[3] / model.opt.timestep) + 1
time[4] = 10
time_step_num[4] = round(time[4] / model.opt.timestep) + 1
time[5] = 20
time_step_num[5] = round(time[5] / model.opt.timestep) + 1
time[6] = 10
time_step_num[6] = round(time[6] / model.opt.timestep) + 1
time[7] = 10
time_step_num[7] = round(time[7] / model.opt.timestep) + 1
time_step_num[8] = len(control2_2)
time[8] = model.opt.timestep * (time_step_num[8] - 1)
time[9] = 5
time_step_num[9] = round(time[9] / model.opt.timestep) + 1
time[10] = 10
time_step_num[10] = round(time[10] / model.opt.timestep) + 1
total_time = sum(time)
total_time_step_num = sum(time_step_num)

times = []
for i in range(steps):
    times_list = np.linspace(0, time[i], time_step_num[i])  # list the time step
    times.append(times_list)

# determined the grab motion
grab = np.concatenate([np.linspace(0, -0.9, int(time_step_num[3] / 2)),
                       np.linspace(-0.9, -0.5, int(time_step_num[3] / 2) + 1)])  # main arm
grab2 = np.concatenate([np.linspace(0, -0.9, int(time_step_num[9] / 2)),
                        np.linspace(-0.9, -0.5, int(time_step_num[9] / 2) + 1)])  # secondary arm

# determine the desired value for all actuator due to time step
desired_poses = np.zeros((total_time_step_num, 23))

mujoco.mj_resetDataKeyframe(model, data, 0)  # stand, hands down, the init pose
mujoco.mj_forward(model, data)

for h, _ in enumerate(times[0]):  # no motion, stand by
    num = h
    desired_poses[num, 0] = 0.2  # keep stand
    desired_poses[num, 1:7] = data.qpos[5:11]  # no motion for arms
    desired_poses[num, 7:15] = data.qpos[11:19]  # keep the position of the m_arm
    desired_poses[num, 15:] = data.qpos[22:30]  # keep the position of the s_arm
for i, _ in enumerate(times[1]):  # squat
    num = h + 1 + i
    desired_poses[num, :] = desired_poses[num - 1, :]  # keep the init value
    desired_poses[num, 0] = 0.2 + (i * -0.87 / time_step_num[1])

for j, _ in enumerate(times[2]):  # 'move' for arm and 'tra_planning' for robot
    num = h + i + 1 + j
    desired_poses[num, :] = desired_poses[num - 1, :]
    desired_poses[num, 7:15] = control1[j]
    desired_poses[num, 15:] = control2[j]

for k, _ in enumerate(times[3]):  # robot arm moves down and grabs the tube
    num = h + i + j + 1 + k
    desired_poses[num, :] = desired_poses[num - 1, :]
    desired_poses[num, 7:15] = control1[-1] + k * (control3 - control1[-1]) / time_step_num[3]
    desired_poses[num, 14] = grab[k]

for l, _ in enumerate(times[4]):  # stand up
    num = h + i + j + k + 1 + l
    desired_poses[num, :] = desired_poses[num - 1, :]
    # 人站立
    desired_poses[num, 0] = -0.67 + (l * 0.87 / time_step_num[4])

for m, _ in enumerate(times[5]):  # rise the humane arm, "tra_tracing" for robot
    num = h + i + j + k + l + 1 + m
    desired_poses[num, :] = desired_poses[num - 1, :]
    # 右臂抬起
    desired_poses[num, 1:4] = data.qpos[5:8] + m * ((armr_up_qpos[:] - data.qpos[5:8]) / time_step_num[5])
    desired_poses[num, 4:7] = data.qpos[8:11] + m * ((arml_up_qpos[:] - data.qpos[8:11]) / time_step_num[5])
    desired_poses[num, 7:15] = tracing[m]
    desired_poses[num, 14] = grab[-1]

for n, _ in enumerate(times[6]):  # hand exchange
    num = h + i + j + k + l + m + 1 + n
    desired_poses[num, :] = desired_poses[num - 1, :]
    desired_poses[num, 7:15] = tracing[-1] + n * ((tracing2[0] - tracing[-1]) / time_step_num[6])
    desired_poses[num, 14] = grab[-1]

for o, _ in enumerate(times[7]):  # left hand reaches the correct position
    num = h + i + j + k + l + m + n + 1 + o
    desired_poses[num, :] = desired_poses[num - 1, :]
    # 右臂抬起
    desired_poses[num, 1:4] = armr_up_qpos[:] + o * ((armr_hold_qpos[:] - armr_up_qpos[:]) / time_step_num[7])
    desired_poses[num, 4:7] = arml_up_qpos[:] + o * ((arml_hold_qpos[:] - arml_up_qpos[:]) / time_step_num[7])
    desired_poses[num, 7:15] = tracing2[o]
    desired_poses[num, 14] = grab[-1]

for p, _ in enumerate(times[8]):  # humanoid waits, 'tra_planning' for robot
    num = h + i + j + k + l + m + n + o + 1 + p
    desired_poses[num, :] = desired_poses[num - 1, :]
    desired_poses[num, 7:15] = control2_1[p]
    desired_poses[num, 14] = grab[-1]
    desired_poses[num, 15:] = control2_2[p]

for q, _ in enumerate(times[9]):  # robot arm moves up and grabs the tube
    num = h + i + j + k + l + m + n + o + p + 1 + q
    desired_poses[num, :] = desired_poses[num - 1, :]
    desired_poses[num, 15:] = control2_2[-1] + q * (control4 - control2_2[-1]) / time_step_num[9]
    desired_poses[num, 22] = grab2[q]

for r, _ in enumerate(times[10]):  # hands down
    num = h + i + j + k + l + m + n + o + p + q + 1 + r
    desired_poses[num, :] = desired_poses[num - 1, :]
    desired_poses[num, 4:7] = arml_hold_qpos[:] + r * ((data.qpos[8:11] - arml_hold_qpos[:]) / time_step_num[10])


# Save the Data
real_pos = data.sensordata.copy()  # init joint angles
desired_pos = desired_poses[0, :].copy()

folder_path = 'D:/PythonProject/Development-of-Trajectory-Based-Control-of-Supernumerary-Robotic-Limbs-for-Assembling-Tasks-main/data'
os.makedirs(folder_path, exist_ok=True)
# desired_poses
file_path1 = os.path.join(folder_path, 'desired_poses.npy')
np.save(file_path1, desired_poses)
# real_pos
file_path2 = os.path.join(folder_path, 'real_pos.npy')
np.save(file_path2, real_pos)
# total_time
file_path3 = os.path.join(folder_path, 'total_time.npy')
np.save(file_path3, total_time)
# total_time_step_num
file_path4 = os.path.join(folder_path, 'total_time_step_num.npy')
np.save(file_path4, total_time_step_num)
# mark
file_path5 = os.path.join(folder_path, 'mark.npy')
np.save(file_path5, mark)
# mark_r
file_path6 = os.path.join(folder_path, 'mark_r.npy')
np.save(file_path6, mark_r)

time_num = 0
end_tras1 = []
end_tras2 = []
robot_tras1 = []
robot_tras2 = []
robot_tras3 = []
rhand_tras = []
lhand_tras = []
desire = []
sensor = []
pid_ctrl = np.zeros(23)
test1 = 0
test2 = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    # while True:
    while viewer.is_running() and data.time <= total_time:
        # target point of first tra_planning
        if total_time - time[3] - time[4] - time[5] - time[6] - time[7] - time[8] - time[9] - time[
            10] > data.time >= total_time - \
                time[2] - time[3] - time[4] - time[5] - time[6] - time[7] - time[8] - time[9] - time[10]:
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.04, 0, 0],
                pos=[armr_sq_down_xpos[0] - 1, armr_sq_down_xpos[1], armr_sq_down_xpos[2]],
                mat=np.eye(3).flatten(),
                rgba=[1, 0.5, 0.5, 1]
            )
        # target point of second tra_planning
        if total_time - time[9] - time[10] > data.time >= total_time - time[8] - time[9] - time[10]:
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.04, 0, 0],
                pos=[init_lhand_pos[0] + 0.19, init_lhand_pos[1], init_lhand_pos[2]],
                mat=np.eye(3).flatten(),
                rgba=[1, 0.5, 0.5, 1]
            )
        time_num += 1

        if time_num >= total_time_step_num:
            break

        desired_pos_prev = desired_pos.copy()

        desired_pos = desired_poses[time_num, :].copy()
        sensor_data = data.sensordata.copy()
        real_pos_prev = real_pos.copy()
        real_pos = sensor_data.copy()
        real_vel = (real_pos - real_pos_prev) / model.opt.timestep
        error_pos = desired_pos - sensor_data

        desired_vel = (desired_pos - desired_pos_prev) / model.opt.timestep
        qfrc_bias = data.qfrc_bias.copy()

        # PID control
        pos_out = [position_controllers[i].control(error_pos[i]) for i in range(23)]

        ctrl1 = Impedence1.control(desired_pos[7:14], desired_vel[7:14], data.qpos[11:18].copy(),
                                   data.qvel[11:18].copy(), qfrc_bias[11:18])
        # print("ctrl1",ctrl1[0])
        ctrl2 = Impedence2.control(desired_pos[15:22], desired_vel[15:22], data.qpos[22:29].copy(),
                                   data.qvel[22:29].copy(), qfrc_bias[22:29])
        # print("ctrl2", ctrl2)

        pid_ctrl = [velocity_controllers[i].control(pos_out[i] - real_vel[i]) for i in range(23)]
        # data.ctrl = pid_ctrl
        for i in range(23):
            if 6 < i < 14:
                pid_ctrl[i] = ctrl1[i-7]
            elif 22 > i > 14:
                pid_ctrl[i] = ctrl2[i-15]


        data.ctrl = pid_ctrl

        # showing the tube
        if total_time - time[6] - time[7] - time[8] - time[9] - time[10] > data.time >= total_time - time[3] - time[4] - \
                time[5] - time[6] - time[7] - time[8] - time[9] - time[10]:
            updating_tube = data.site_xpos[rhand_p]

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[2],
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.04, 1.25, 0],
                pos=[updating_tube[0] - 0.8, updating_tube[1], updating_tube[2]],
                mat=np.array([[0, 0, 1],
                              [0, 1, 0],
                              [-1, 0, 0]]).flatten(),
                rgba=[0.7, 0.7, 0.7, 0.4]
            )

        elif total_time - time[8] - time[9] - time[10] > data.time >= total_time - time[6] - time[7] - time[8] - time[9] - time[10]:
            updating_tube2 = data.site_xpos[lhand_p].copy()

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[2],
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.04, 1.25, 0],
                pos=[updating_tube2[0] - 0.8, updating_tube2[1], updating_tube2[2]],
                mat=np.array([[0, 0, 1],
                              [0, 1, 0],
                              [-1, 0, 0]]).flatten(),
                rgba=[0.7, 0.7, 0.7, 0.4]
            )
        elif data.time >= total_time - time[8] - time[9] - time[10]:

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[2],
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.04, 1.25, 0],
                pos=[updating_tube2[0] - 0.8, updating_tube2[1], updating_tube2[2]],
                mat=np.array([[0, 0, 1],
                              [0, 1, 0],
                              [-1, 0, 0]]).flatten(),
                rgba=[0.7, 0.7, 0.7, 0.4]
            )
        else:
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[2],
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.04, 1.25, 0],
                pos=[armr_sq_down_xpos[0] - 0.8, armr_sq_down_xpos[1], armr_sq_down_xpos[2]],
                mat=np.array([[0, 0, 1],
                              [0, 1, 0],
                              [-1, 0, 0]]).flatten(),
                rgba=[0.7, 0.7, 0.7, 0.4]
            )

        # record the trajectory of human hands
        if total_time - time[6] - time[7] - time[8] - time[9] > data.time >= total_time - time[5] - time[6] - time[7] \
                - time[8] - time[9]:
            end_tra = data.site_xpos[rhand_p].copy()
            end_tra[0] -= 1
            end_tras1.append(end_tra)
        elif total_time - time[8] - time[9] > data.time >= total_time - time[7] - time[8] - time[9]:
            end_tra = data.site_xpos[lhand_p].copy()
            end_tra[0] -= 1
            end_tras2.append(end_tra)
        # record the data for graphs
        if total_time - time[6] - time[7] - time[8] - time[9] - time[10] > data.time >= total_time - time[5] - time[6] \
                - time[7] - time[8] - time[9] - time[10]:
            robot_tracing = data.site_xpos[m_tip_p]
            robot_tras1.append(robot_tracing.copy())
        if total_time - time[7] - time[8] - time[9] - time[10] > data.time >= total_time - time[6] - time[7] - time[8] \
                - time[9] - time[10]:
            robot_tracing = data.site_xpos[m_tip_p]
            robot_tras2.append(robot_tracing.copy())
        if total_time - time[8] - time[9] - time[10] > data.time >= total_time - time[7] - time[8] - time[9] - time[10]:
            robot_tracing = data.site_xpos[m_tip_p]
            robot_tras3.append(robot_tracing.copy())
        if total_time - time[6] - time[7] - time[8] - time[9] - time[10] > data.time >= total_time - time[5] - time[6] \
                - time[7] - time[8] - time[9] - time[10]:
            rhand_tra = data.site_xpos[rhand_p]
            rhand_tra[0] -= 1
            rhand_tras.append(rhand_tra.copy())
        if total_time - time[8] - time[9] - time[10] > data.time >= total_time - time[7] - time[8] - time[9] - time[10]:
            lhand_tra = data.site_xpos[lhand_p].copy()
            lhand_tra[0] -= 1
            lhand_tras.append(lhand_tra)
        if total_time - time[8] - time[9] - time[10] > data.time >= total_time - time[2] - time[3] - time[4] - time[5] \
                - time[6] - time[7] - time[8] - time[9] - time[10]:
            desire.append(desired_pos[7:15].copy())
            sensor.append(sensor_data[7:15].copy())

        viewer.user_scn.ngeom = 4
        mujoco.mj_step(model, data)
        viewer.sync()

# # pre-determine the hands trajectory for tracing(pay attention if you change the human's motion)
# file_path7 = os.path.join(folder_path, 'hand_tras1.npy')
# np.save(file_path7, end_tras1)
# file_path8 = os.path.join(folder_path, 'hand_tras2.npy')
# np.save(file_path8, end_tras2)

file_path9 = os.path.join(folder_path, 'rhand_tras.npy')
np.save(file_path9, rhand_tras)
file_path10 = os.path.join(folder_path, 'lhand_tras.npy')
np.save(file_path10, lhand_tras)
file_path11 = os.path.join(folder_path, 'robot_tras1.npy')
np.save(file_path11, robot_tras1)
file_path12 = os.path.join(folder_path, 'robot_tras2.npy')
np.save(file_path12, robot_tras2)
file_path13 = os.path.join(folder_path, 'robot_tras3.npy')
np.save(file_path13, robot_tras3)
file_path14 = os.path.join(folder_path, 'desire.npy')
np.save(file_path14, desire)
file_path15 = os.path.join(folder_path, 'sensor.npy')
np.save(file_path15, sensor)
