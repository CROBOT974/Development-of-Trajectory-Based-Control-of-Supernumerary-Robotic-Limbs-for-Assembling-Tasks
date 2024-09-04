from unittest import TestCase

import mujoco
import mujoco.viewer
import numpy as np
from pid_control.src.controller import PIDController
from matplotlib import pyplot as plt

import os

import time
from os.path import dirname, abspath
from ikpy.chain import Chain

print(os.listdir(os.getcwd()))


class TestPIDControl(TestCase):

    def test_pid_control(self):

        model = mujoco.MjModel.from_xml_path("D:/dissertation/Mujoco/model/kortex/mjmodel.xml")
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)

        kp, ki, kd = 1.0, 0.0, 0.0

        position_controllers = [PIDController(kp, ki, kd, ts=model.opt.timestep) for _ in range(8)]
        velocity_controllers = [PIDController(kp, ki, kd, ts=model.opt.timestep) for _ in range(8)]

        kp_p_v = [[30, 40, 100, 75, 50, 50, 50, 50], [45, 90, 25, 22, 20, 10, 10, 10]]
        ki_p_v = [[0,  0,  0,   0,  0,  0,  0,  0], [20, 70, 240, 110, 110, 220, 220, 220]]
        kd_p_v = [[0,  0,  0,   0,  0,  0,  0,  0], [0, 0, 0, 0, 0, 0, 0, 0]]
        for i in range(8):
            position_controllers[i].set_parameter(kp_p_v[0][i], ki_p_v[0][i], kd_p_v[0][i])
            velocity_controllers[i].set_parameter(kp_p_v[1][i], ki_p_v[1][i], kd_p_v[1][i])

        time = 5
        time_step_num = round(time / model.opt.timestep) + 1  # 时间的步进次数
        desired_poses = np.zeros((time_step_num, model.nv - 3))
        desired_poses[:] = data.qpos[:8]
        print(data.qpos[:8])
        desired_vels = np.zeros_like(desired_poses)

        real_poses = np.zeros((time_step_num, model.nv - 3))
        real_pos_prevs = np.zeros_like(real_poses)
        sensor_datas = np.zeros_like(real_poses)  # ？
        real_vels = np.zeros_like(real_poses)
        motor_ctrls = np.zeros_like(real_poses)
        error_poses = np.zeros_like(real_poses)
        times = np.linspace(0, time, time_step_num)  # 将时间列出

        initial_state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_PHYSICS))
        initial_state[3] = 0  # np.pi / 2

        target_pos = [0, -0.4, 0.6]
        target_pos[0] = 1e-3 if target_pos[0] == 0 else target_pos[0]  # avoid the singularity

        my_chain = Chain.from_urdf_file("D:/dissertation/Mujoco/model/kortex/GEN3_URDF_V12.urdf",
                                        active_links_mask=[False, True, True, True, True, True, True, True, False])

        target_matrix = np.eye(4)
        target_matrix[: 3, 3] = target_pos
        ik = my_chain.inverse_kinematics_frame(target_matrix,
                                               initial_position=np.append(np.append(np.array(0), data.qpos[:7]),
                                                                          np.array(0)), orientation_mode='all')[1:8]
        print('ik:', ik)

        time_num = time / model.opt.timestep
        print(time_num)
        for i, timei in enumerate(times):
            for j, _ in enumerate(ik):
                desired_poses[i, j] += i * (ik[j] - data.qpos[j]) / time_step_num
            desired_poses[i, -1] += i * -0.3 / time_step_num
            # for j, _ in enumerate(ik):
            #     desired_poses[i, j] = i * (ik[j]-data.qpos[j]) / time_num
            # desired_poses[i, -1] += i * -0.3 / time_num
            # desired_poses[i, :7] = desired_poses[i, :7] + data.qpos[:7]

            if i > 0:
                desired_vels[i, :] = (desired_poses[i, :] - desired_poses[i - 1, :]) / model.opt.timestep

        # 增加对gripper的控制
        # zeros_column = np.zeros((desired_poses.shape[0], 1))
        # desired_poses = np.hstack((desired_poses, zeros_column))
        # desired_vels = np.hstack((desired_vels, zeros_column))

        # print(i)
        print("desired_poses: ", desired_poses[-1])

        # 输入的是从起始点到终点的一系列的角坐标集合，数量与时间相关，可用贝塞尔差值
        time_num = 0

        # 重新设置机械臂的姿态
        # mujoco.mj_resetData(model, data)
        # mujoco.mj_setState(model, data, initial_state, mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_forward(model, data)  # 似乎没有影响

        sensor_data = data.sensordata.copy()  # position(旋转弧度)
        real_pos = sensor_data.copy()
        real_pos_prev = real_pos.copy()
        real_vel = (real_pos - real_pos_prev) / model.opt.timestep
        sensor_datas[time_num, :] = sensor_data.copy()
        real_poses[time_num, :] = real_pos.copy()
        real_pos_prevs[time_num, :] = real_pos_prev.copy()
        real_vels[time_num, :] = real_vel.copy()
        motor_ctrls[time_num, :] = data.ctrl.copy()

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= time:
                # print(data.time)
                time_num += 1

                if time_num >= time_step_num:
                    break

                desired_pos = desired_poses[time_num, :].copy()
                desired_vel = desired_vels[time_num, :].copy()
                sensor_data = data.sensordata.copy()
                real_pos_prev = real_pos.copy()
                real_pos = sensor_data.copy()
                real_vel = (real_pos - real_pos_prev) / model.opt.timestep

                error_pos = desired_pos - sensor_data

                pos_out = [position_controllers[i].control(error_pos[i]) for i in range(model.nv - 3)]
                for i in range(model.nv - 3):
                    data.ctrl[i] = velocity_controllers[i].control(pos_out[i] - real_vel[i])
                # data.ctrl[axis_id] = velocity_controllers[axis_id].control(desired_vel[axis_id] - real_vel[axis_id])

                sensor_datas[time_num, :] = sensor_data.copy()
                real_poses[time_num, :] = real_pos.copy()
                real_pos_prevs[time_num, :] = real_pos_prev.copy()
                real_vels[time_num, :] = real_vel.copy()
                motor_ctrls[time_num, :] = data.ctrl.copy()
                error_poses[time_num, :] = error_pos.copy()

                mujoco.mj_step(model, data)

                # with viewer.lock():
                #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

                # add a marker
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.02, 0, 0],
                    pos=target_pos,
                    mat=np.eye(3).flatten(),
                    rgba=[1, 0.5, 0.5, 1]
                )
                viewer.user_scn.ngeom = 1

                viewer.sync()
        #
        # plt.figure(1)
        # plt.plot(times, desired_poses[:, axis_id], '-', label='desired position')
        # plt.plot(times, sensor_datas[:, axis_id], '--', label='actual position')
        # plt.legend()
        # plt.tight_layout()
        #
        # plt.figure(2)
        # plt.plot(times, desired_vels[:, axis_id], '-', label='desired velocity')
        # plt.plot(times, real_vels[:, axis_id], '--', label='actual velocity')
        # plt.legend()
        # plt.tight_layout()
        #
        # plt.figure(3)
        # plt.plot(times, motor_ctrls[:, axis_id], '-', label='motor torque')
        # plt.legend()
        # plt.tight_layout()
        #
        # plt.figure(4)
        # plt.plot(times, error_poses[:, axis_id], '-', label='position error')
        # plt.legend()
        # plt.tight_layout()
        #
        # plt.show()
