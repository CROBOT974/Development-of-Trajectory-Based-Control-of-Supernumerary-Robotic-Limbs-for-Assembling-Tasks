from unittest import TestCase

import mujoco
import mujoco.viewer
import numpy as np

from ..src.controller import PIDController
from matplotlib import pyplot as plt

from ..src.robot import *
from ..src.motion_planning import *
from spatialmath import SO3

import os
print(os.listdir(os.getcwd()))
class TestPIDControl(TestCase):

    def test_pid_control(self):

        model = mujoco.MjModel.from_xml_path("pid_control/src/assets/universal_robots_ur5e/scene.xml")
        data = mujoco.MjData(model)

        robot = UR5e()
        q0 = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0])
        robot.set_joint(q0)

        kp = 1.0
        ki = 0.0
        kd = 0.0

        position_controllers = [PIDController(kp, ki, kd, ts=model.opt.timestep) for _ in range(6)]
        velocity_controllers = [PIDController(kp, ki, kd, ts=model.opt.timestep) for _ in range(6)]

        axis_id = 1

        position_controllers[0].set_parameter(50.0, 0.0, 0.0)
        position_controllers[1].set_parameter(40.0, 0.0, 0.0)
        position_controllers[2].set_parameter(100.0, 0.0, 0.0)
        position_controllers[3].set_parameter(75.0, 0.0, 0.0)
        position_controllers[4].set_parameter(50.0, 0.0, 0.0)
        position_controllers[5].set_parameter(50.0, 0.0, 0.0)

        velocity_controllers[0].set_parameter(45.0, 20.0, 0.0)
        velocity_controllers[1].set_parameter(90.0, 70.0, 0.0)
        velocity_controllers[2].set_parameter(50.0, 240.0, 0.0)
        velocity_controllers[3].set_parameter(22.0, 110.0, 0.0)
        velocity_controllers[4].set_parameter(22.0, 110.0, 0.0)
        velocity_controllers[5].set_parameter(22.0, 220.0, 0.0)

        total_time = 10
        time_step_num = round(total_time / model.opt.timestep) + 1 # 时间的步进次数
        desired_poses = np.zeros((time_step_num, robot.dof))
        desired_vels = np.zeros_like(desired_poses)
        real_poses = np.zeros_like(desired_poses)
        real_pos_prevs = np.zeros_like(desired_poses)
        sensor_datas = np.zeros_like(desired_poses) # ？
        real_vels = np.zeros_like(desired_poses)
        motor_ctrls = np.zeros_like(desired_poses)
        error_poses = np.zeros_like(desired_poses)
        times = np.linspace(0, total_time, time_step_num) # 将时间列出

        joint_position = np.zeros(robot.dof)
        for i, timei in enumerate(times):
            desired_poses[i, 0] = np.sin(timei)
            desired_poses[i, 2] = np.pi / 2
            if i > 0:
                desired_vels[i, :] = (desired_poses[i, :] - desired_poses[i - 1, :]) / model.opt.timestep
        print("desired_vels: ", desired_vels)

        time_num = 0

        # 重新设置机械臂的姿态
        mujoco.mj_resetData(model, data)
        initial_state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_PHYSICS))
        initial_state[2] = np.pi / 2
        mujoco.mj_setState(model, data, initial_state, mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_forward(model, data) # 似乎没有影响

        sensor_data = data.sensordata.copy()  #position(旋转弧度)
        real_pos = sensor_data.copy()
        real_pos_prev = real_pos.copy()
        real_vel = (real_pos - real_pos_prev) / model.opt.timestep

        sensor_datas[time_num, :] = sensor_data.copy()
        real_poses[time_num, :] = real_pos.copy()
        real_pos_prevs[time_num, :] = real_pos_prev.copy()
        real_vels[time_num, :] = real_vel.copy()
        motor_ctrls[time_num, :] = data.ctrl.copy()

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
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

                pos_out = [position_controllers[i].control(error_pos[i]) for i in range(robot.dof)]
                for i in range(robot.dof):
                    data.ctrl[i] = velocity_controllers[i].control(pos_out[i] - real_vel[i])
                # data.ctrl[axis_id] = velocity_controllers[axis_id].control(desired_vel[axis_id] - real_vel[axis_id])

                sensor_datas[time_num, :] = sensor_data.copy()
                real_poses[time_num, :] = real_pos.copy()
                real_pos_prevs[time_num, :] = real_pos_prev.copy()
                real_vels[time_num, :] = real_vel.copy()
                motor_ctrls[time_num, :] = data.ctrl.copy()
                error_poses[time_num, :] = error_pos.copy()

                mujoco.mj_step(model, data)

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

                viewer.sync()

        plt.figure(1)
        plt.plot(times, desired_poses[:, axis_id], '-', label='desired position')
        plt.plot(times, sensor_datas[:, axis_id], '--', label='actual position')
        plt.legend()
        plt.tight_layout()

        plt.figure(2)
        plt.plot(times, desired_vels[:, axis_id], '-', label='desired velocity')
        plt.plot(times, real_vels[:, axis_id], '--', label='actual velocity')
        plt.legend()
        plt.tight_layout()

        plt.figure(3)
        plt.plot(times, motor_ctrls[:, axis_id], '-', label='motor torque')
        plt.legend()
        plt.tight_layout()

        plt.figure(4)
        plt.plot(times, error_poses[:, axis_id], '-', label='position error')
        plt.legend()
        plt.tight_layout()

        plt.show()

    def test_robot_move(self):

        model = mujoco.MjModel.from_xml_path("pid_control/src/assets/universal_robots_ur5e/scene.xml")
        data = mujoco.MjData(model)

        robot = UR5e()
        q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot.set_joint(q0)

        kp = 1.0
        ki = 0.0
        kd = 0.0

        position_controllers = [PIDController(kp, ki, kd, ts=model.opt.timestep) for _ in range(6)]
        velocity_controllers = [PIDController(kp, ki, kd, ts=model.opt.timestep) for _ in range(6)]

        position_controllers[0].set_parameter(50.0, 0.0, 0.0)
        position_controllers[1].set_parameter(40.0, 0.0, 0.0)
        position_controllers[2].set_parameter(100.0, 0.0, 0.0)
        position_controllers[3].set_parameter(75.0, 0.0, 0.0)
        position_controllers[4].set_parameter(50.0, 0.0, 0.0)
        position_controllers[5].set_parameter(50.0, 0.0, 0.0)

        velocity_controllers[0].set_parameter(45.0, 20.0, 0.0)
        velocity_controllers[1].set_parameter(90.0, 70.0, 0.0)
        velocity_controllers[2].set_parameter(50.0, 240.0, 0.0)
        velocity_controllers[3].set_parameter(22.0, 110.0, 0.0)
        velocity_controllers[4].set_parameter(22.0, 110.0, 0.0)
        velocity_controllers[5].set_parameter(22.0, 220.0, 0.0)

        time1 = 5.0
        q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        joint_parameter1 = JointParameter(q0, q1)
        velocity_parameter1 = CubicVelocityParameter(time1)
        trajectory_parameter1 = TrajectoryParameter(joint_parameter1, velocity_parameter1)
        trajectory_planner1 = TrajectoryPlanner(trajectory_parameter1)

        time2 = 5.0
        robot.set_joint(q1)
        T1 = robot.get_cartesian()
        t1 = T1.t
        R1 = SO3(T1.R)
        t2 = t1 + np.array([0.0, -0.3, 0.0])
        R2 = SO3(T1.R)
        position_parameter2 = LinePositionParameter(t1, t2)
        attitude_parameter2 = OneAttitudeParameter(R1, R2)
        cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
        velocity_parameter2 = CubicVelocityParameter(time2)
        trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
        trajectory_planner2 = TrajectoryPlanner(trajectory_parameter2)

        time3 = 5.0
        t3 = t2 + np.array([0.3, 0.3, -0.3])
        R3 = R2
        position_parameter3 = LinePositionParameter(t2, t3)
        attitude_parameter3 = OneAttitudeParameter(R2, R3)
        cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
        velocity_parameter3 = CubicVelocityParameter(time3)
        trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
        trajectory_planner3 = TrajectoryPlanner(trajectory_parameter3)

        time4 = 5.0
        t4 = t3 + np.array([-0.3, 0.3, 0.3])
        R4 = R3
        position_parameter4 = LinePositionParameter(t3, t4)
        attitude_parameter4 = OneAttitudeParameter(R3, R4)
        cartesian_parameter4 = CartesianParameter(position_parameter4, attitude_parameter4)
        velocity_parameter4 = CubicVelocityParameter(time4)
        trajectory_parameter4 = TrajectoryParameter(cartesian_parameter4, velocity_parameter4)
        trajectory_planner4 = TrajectoryPlanner(trajectory_parameter4)

        time5 = 5.0
        t5 = t4 + np.array([-0.6, 0.0, 0.0])
        R5 = R4
        position_parameter5 = LinePositionParameter(t4, t5)
        attitude_parameter5 = OneAttitudeParameter(R4, R5)
        cartesian_parameter5 = CartesianParameter(position_parameter5, attitude_parameter5)
        velocity_parameter5 = CubicVelocityParameter(time5)
        trajectory_parameter5 = TrajectoryParameter(cartesian_parameter5, velocity_parameter5)
        trajectory_planner5 = TrajectoryPlanner(trajectory_parameter5)

        time6 = 5.0
        t6 = t5 + np.array([0.0, -0.8, 0.0])
        R6 = R5
        tc6 = (t5 + t6) / 2.0
        tc6[0] -= 0.2
        tc6[2] -= 0.3
        position_parameter6 = ArcPointPositionParameter(t5, t6, tc6)
        attitude_parameter6 = OneAttitudeParameter(R5, R6)
        cartesian_parameter6 = CartesianParameter(position_parameter6, attitude_parameter6)
        velocity_parameter6 = CubicVelocityParameter(time6)
        trajectory_parameter6 = TrajectoryParameter(cartesian_parameter6, velocity_parameter6)
        trajectory_planner6 = TrajectoryPlanner(trajectory_parameter6)

        time7 = 5.0
        t7 = t1
        R7 = R1
        position_parameter7 = LinePositionParameter(t6, t7)
        attitude_parameter7 = OneAttitudeParameter(R6, R7)
        cartesian_parameter7 = CartesianParameter(position_parameter7, attitude_parameter7)
        velocity_parameter7 = CubicVelocityParameter(time7)
        trajectory_parameter7 = TrajectoryParameter(cartesian_parameter7, velocity_parameter7)
        trajectory_planner7 = TrajectoryPlanner(trajectory_parameter7)

        time8 = 5.0

        total_time = time1 + time2 + time3 + time4 + time5 + time6 + time7 + time8
        time_step_num = round(total_time / model.opt.timestep) + 1
        desired_poses = np.zeros((time_step_num, robot.dof))
        desired_vels = np.zeros_like(desired_poses)
        real_poses = np.zeros_like(desired_poses)
        real_pos_prevs = np.zeros_like(desired_poses)
        sensor_datas = np.zeros_like(desired_poses)
        real_vels = np.zeros_like(desired_poses)
        motor_ctrls = np.zeros_like(desired_poses)
        error_poses = np.zeros_like(desired_poses)
        times = np.linspace(0, total_time, time_step_num)

        joint_position = np.zeros(robot.dof)
        for i, timei in enumerate(times):
            if timei < time1:
                planner_interpolate = trajectory_planner1.interpolate(timei)
                joint_position = planner_interpolate
                robot.move_joint(joint_position)

            elif timei <= time1 + time2:
                planner_interpolate = trajectory_planner2.interpolate(timei - time1)
                robot.move_cartesian(planner_interpolate)
                joint_position = robot.get_joint()

            elif timei <= time1 + time2 + time3:
                planner_interpolate = trajectory_planner3.interpolate(timei - time1 - time2)
                robot.move_cartesian(planner_interpolate)
                joint_position = robot.get_joint()

            elif timei <= time1 + time2 + time3 + time4:
                planner_interpolate = trajectory_planner4.interpolate(timei - time1 - time2 - time3)
                robot.move_cartesian(planner_interpolate)
                joint_position = robot.get_joint()

            elif timei <= time1 + time2 + time3 + time4 + time5:
                planner_interpolate = trajectory_planner5.interpolate(timei - time1 - time2 - time3 - time4)
                robot.move_cartesian(planner_interpolate)
                joint_position = robot.get_joint()

            elif timei <= time1 + time2 + time3 + time4 + time5 + time6:
                planner_interpolate = trajectory_planner6.interpolate(timei - time1 - time2 - time3 - time4 - time5)
                robot.move_cartesian(planner_interpolate)
                joint_position = robot.get_joint()

            elif timei <= time1 + time2 + time3 + time4 + time5 + time6 + time7:
                planner_interpolate = trajectory_planner7.interpolate(
                    timei - time1 - time2 - time3 - time4 - time5 - time6)
                robot.move_cartesian(planner_interpolate)
                joint_position = robot.get_joint()

            desired_poses[i, :] = joint_position
            if i > 0:
                desired_vels[i, :] = (desired_poses[i, :] - desired_poses[i - 1, :]) / model.opt.timestep

        time_num = 0
        mujoco.mj_resetData(model, data)
        initial_state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_PHYSICS))
        mujoco.mj_setState(model, data, initial_state, mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_forward(model, data)

        sensor_data = data.sensordata.copy()
        real_pos = sensor_data.copy()
        real_pos_prev = real_pos.copy()
        real_vel = (real_pos - real_pos_prev) / model.opt.timestep

        sensor_datas[time_num, :] = sensor_data.copy()
        real_poses[time_num, :] = real_pos.copy()
        real_pos_prevs[time_num, :] = real_pos_prev.copy()
        real_vels[time_num, :] = real_vel.copy()
        motor_ctrls[time_num, :] = data.ctrl.copy()

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
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

                pos_out = [position_controllers[i].control(error_pos[i]) for i in range(robot.dof)]
                for i in range(robot.dof):
                    data.ctrl[i] = velocity_controllers[i].control(pos_out[i] - real_vel[i])

                sensor_datas[time_num, :] = sensor_data.copy()
                real_poses[time_num, :] = real_pos.copy()
                real_pos_prevs[time_num, :] = real_pos_prev.copy()
                real_vels[time_num, :] = real_vel.copy()
                motor_ctrls[time_num, :] = data.ctrl.copy()
                error_poses[time_num, :] = error_pos.copy()

                mujoco.mj_step(model, data)

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

                viewer.sync()

        plt.figure(1)
        for i in range(robot.dof):
            plt.subplot(3, 2, i + 1)
            plt.plot(times, desired_poses[:, i], '-', label='desired position')
            plt.plot(times, sensor_datas[:, i], '--', label='actual position')
            plt.legend()
        plt.tight_layout()

        plt.figure(2)
        for i in range(robot.dof):
            plt.subplot(3, 2, i + 1)
            plt.plot(times, desired_vels[:, i], '-', label='desired velocity')
            plt.plot(times, real_vels[:, i], '--', label='actual velocity')
            plt.legend()
        plt.tight_layout()

        plt.figure(3)
        for i in range(robot.dof):
            plt.subplot(3, 2, i + 1)
            plt.plot(times, motor_ctrls[:, i], '-', label='motor torque')
            plt.legend()
        plt.tight_layout()

        plt.figure(4)
        for i in range(robot.dof):
            plt.subplot(3, 2, i + 1)
            plt.plot(times, error_poses[:, i], '-', label='position error')
            plt.legend()
        plt.tight_layout()

        plt.show()
