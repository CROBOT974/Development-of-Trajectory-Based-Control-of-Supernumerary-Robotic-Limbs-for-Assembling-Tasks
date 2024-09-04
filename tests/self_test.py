from unittest import TestCase
import time
import mujoco
import mujoco.viewer
import numpy as np

from pid_control.src.controller import PIDController
from matplotlib import pyplot as plt

from pid_control.src.robot import *
from pid_control.src.motion_planning import *
from spatialmath import SO3


model = mujoco.MjModel.from_xml_path("../src/assets/universal_robots_ur5e/scene.xml")
data = mujoco.MjData(model)

robot = UR5e()
q0 = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0])
robot.set_joint(q0)
time_num = 0
total_time = 10
time_step_num = round(total_time / model.opt.timestep) + 1

# 重新设置机械臂的姿态
mujoco.mj_resetData(model, data)
initial_state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_PHYSICS))
initial_state[2] = np.pi / 2
mujoco.mj_setState(model, data, initial_state, mujoco.mjtState.mjSTATE_PHYSICS)
mujoco.mj_forward(model, data)  # 似乎没有影响


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time <= total_time:
        time_num += 1

        if time_num >= time_step_num:
            break


        mujoco.mj_step(model, data)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        viewer.sync()