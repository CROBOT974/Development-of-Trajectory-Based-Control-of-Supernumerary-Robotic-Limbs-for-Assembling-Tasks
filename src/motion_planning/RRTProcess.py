import numpy as np
from scipy.spatial.transform import Rotation
from .bezier.Bézier_curve import bezier_curve
from .rrt_connect_kinova import RRTConnect


class TrajectoryProcess:
    def __init__(self, chain, chain_r, obstacle, size, init_pos, init_pos_r, end_pos, end_pos_r, init_ori, init_ori_r,
                 end_ori, end_ori_r, init_qpos, init_qpos_r):
        self.chain = chain
        self.chain_r = chain_r
        self.obstacle = obstacle
        self.size = size
        self.init_pos = init_pos
        self.init_pos_r = init_pos_r
        self.end_pos = end_pos
        self.end_pos_r = end_pos_r
        self.init_ori = init_ori
        self.init_ori_r = init_ori_r
        self.end_ori = end_ori
        self.end_ori_r = end_ori_r
        self.init_chain_pose = np.append(np.append(np.array(0), init_qpos), np.array(0))
        self.init_chain_pose_r = np.append(np.append(np.array(0), init_qpos_r), np.array(0))

    def rrt_connect(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=100)

        # loading the Mujoco model
        end_orientation = Rotation.from_euler('xyz', self.end_ori, degrees=True)
        end_orientation_r = Rotation.from_euler('xyz', self.end_ori_r, degrees=True)

        init_pos = self.init_pos
        init_pos_r = self.init_pos_r  # in Cartesian space
        init_chain_pose = self.init_chain_pose
        init_chain_pose_r = self.init_chain_pose_r  # in joint space

        end_position = self.end_pos
        end_position_r = self.end_pos_r  # in Cartesian space
        # avoid the singularity
        end_position[0] = 1e-3 if end_position[0] == 0 else end_position[0]
        end_position_r[0] = 1e-3 if end_position_r[0] == 0 else end_position_r[0]

        # init_matrix = self.chain.forward_kinematics(init_chain_pose)
        # init_matrix_r = self.chain_r.forward_kinematics(init_chain_pose_r)

        # init orientation of end effector
        init_matrix = self.init_ori
        init_matrix_r = self.init_ori_r
        # target orientation of end effector
        target_matrix = np.eye(4)
        target_matrix_r = np.eye(4)
        target_matrix[:3, :3] = np.dot(init_matrix, end_orientation.as_matrix())
        # print("the orientation", target_matrix[:3, :3])
        # print("end1", end_position)
        target_matrix[:3, 3] = end_position
        target_matrix_r[:3, :3] = np.dot(init_matrix_r, end_orientation_r.as_matrix())
        # print("end2", end_position_r)
        target_matrix_r[:3, 3] = end_position_r

        # IK for target pose
        ik_joint = self.chain.inverse_kinematics_frame(target_matrix,
                                                       initial_position=init_chain_pose,
                                                       orientation_mode='all')
        ik_joint_r = self.chain_r.inverse_kinematics_frame(target_matrix_r, initial_position=init_chain_pose_r,
                                                           orientation_mode='all')

        # for no move of one arm, keeping the initial joint values
        if np.allclose(end_orientation.as_euler('xyz'), [0, 0, 0], atol=1e-6) and np.allclose(end_position, init_pos,
                                                                                              atol=1e-6):
            ik_joint = init_chain_pose
        if np.allclose(end_orientation_r.as_euler('xyz'), [0, 0, 0], atol=1e-6) and np.allclose(end_position_r,
                                                                                                init_pos_r, atol=1e-6):
            ik_joint_r = init_chain_pose_r

        # RRT Connect
        rrt = RRTConnect(self.chain, self.chain_r, self.obstacle, self.size, start_m=init_chain_pose[1:8],
                         start_r=init_chain_pose_r[1:8],
                         goal_m=ik_joint[1:8], goal_r=ik_joint_r[1:8], max_iter=2000, step_size=0.04)
        # path_m and r is the trajectory for main and secondary arm in joint space
        # p is the connect point of T_start in the path
        path_m, path_r, p = rrt.planning(init_chain_pose[1:8], init_chain_pose_r[1:8], ik_joint[1:8], ik_joint_r[1:8])

        # no path found, keep the position
        if not path_m:
            path_m.append(ik_joint[1:8])
            p.append(ik_joint[1:8])
        if not path_r:
            path_r.append(ik_joint_r[1:8])
            p.append(ik_joint_r[1:8])

        print("tra_length 1：", len(path_m) - 1)
        print("tra_length 2：", len(path_r) - 1)
        print(p)

        # fill the two path to be same length in the connect point
        full_path_m = np.vstack(path_m)
        full_path_r = np.vstack(path_r)

        if len(path_m) < len(path_r):  # m_arm needs path fill
            n = len(path_r) - len(path_m)
            target_indices_m = np.where(np.all(path_m == p[0], axis=1))[0]
            if len(target_indices_m) == 0:  # p[0] is for r_arm，p[1] is for m_arm
                target_indices_m = np.where(np.all(path_m == p[1], axis=1))[0]
                insert_points_m = np.tile(p[1], (n, 1))
                full_path_m = np.insert(path_m, target_indices_m[0] + 1, insert_points_m, axis=0)
            else:  # p[0] is for m_arm，p[1] is for r_arm
                insert_points_m = np.tile(p[0], (n, 1))
                full_path_m = np.insert(path_m, target_indices_m[0] + 1, insert_points_m, axis=0)

        elif len(path_r) < len(path_m):  # r_arm needs path fill
            n = len(path_m) - len(path_r)
            target_indices_r = np.where(np.all(path_r == p[0], axis=1))[0]
            if len(target_indices_r) == 0:  # p[0] is for m_arm，p[1] is for r_arm
                target_indices_r = np.where(np.all(path_r == p[1], axis=1))[0]
                insert_points_r = np.tile(p[1], (n, 1))
                full_path_r = np.insert(path_r, target_indices_r[0] + 1, insert_points_r, axis=0)
            else: # p[0] is for r_arm，p[1] is for m_arm
                insert_points_r = np.tile(p[0], (n, 1))
                full_path_r = np.insert(path_r, target_indices_r[0] + 1, insert_points_r, axis=0)

        zeros_column = np.zeros((full_path_m.shape[0], 1))
        # add 1 null space for gripper
        full_path_m = np.concatenate((full_path_m, zeros_column), axis=1)
        full_path_r = np.concatenate((full_path_r, zeros_column), axis=1)

        # use Bezier_curve to make the paths smooth
        print("Bezier Curve Running......")
        control1 = bezier_curve(full_path_m, seg=3000)
        control2 = bezier_curve(full_path_r, seg=3000)

        return control1, control2, target_matrix, target_matrix_r
