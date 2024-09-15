import numpy as np
import random
from .collision.collision_detection import check_collision
from .collision.collision_detection import check_collision_obstacle


def compute_link_info(chain, joint_angles, index_begin=0, y_offset=0):
    link_info = {}
    joint_angles = np.append(np.array([0]), np.append(joint_angles, np.array([0])))
    transformation_matrices = chain.forward_kinematics(joint_angles, True)

    for i in range(1, len(joint_angles)):  # check all joint from beginning to the end
        parent_pos = transformation_matrices[i - 1][:3, 3].copy()

        parent_pos[1] = parent_pos[1] + y_offset
        child_pos = transformation_matrices[i][:3, 3].copy()
        child_pos[1] = child_pos[1] + y_offset

        length = np.linalg.norm(child_pos - parent_pos)
        center_pos = (parent_pos + child_pos) / 2

        link_info[i - 1 + index_begin] = {
            'center_joint': child_pos,
            'joint_R': 0.05,  # assume the joint has radius as 0.05
            'length': length,
            'center_pos': center_pos,
            'link_R': 0.05
        }
    return link_info


class Node:
    def __init__(self, position, parent=None, link_info=None):
        self.position = position
        self.parent = parent
        self.link_info = link_info


class RRTConnect:
    def __init__(self, chain, chain_r, obstacle, size, start_m, start_r, goal_m, goal_r, max_iter=10000, step_size=0.1,
                 goal_sample_rate=0.1):
        self.chain = chain
        self.chain_r = chain_r
        self.obstacle = obstacle
        self.size = size
        self.start_m = Node(start_m)
        self.goal_m = Node(goal_m)
        self.start_r = Node(start_r)
        self.goal_r = Node(goal_r)
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.tree_start_m = [self.start_m]
        self.tree_start_r = [self.start_r]
        self.tree_goal_m = [self.goal_m]
        self.tree_goal_r = [self.goal_r]
        self.tree_save = []
        self.path_m = []
        self.path_r = []

        self.p = []

    def planning(self, start_main, start_slave, goal_main, goal_slave):
        # starting and ending joint angles are used to calculate the link info at the beginning and end of robots
        forward_link_info_main = compute_link_info(self.chain, start_main)
        forward_link_info_slave = compute_link_info(self.chain_r, start_slave, index_begin=8, y_offset=0)
        backward_link_info_main = compute_link_info(self.chain, goal_main)
        backward_link_info_slave = compute_link_info(self.chain_r, goal_slave, index_begin=8, y_offset=0)
        # add those info into the tree
        self.start_m.link_info = forward_link_info_main
        self.start_r.link_info = forward_link_info_slave
        self.goal_m.link_info = backward_link_info_main
        self.goal_r.link_info = backward_link_info_slave

        iiii = 0
        # signal to stop expanding the RRT of main and secondary robot
        stop_main_rrt = 1 if np.allclose(start_main, goal_main, atol=1e-3) else 0
        stop_slave_rrt = 1 if np.allclose(start_slave, goal_slave, atol=1e-3) else 0

        for i in range(self.max_iter):
            print(iiii)
            iiii += 1
            if stop_slave_rrt == 0:  # secondary_arm
                rnd_node_r = self.get_random_node(goal_slave)  # random point
                # find the nearest point in start_tree from rnd, steer a new point from here
                nearest_start = self.get_nearest_node(self.tree_start_r, rnd_node_r)
                new_start = self.steer(nearest_start, rnd_node_r)
                # print("1here1")
                forward_link_info_slave_try, backward_link_info_slave_try, connect = self.slave_rrt(nearest_start,
                                                                                                    new_start,
                                                                                                    forward_link_info_main,
                                                                                                    backward_link_info_main)
                # print("1here2")
                if backward_link_info_slave_try != 0:
                    backward_link_info_slave = backward_link_info_slave_try
                    # print("2here1")
                if forward_link_info_slave_try != 0:
                    forward_link_info_slave = forward_link_info_slave_try
                if connect == 1:  # which means the tree is connected, return the stop signal
                    # print("3here1")
                    stop_slave_rrt = 1

            if stop_main_rrt == 0:  # main_arm
                rnd_node_m = self.get_random_node(goal_main)
                nearest_start = self.get_nearest_node(self.tree_start_m, rnd_node_m)
                new_start = self.steer(nearest_start, rnd_node_m)
                # print("4here1")
                forward_link_info_main_try, backward_link_info_main_try, connect = self.main_rrt(nearest_start,
                                                                                                 new_start,
                                                                                                 forward_link_info_slave,
                                                                                                 backward_link_info_slave)
                # print("4here2")
                if backward_link_info_main_try != 0:
                    # print("5here1")
                    backward_link_info_main = backward_link_info_main_try
                if forward_link_info_main_try != 0:
                    forward_link_info_main = forward_link_info_main_try
                if connect == 1:
                    # print("6here1")
                    stop_main_rrt = 1

            if stop_main_rrt == 1 and stop_slave_rrt == 1:  # both trees are connected
                return self.path_m, self.path_r, self.p

        return None

    def slave_rrt(self, nearest_start, new_start, forward_link_info_main, backward_link_info_main):
        # steer from tree_start
        if not self.check_collision(new_start, forward_link_info_main):  # if no collision in T_start
            self.tree_start_r.append(new_start)
            next_forward_link_info_slave = compute_link_info(self.chain_r, new_start.position, index_begin=8,
                                                             y_offset=0)
            nearest_goal = self.get_nearest_node(self.tree_goal_r, new_start)
            new_goal = self.steer(nearest_goal, new_start)
            if not self.check_collision(new_goal, backward_link_info_main):  # if no collision in T_goal
                self.tree_goal_r.append(new_goal)
                next_backward_link_info_slave = compute_link_info(self.chain_r, new_goal.position, index_begin=8,
                                                                  y_offset=0)
                if self.connect_trees(self.tree_start_r, self.tree_goal_r, forward_link_info_main):  # if connected
                    path_start = self.get_final_path(new_start)
                    path_goal = self.get_final_path(new_goal)
                    self.path_r = path_start + path_goal[::-1]
                    return next_forward_link_info_slave, next_backward_link_info_slave, 1  # the "1" means connected
                else:  # if not connected
                    return next_forward_link_info_slave, next_backward_link_info_slave, 0
            else:  # if collision in T_goal
                # swap T_start and T_goal
                self.tree_save = self.tree_goal_r
                self.tree_goal_r = self.tree_start_r
                self.tree_start_r = self.tree_save
                return next_forward_link_info_slave, 0, 0
                # new_goal = self.avoid_obstacle(nearest_goal, backward_link_info_main)
                # # test = self.chain.forward_kinematics(np.append(np.append(np.array(0), new_goal.position), np.array(0)),
                # #                                      True)
                # print("new_goal",new_goal)
                # self.tree_goal_r.append(new_goal)
                # next_backward_link_info_slave = compute_link_info(self.chain_r, new_goal.position, index_begin=8,
                #                                                   y_offset=0)
                # return next_forward_link_info_slave, next_backward_link_info_slave, 0

        else:  # if collision in T_start
            # swap T_start and T_goal
            self.tree_save = self.tree_goal_r
            self.tree_goal_r = self.tree_start_r
            self.tree_start_r = self.tree_save
            return 0, 0, 0
            # new_start = self.avoid_obstacle(nearest_start, forward_link_info_main)
            # self.tree_start_r.append(new_start)
            # next_forward_link_info_slave = compute_link_info(self.chain_r, new_start.position, index_begin=8,
            #                                                  y_offset=0)
            # return next_forward_link_info_slave, 0, 0

    def main_rrt(self, nearest_start, new_start, forward_link_info_slave, backward_link_info_slave):
        if not self.check_collision(new_start, forward_link_info_slave):
            self.tree_start_m.append(new_start)
            next_forward_link_info_main = compute_link_info(self.chain, new_start.position)
            nearest_goal = self.get_nearest_node(self.tree_goal_m, new_start)
            new_goal = self.steer(nearest_goal, new_start)
            if not self.check_collision(new_goal, backward_link_info_slave):
                # print("kkkk")
                self.tree_goal_m.append(new_goal)
                next_backward_link_info_main = compute_link_info(self.chain, new_goal.position)
                if self.connect_trees(self.tree_start_m, self.tree_goal_m, forward_link_info_slave):
                    path_start = self.get_final_path(new_start)
                    path_goal = self.get_final_path(new_goal)
                    # print("path_goal", path_goal)
                    self.path_m = path_start + path_goal[::-1]
                    return next_forward_link_info_main, next_backward_link_info_main, 1
                else:
                    return next_forward_link_info_main, next_backward_link_info_main, 0
            else:
                self.tree_save = self.tree_goal_m
                self.tree_goal_m = self.tree_start_m
                self.tree_start_m = self.tree_save
                return next_forward_link_info_main, 0, 0
                # new_goal = self.avoid_obstacle(nearest_goal, backward_link_info_slave)
                # self.tree_goal_m.append(new_goal)
                # next_backward_link_info_main = compute_link_info(self.chain, new_goal.position)
                # return next_forward_link_info_main, next_backward_link_info_main, 0
        else:
            self.tree_save = self.tree_goal_m
            self.tree_goal_m = self.tree_start_m
            self.tree_start_m = self.tree_save
            return 0, 0, 0
            # new_start = self.avoid_obstacle(nearest_start, forward_link_info_slave)
            # self.tree_start_m.append(new_start)
            # next_forward_link_info_main = compute_link_info(self.chain, new_start.position)
            # return next_forward_link_info_main, 0, 0

    def get_random_node(self, target):
        rnd_position = np.zeros(7)
        # goal_directional "random" node
        for i in range(len(rnd_position)):
            rnd_position[i] = np.array([random.uniform(target[i] - 0.5, target[i] + 0.5)])
        # # true random node
        # rnd_position[0] = np.array([random.uniform(-6.28, 6.28)])
        # rnd_position[2] = np.array([random.uniform(-6.28, 6.28)])
        # rnd_position[4] = np.array([random.uniform(-6.28, 6.28)])
        # rnd_position[6] = np.array([random.uniform(-6.28, 6.28)])
        #
        # rnd_position[1] = np.array([random.uniform(-2.25, 2.25)])
        # rnd_position[3] = np.array([random.uniform(-2.58, 2.58)])
        # rnd_position[5] = np.array([random.uniform(-2.1, 2.1)])
        return Node(rnd_position)

    def get_nearest_node(self, tree, node):
        distances = [np.linalg.norm(n.position - node.position) for n in tree]
        nearest_index = distances.index(min(distances))
        return tree[nearest_index]

    def steer(self, from_node, to_node):
        direction = to_node.position - from_node.position
        distance = np.linalg.norm(direction)
        # print(distance)
        direction = direction / distance if distance != 0 else direction

        new_position = from_node.position + direction * min(self.step_size, distance)
        return Node(new_position, from_node)

    def connect_trees(self, tree_from, tree_to, link_info_main):
        nearest_node = self.get_nearest_node(tree_to, tree_from[-1])
        new_node = self.steer(nearest_node, tree_from[-1])
        if not self.check_collision(new_node, link_info_main):
            tree_to.append(new_node)
            if np.linalg.norm(new_node.position - tree_from[-1].position) < self.step_size:  # dis between two tree < step_size
                self.p.append(tree_from[-1].position)
                return True
        return False

    def check_collision(self, node, link_info_1):
        if all(0 <= key <= 7 for key in link_info_1):  # link1 is m_arm，link2 is r_arm
            link_info_2 = compute_link_info(self.chain_r, node.position, index_begin=8, y_offset=0)
            link_info = {**link_info_1, **link_info_2}
        else:  # link1 is r_arm，link2 is m_arm
            link_info_2 = compute_link_info(self.chain, node.position)
            link_info = {**link_info_2, **link_info_1}
        # print("link_main:", link_info_main)
        # print("link", link_info)
        num_links = len(link_info)
        num_features = 3 + 1 + 1 + 3 + 1  # center_joint (3), joint_R (1), length (1), center_pos (3), link_R (1)

        # 初始化一个矩阵来存储link_info中的值
        link_matrix = np.zeros((num_links, num_features))

        # 将link_info中的值填充到矩阵中
        for iii, (link_id, info) in enumerate(link_info.items()):
            link_matrix[iii, :3] = info['center_joint']
            link_matrix[iii, 3] = info['joint_R']
            link_matrix[iii, 4] = info['length']
            link_matrix[iii, 5:8] = info['center_pos']
            link_matrix[iii, 8] = info['link_R']
        # test collision between robots
        collision = check_collision(O1=link_matrix[:8, :3], O2=link_matrix[8:, :3],
                                    R1=link_matrix[:8, 3], R2=link_matrix[8:, 3],
                                    L1=link_matrix[:8, 4], L2=link_matrix[8:, 4],
                                    A=link_matrix[:8, 5:8], B=link_matrix[8:, 5:8],
                                    Ra=link_matrix[:8, 8], Rb=link_matrix[8:, 8],
                                    )  # left is m_arm, right is s_arm
        # test collision between robot and environment
        collision_obs = check_collision_obstacle(O1=link_matrix[:8, :3], O2=link_matrix[8:, :3],
                                                 R1=link_matrix[:8, 3], R2=link_matrix[8:, 3],
                                                 L1=link_matrix[:8, 4], L2=link_matrix[8:, 4],
                                                 A=link_matrix[:8, 5:8], B=link_matrix[8:, 5:8],
                                                 Ra=link_matrix[:8, 8], Rb=link_matrix[8:, 8],
                                                 obstacle=self.obstacle, size=self.size
                                                 )
        if collision or collision_obs:
            collide = 1
        else:
            collide = 0
        return collide

    def avoid_obstacle(self, nearest_node, link_info_main, step_size=0.05):  # bi-RRT_Connect, not for our method
        for _ in range(5000):  # test 5000 different directions
            direction = np.zeros(7)
            direction[0] = np.array([random.uniform(-6.28, 6.28)])
            direction[2] = np.array([random.uniform(-6.28, 6.28)])
            direction[4] = np.array([random.uniform(-6.28, 6.28)])
            direction[6] = np.array([random.uniform(-6.28, 6.28)])

            direction[1] = np.array([random.uniform(-2.25, 2.25)])
            direction[3] = np.array([random.uniform(-2.58, 2.58)])
            direction[5] = np.array([random.uniform(-2.1, 2.1)])
            direction = direction / np.linalg.norm(direction)
            new_position = nearest_node.position + direction * step_size
            new_node = Node(new_position, nearest_node)
            if not self.check_collision(new_node, link_info_main):
                return new_node
        return None

    def get_final_path(self, node):
        path = []
        while node.parent is not None:
            path.append(node.position)
            node = node.parent
        path.append(node.position)
        return path[::-1]  # Reverse the path
