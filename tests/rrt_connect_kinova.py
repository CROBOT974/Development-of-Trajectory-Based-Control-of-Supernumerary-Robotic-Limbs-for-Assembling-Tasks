import numpy as np
from scipy.spatial.transform import Rotation
import random
from collision_detection import check_collision
from collision_detection import check_collision_obstacle


def compute_link_info(chain, joint_angles, index_begin=0, y_offset=0):
    link_info = {}
    joint_angles = np.append(np.array([0]), np.append(joint_angles, np.array([0])))
    transformation_matrices = chain.forward_kinematics(joint_angles, True)

    for i in range(1, len(joint_angles)):  # 遍历从第一个有效链接到最后一个有效链接
        parent_pos = transformation_matrices[i - 1][:3, 3].copy()

        parent_pos[1] = parent_pos[1] + y_offset
        child_pos = transformation_matrices[i][:3, 3].copy()
        child_pos[1] = child_pos[1] + y_offset

        length = np.linalg.norm(child_pos - parent_pos)
        center_pos = (parent_pos + child_pos) / 2

        link_info[i - 1 + index_begin] = {
            'center_joint': child_pos,
            'joint_R': 0.05,  # 假设关节半径为0.1
            'length': length,
            'center_pos': center_pos,
            'link_R': 0.05
        }
    #考虑到有摄像头，最后一个link的半径加长
    # link_info[i - 1 + index_begin]['link_R'] = 0.12

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
        # 用开始和结束的关节角度计算主从臂开始结束时的位置信息
        forward_link_info_main = compute_link_info(self.chain, start_main)
        forward_link_info_slave = compute_link_info(self.chain_r, start_slave, index_begin=8, y_offset=0)
        backward_link_info_main = compute_link_info(self.chain, goal_main)
        backward_link_info_slave = compute_link_info(self.chain_r, goal_slave, index_begin=8, y_offset=0)

        self.start_m.link_info = forward_link_info_main
        self.start_r.link_info = forward_link_info_slave
        self.goal_m.link_info = backward_link_info_main
        self.goal_r.link_info = backward_link_info_slave

        iiii = 0
        stop_main_rrt = 1 if np.allclose(start_main, goal_main, atol=1e-3) else 0  # 控制主臂RRT停止生长的信号
        stop_slave_rrt = 1 if np.allclose(start_slave, goal_slave, atol=1e-3) else 0  # 控制从臂RRT停止生长的信号
        for i in range(self.max_iter):
            print(iiii)
            iiii += 1

            if stop_slave_rrt == 0:
                rnd_node_r = self.get_random_node(goal_slave)
                nearest_start = self.get_nearest_node(self.tree_start_r, rnd_node_r)  # start_tree上离rnd最近的点
                new_start = self.steer(nearest_start, rnd_node_r)  # 往rnd的方向求出新点
                print("1这里1")
                forward_link_info_slave_try, backward_link_info_slave_try, connect = self.slave_rrt(nearest_start,
                                                                                                new_start,
                                                                                                forward_link_info_main,
                                                                                                backward_link_info_main)
                print("1这里2")
                if backward_link_info_slave_try != 0:
                    backward_link_info_slave = backward_link_info_slave_try
                    print("2这里1")
                if forward_link_info_slave_try != 0:
                    forward_link_info_slave = forward_link_info_slave_try
                if connect == 1:  # 代表树可以连接
                    print("3这里1")
                    stop_slave_rrt = 1

            if stop_main_rrt == 0:
                rnd_node_m = self.get_random_node(goal_main)
                nearest_start = self.get_nearest_node(self.tree_start_m, rnd_node_m)  # start_tree上离rnd最近的点
                new_start = self.steer(nearest_start, rnd_node_m)  # 往rnd的方向求出新点
                print("4这里1")
                forward_link_info_main_try, backward_link_info_main_try, connect = self.main_rrt(nearest_start, new_start,
                                                                                             forward_link_info_slave,
                                                                                             backward_link_info_slave)
                print("4这里2")
                if backward_link_info_main_try != 0:
                    print("5这里1")
                    backward_link_info_main = backward_link_info_main_try
                if forward_link_info_main_try != 0:
                    forward_link_info_main = forward_link_info_main_try
                if connect == 1:  # 代表树可以连接
                    print("6这里1")
                    stop_main_rrt = 1

            if stop_main_rrt == 1 and stop_slave_rrt == 1:
                return self.path_m, self.path_r, self.p

        return None

    def slave_rrt(self, nearest_start, new_start, forward_link_info_main, backward_link_info_main):
        # 从tree_start处加长tree
        if not self.check_collision(new_start, forward_link_info_main):  # 如果没有碰撞
            self.tree_start_r.append(new_start)
            next_forward_link_info_slave = compute_link_info(self.chain_r, new_start.position, index_begin=8,
                                                             y_offset=0)
            nearest_goal = self.get_nearest_node(self.tree_goal_r, new_start)
            new_goal = self.steer(nearest_goal, new_start)

            # test = self.chain.forward_kinematics(np.append(np.append(np.array(0), new_goal.position), np.array(0)), True)
            # print(test[-1])

            # 这里加入link_infomation的充值，但是要记得后面check_collision的link_information

            if not self.check_collision(new_goal, backward_link_info_main):
                self.tree_goal_r.append(new_goal)
                next_backward_link_info_slave = compute_link_info(self.chain_r, new_goal.position, index_begin=8,
                                                                  y_offset=0)
                # 需要修改connect_tree
                if self.connect_trees(self.tree_start_r, self.tree_goal_r, forward_link_info_main):
                    path_start = self.get_final_path(new_start)
                    path_goal = self.get_final_path(new_goal)
                    self.path_r = path_start + path_goal[::-1]
                    return next_forward_link_info_slave, next_backward_link_info_slave, 1  # 这里的1代表可以连接树
                else:
                    return next_forward_link_info_slave, next_backward_link_info_slave, 0
            else:
                self.tree_save = self.tree_goal_r
                self.tree_goal_r = self.tree_start_r
                self.tree_start_r = self.tree_save
                return next_forward_link_info_slave, 0, 0
                # new_goal = self.avoid_obstacle(nearest_goal, backward_link_info_main)
                # # test = self.chain.forward_kinematics(np.append(np.append(np.array(0), new_goal.position), np.array(0)),
                # #                                      True)
                # #
                # print("new_goal",new_goal)
                # self.tree_goal_r.append(new_goal)
                # next_backward_link_info_slave = compute_link_info(self.chain_r, new_goal.position, index_begin=8,
                #                                                   y_offset=0)
                # return next_forward_link_info_slave, next_backward_link_info_slave, 0

        else:
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
        # 从tree_start处加长tree
        if not self.check_collision(new_start, forward_link_info_slave):  # 如果前向没有碰撞
            self.tree_start_m.append(new_start)
            next_forward_link_info_main = compute_link_info(self.chain, new_start.position)
            nearest_goal = self.get_nearest_node(self.tree_goal_m, new_start)
            new_goal = self.steer(nearest_goal, new_start)

            # 这里加入link_infomation的充值，但是要记得后面check_collision的link_information

            if not self.check_collision(new_goal, backward_link_info_slave):  # 如果后向没有碰撞
                print("kkkk")
                self.tree_goal_m.append(new_goal)
                next_backward_link_info_main = compute_link_info(self.chain, new_goal.position)
                ##需要修改connect_tree
                if self.connect_trees(self.tree_start_m, self.tree_goal_m, forward_link_info_slave):
                    path_start = self.get_final_path(new_start)
                    path_goal = self.get_final_path(new_goal)
                    print("path_goal", path_goal)
                    self.path_m = path_start + path_goal[::-1]
                    return next_forward_link_info_main, next_backward_link_info_main, 1  # 这里的1代表可以连接树
                else:
                    return next_forward_link_info_main, next_backward_link_info_main, 0
            else:
                self.tree_save = self.tree_goal_m
                self.tree_goal_m = self.tree_start_m
                self.tree_start_m = self.tree_save
                return next_forward_link_info_main, 0, 0
                # print("qqqqq")
                # new_goal = self.avoid_obstacle(nearest_goal, backward_link_info_slave)
                # self.tree_goal_m.append(new_goal)
                # next_backward_link_info_main = compute_link_info(self.chain, new_goal.position)
                # return next_forward_link_info_main, next_backward_link_info_main, 0
        else:
            self.tree_save = self.tree_goal_m
            self.tree_goal_m = self.tree_start_m
            self.tree_start_m = self.tree_save
            return 0, 0, 0
            # """这里出现了new_start是NoneType的独特情况"""
            # new_start = self.avoid_obstacle(nearest_start, forward_link_info_slave)
            # self.tree_start_m.append(new_start)
            # next_forward_link_info_main = compute_link_info(self.chain, new_start.position)
            # return next_forward_link_info_main, 0, 0

    def get_random_node(self, target):
        rnd_position = np.zeros(7)
        for i in range(len(rnd_position)):
            rnd_position[i] = np.array([random.uniform(target[i]-0.5, target[i]+0.5)])
        # rnd_position[0] = np.array([random.uniform(-6.28, 6.28)])
        # rnd_position[2] = np.array([random.uniform(-6.28, 6.28)])
        # rnd_position[4] = np.array([random.uniform(-6.28, 6.28)])
        # rnd_position[6] = np.array([random.uniform(-6.28, 6.28)])
        #
        # rnd_position[1] = np.array([random.uniform(-2.25, 2.25)])
        # rnd_position[3] = np.array([random.uniform(-2.58, 2.58)])
        # rnd_position[5] = np.array([random.uniform(-2.1, 2.1)])
        return Node(rnd_position)
    # def get_random_node(self):
    #     rnd_position = np.zeros(7)
    #     rnd_position[0] = np.array([random.uniform(-6.28, 6.28)])
    #     rnd_position[2] = np.array([random.uniform(-6.28, 6.28)])
    #     rnd_position[4] = np.array([random.uniform(-6.28, 6.28)])
    #     rnd_position[6] = np.array([random.uniform(-6.28, 6.28)])
    #
    #     rnd_position[1] = np.array([random.uniform(-2.25, 2.25)])
    #     rnd_position[3] = np.array([random.uniform(-2.58, 2.58)])
    #     rnd_position[5] = np.array([random.uniform(-2.1, 2.1)])
    #     return Node(rnd_position)

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

    def connect_trees(self, tree_from, tree_to, link_info_main):  # tree start, tree goal
        nearest_node = self.get_nearest_node(tree_to, tree_from[-1])  # tree_goal上点的找与start_str上最后一点最近的点
        new_node = self.steer(nearest_node, tree_from[-1])  # tree_goal往tree_st末尾方向求出新点

        if not self.check_collision(new_node, link_info_main):
            tree_to.append(new_node)
            if np.linalg.norm(new_node.position - tree_from[-1].position) < self.step_size:  # 两树末端距离小于step，则视为相连
                self.p.append(tree_from[-1].position)
                return True
        return False

    def check_collision(self, node, link_info_1):
        if all(0 <= key <= 7 for key in link_info_1):  # link1为主臂，则2为从臂
            link_info_2 = compute_link_info(self.chain_r, node.position, index_begin=8, y_offset=0)
        else:
            link_info_2 = compute_link_info(self.chain, node.position)  # index_begin代表对应的初始joint，这里是从机械臂的joints
        link_info = {**link_info_1, **link_info_2}  # 这里的谁主谁从没有意义，因为所有link都要两两对比
        # print("link_main:", link_info_main)
        # print("link", link_info)
        # 获取link_info的大小
        num_links = len(link_info)
        # print(num_links)
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
        collision = check_collision(O1=link_matrix[:8, :3], O2=link_matrix[8:, :3],
                                    R1=link_matrix[:8, 3], R2=link_matrix[8:, 3],
                                    L1=link_matrix[:8, 4], L2=link_matrix[8:, 4],
                                    A=link_matrix[:8, 5:8], B=link_matrix[8:, 5:8],
                                    Ra=link_matrix[:8, 8], Rb=link_matrix[8:, 8],
                                    )  # 左边为主机械臂，右边为从机械臂
        collision_obs = check_collision_obstacle(O1=link_matrix[:8, :3], O2=link_matrix[8:, :3],
                                                 R1=link_matrix[:8, 3], R2=link_matrix[8:, 3],
                                                 L1=link_matrix[:8, 4], L2=link_matrix[8:, 4],
                                                 A=link_matrix[:8, 5:8], B=link_matrix[8:, 5:8],
                                                 Ra=link_matrix[:8, 8], Rb=link_matrix[8:, 8],
                                                 obstacle=self.obstacle, size=self.size
                                                 )
        # print(collision)
        if collision or collision_obs:
            collide = 1
        else:
            collide = 0
        return collide

    def avoid_obstacle(self, nearest_node, link_info_main, step_size=0.05):
        for _ in range(5000):  # 尝试5次不同方向
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
