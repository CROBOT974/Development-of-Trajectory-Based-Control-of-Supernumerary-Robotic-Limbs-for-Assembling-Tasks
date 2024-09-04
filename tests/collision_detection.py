import numpy as np
import fcl
from scipy.spatial.transform import Rotation as R


def calculate_rotation_matrix(v1, v2):
    # 归一化向量
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # 计算旋转轴
    axis = np.cross(v1, v2)
    axis_length = np.linalg.norm(axis)

    # 如果两个向量是平行的
    if axis_length == 0:
        return np.eye(3)

    axis = axis / axis_length

    # 计算旋转角度
    angle = np.arccos(np.dot(v1, v2))

    # 使用轴角法生成旋转矩阵
    rotation_matrix = R.from_rotvec(axis * angle).as_matrix()

    return rotation_matrix


def check_collision_obstacle(O1, O2, A, B, R1, R2, L1, L2, Ra, Rb, obstacle, size):
    num_links = O1.shape[0]
    V1 = [0, 0, 1]
    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()

    g0 = fcl.Box(size[0], size[1], size[2])
    t0 = fcl.Transform(obstacle)
    o0 = fcl.CollisionObject(g0, t0)

    # 是否是最后一个link碰撞的
    V2_1 = O1[-1] - A[-1]
    RO_1 = calculate_rotation_matrix(V1, V2_1)
    g1 = fcl.Cylinder(Ra[-1], L1[-1])
    t1 = fcl.Transform(RO_1, A[-1])
    o1 = fcl.CollisionObject(g1, t1)
    ret1 = fcl.collide(o0, o1, request, result)

    V2_2 = O2[-1] - B[-1]
    RO_2 = calculate_rotation_matrix(V1, V2_2)
    g2 = fcl.Cylinder(Rb[-1], L2[-1])
    t2 = fcl.Transform(RO_2, B[-1])
    o2 = fcl.CollisionObject(g2, t2)
    ret2 = fcl.collide(o0, o2, request, result)
    if ret1 == 1 or ret2 == 1:
        print("与身体碰撞")
        return 3  # end effector和障碍碰撞

    for i in range(num_links):
        V2_1 = O1[i] - A[i]
        RO_1 = calculate_rotation_matrix(V1, V2_1)
        g1 = fcl.Cylinder(Ra[i], L1[i])
        t1 = fcl.Transform(RO_1, A[i])
        o1 = fcl.CollisionObject(g1, t1)

        ret = fcl.collide(o0, o1, request, result)
        if ret == 1:
            return 1  # 主臂碰撞
        if O1[i][2] < -0.422 or A[i][2] < -0.422:
            print("与地面碰撞")
            return 4 # 与地面碰撞

    for j in range(num_links):
        V2_2 = O2[j] - B[j]
        RO_2 = calculate_rotation_matrix(V1, V2_2)
        g2 = fcl.Cylinder(Rb[j], L2[j])
        t2 = fcl.Transform(RO_2, B[j])
        o2 = fcl.CollisionObject(g2, t2)

        ret = fcl.collide(o0, o2, request, result)
        if ret == 1:
            return 2  # 从臂碰撞
        if O2[j][2] < -0.422 or B[j][2] < -0.422:
            print("与地面碰撞")
            return 4 # 与地面碰撞
    return 0


# 碰撞检测函数
# 01,02:center_joint    A,B:center_pos    R1,R2:joint_radius    L1,L2:length  Ra,Rb: link_radius
def check_collision(O1, O2, A, B, R1, R2, L1, L2, Ra, Rb):
    num_links = O1.shape[0]
    # V1 = [0, 0, 1]
    #
    #
    # for i in range(num_links):
    #     for j in range(num_links):
    #         V2_1 = O1[i]-A[i]
    #         RO_1 = calculate_rotation_matrix(V1, V2_1)
    #         g1 = fcl.Cylinder(Ra[i], L1[i])
    #         t1 = fcl.Transform(RO_1, A[i])
    #         o1 = fcl.CollisionObject(g1, t1)
    #
    #         V2_2 =O2[j]-B[j]
    #         RO_2 = calculate_rotation_matrix(V1, V2_2)
    #         g2 = fcl.Cylinder(Rb[j], L2[j])
    #         t2 = fcl.Transform(RO_2, B[j])
    #         o2 = fcl.CollisionObject(g2, t2)
    #
    #         request = fcl.CollisionRequest()
    #         result = fcl.CollisionResult()
    #
    #         ret = fcl.collide(o1, o2, request, result)
    #         if ret == 1:
    #             return ret
    # return 0

    # 主机械臂与从机械臂之间的碰撞检测
    # 01,02:center_joint    A,B:center_pos    R1,R2:joint_radius    L1,L2:length  Ra,Rb: link_radius
    if ((np.linalg.norm(O1[-1] - B[-1]) < (R1[-1] + Rb[-1]) or
         np.linalg.norm(A[-1] - O2[-1]) < (Ra[-1] + R2[-1])) or
            np.linalg.norm(A[-1] - B[-1]) < (Ra[-1] + Rb[-1])):
        # print(f"Collision detected between O1[{i}] and B[{j}]")
        print("与机械臂碰撞")
        return 3

    for i in range(num_links):
        for j in range(num_links):
            # if np.linalg.norm(O1[i] - O2[j]) < (R1[i] + R2[j]):
            #     # print(f"Collision detected between O1[{i}] and O2[{j}]")
            #     return 1
            # if np.linalg.norm(O1[i] - B[j]) < (R1[i] + Rb[j]):
            #     print(f"Collision detected between O1[{i}] and B[{j}]")
            #     return 1
            if np.linalg.norm(A[i] - O2[j]) < (Ra[i] + R2[j]):
                # print(f"Collision detected between A[{i}] and O2[{j}]")
                return 1
            if np.linalg.norm(A[i] - B[j]) < (Ra[i] + Rb[j]):
                # print(f"Collision detected between A[{i}] and B[{j}]")
                return 1

    # 如果没有检测到碰撞，返回 0
    return 0
