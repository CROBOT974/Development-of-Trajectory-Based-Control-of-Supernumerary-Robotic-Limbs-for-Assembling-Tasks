import numpy as np
import matplotlib.pyplot as plt
import math


def compute_bezier_curve(points, t):
    n = len(points) - 1
    v0 = 0
    v1 = 0
    v2 = 0
    v3 = 0
    v4 = 0
    v5 = 0
    v6 = 0
    v7 = 0
    for i in range(n + 1):
        coefficient = math.comb(n, i) * (1 - t) ** (n - i) * t ** i
        v0 += coefficient * points[i][0]
        v1 += coefficient * points[i][1]
        v2 += coefficient * points[i][2]
        v3 += coefficient * points[i][3]
        v4 += coefficient * points[i][4]
        v5 += coefficient * points[i][5]
        v6 += coefficient * points[i][6]
        v7 += coefficient * points[i][7]

    return v0, v1, v2, v3, v4, v5, v6, v7


def bezier_curve(control_points, seg=2000):
    t = np.linspace(0, 1, seg)
    curve_points = np.array([compute_bezier_curve(control_points, ti) for ti in t])
    return curve_points


if __name__ == '__main__':
    control_points_ = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 3, 1, 1, 1, 1, 1, 1], [4, 2, 1.5, 1, 1, 1, 1, 1], [5, 5, 2, 1, 1, 1, 1, 1]])
    result = bezier_curve(control_points_, seg=10)
    print(result)
