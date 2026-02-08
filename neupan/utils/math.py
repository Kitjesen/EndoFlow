"""
Mathematical utility functions.
"""

import numpy as np
from math import pi, cos, sin, sqrt


def wrap_to_pi(rad: float, positive: bool = False) -> float:
    """Wrap angle to [-pi, pi]."""
    while rad > pi:
        rad -= 2 * pi
    while rad < -pi:
        rad += 2 * pi
    return rad if not positive else abs(rad)


# Backward-compatible alias
WrapToPi = wrap_to_pi


def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Euclidean distance between two 2D column vectors (2x1)."""
    return sqrt((point1[0, 0] - point2[0, 0]) ** 2 + (point1[1, 0] - point2[1, 0]) ** 2)


def get_transform(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract (translation, rotation) from state [x, y, theta] (3x1) or [x, y] (2x1)."""
    if state.shape == (2, 1):
        rot = np.eye(2)
        trans = state[0:2]
    else:
        rot = np.array([
            [cos(state[2, 0]), -sin(state[2, 0])],
            [sin(state[2, 0]),  cos(state[2, 0])],
        ])
        trans = state[0:2]
    return trans, rot


def cross_product_2d(o, a, b):
    """Cross product of vectors OA and OB (2D)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


# Backward-compatible alias
cross_product = cross_product_2d


def downsample_decimation(mat, m):
    """Uniformly downsample (dim x n) matrix to (dim x m)."""
    n = mat.shape[1]
    if m >= n:
        return mat
    indices = np.linspace(0, n - 1, m).astype(int)
    return mat[:, indices]
