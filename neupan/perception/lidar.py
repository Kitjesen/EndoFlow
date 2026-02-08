"""
Lidar point-cloud pre-processing: scan → 2D obstacle points (with optional velocity).
"""

import numpy as np
from math import cos, sin
from neupan.utils.math import get_transform


def scan_to_points(state, scan, scan_offset=(0, 0, 0),
                   angle_range=(-np.pi, np.pi), down_sample=1):
    """Convert lidar scan to 2D point cloud in global frame.

    Returns (2, n) array or None.
    """
    pts = []
    ranges = np.array(scan["ranges"])
    angles = np.linspace(scan["angle_min"], scan["angle_max"], len(ranges))
    for r, a in zip(ranges, angles):
        if scan["range_min"] < r < scan["range_max"] - 0.02 and angle_range[0] < a < angle_range[1]:
            pts.append(np.array([[r * cos(a)], [r * sin(a)]]))
    if not pts:
        return None
    arr = np.hstack(pts)
    st, sR = get_transform(np.c_[list(scan_offset)])
    arr = sR @ arr + st
    t, R = get_transform(state)
    return (R @ arr + t)[:, ::down_sample]


def scan_to_points_with_velocity(state, scan, scan_offset=(0, 0, 0),
                                  angle_range=(-np.pi, np.pi), down_sample=1):
    """Like scan_to_points but also returns per-point velocity (2, n).

    Returns (points, velocities) or (None, None).
    """
    pts, vels = [], []
    ranges = np.array(scan["ranges"])
    angles = np.linspace(scan["angle_min"], scan["angle_max"], len(ranges))
    vel_data = scan.get("velocity", np.zeros((2, len(ranges))))
    for i, (r, a) in enumerate(zip(ranges, angles)):
        if scan["range_min"] <= r < scan["range_max"] - 0.02 and angle_range[0] < a < angle_range[1]:
            pts.append(np.array([[r * cos(a)], [r * sin(a)]]))
            vels.append(vel_data[:, i:i+1])
    if not pts:
        return None, None
    arr = np.hstack(pts)
    st, sR = get_transform(np.c_[list(scan_offset)])
    arr = sR.T @ (arr - st)
    t, R = get_transform(state)
    return (R @ arr + t)[:, ::down_sample], np.hstack(vels)[:, ::down_sample]
