"""
Convex polygon geometry utilities (inequality constraints, convexity check).
"""

import numpy as np
from neupan.utils.math import cross_product_2d


def gen_inequal_from_vertex(vertex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate half-plane inequalities Gx <= h for a convex polygon.

    Args:
        vertex: Polygon vertices (2 x N).

    Returns:
        (G, h) matrices, or (None, None) if not convex.
    """
    convex_flag, order = is_convex_and_ordered(vertex)
    if not convex_flag:
        print("The polygon constructed by vertex is not convex.")
        return None, None

    if order == "CW":
        first = vertex[:, 0:1]
        rest = vertex[:, 1:]
        vertex = np.hstack([first, rest[:, ::-1]])

    num = vertex.shape[1]
    G = np.zeros((num, 2))
    h = np.zeros((num, 1))

    for i in range(num):
        pre = vertex[:, i]
        nxt = vertex[:, (i + 1) % num]
        diff = nxt - pre
        a, b = diff[1], -diff[0]
        G[i, 0], G[i, 1] = a, b
        h[i, 0] = a * pre[0] + b * pre[1]

    return G, h


def is_convex_and_ordered(points: np.ndarray) -> tuple[bool, str | None]:
    """Check if polygon is convex and return winding order ('CW' or 'CCW')."""
    n = points.shape[1]
    if n < 3:
        return False, None

    direction = 0
    for i in range(n):
        o = points[:, i]
        a = points[:, (i + 1) % n]
        b = points[:, (i + 2) % n]
        cross = cross_product_2d(o, a, b)
        if cross != 0:
            if direction == 0:
                direction = 1 if cross > 0 else -1
            elif (cross > 0 and direction < 0) or (cross < 0 and direction > 0):
                return False, None

    return True, ("CCW" if direction > 0 else "CW")
