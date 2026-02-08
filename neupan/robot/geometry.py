"""
Robot geometry: convex polygon shape and inequality constraints G x <= h.
"""

import numpy as np
from neupan.utils.geometry import gen_inequal_from_vertex


class RobotGeometry:
    """Convex polygon robot shape, providing G and h matrices."""

    def __init__(self, vertices=None, length=None, width=None, wheelbase=None):
        self.shape = None
        self.length, self.width, self.wheelbase = length, width, wheelbase
        self.vertices = self._compute(vertices, length, width, wheelbase)
        self.G, self.h = gen_inequal_from_vertex(self.vertices)

    def _compute(self, vertices, length, width, wheelbase):
        if vertices is not None:
            v = np.array(vertices).T if isinstance(vertices, list) else vertices
        else:
            self.shape = "rectangle"
            v = self._rect(length, width, wheelbase)
        assert v.shape[1] >= 3
        return v

    @staticmethod
    def _rect(length, width, wheelbase=None):
        wb = wheelbase or 0
        sx, sy = -(length - wb) / 2, -width / 2
        return np.hstack([
            [[sx], [sy]], [[sx+length], [sy]],
            [[sx+length], [sy+width]], [[sx], [sy+width]],
        ])
