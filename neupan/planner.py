"""
EndoFlow Planner: main entry point — orchestrates robot, path, and PAN modules.
"""

import yaml
import torch
import numpy as np
from math import cos, sin

from neupan.robot import robot
from neupan.planning import InitialPath
from neupan.models import PAN
from neupan.utils.transforms import np_to_tensor, tensor_to_np
from neupan.utils.io import time_it, file_check
import neupan.utils.transforms as _transforms


class Planner(torch.nn.Module):
    """End-to-end motion planner.

    Create via YAML:  ``planner = Planner.init_from_yaml('planner.yaml')``
    """

    def __init__(self, receding=10, step_time=0.1, ref_speed=4.0, device="cpu",
                 robot_kwargs=None, ipath_kwargs=None, pan_kwargs=None,
                 adjust_kwargs=None, train_kwargs=None, **kw):
        super().__init__()
        self.T, self.dt, self.ref_speed = receding, step_time, ref_speed

        _transforms.device = torch.device(device)
        _transforms.time_print = kw.get("time_print", False)
        self.collision_threshold = kw.get("collision_threshold", 0.1)

        self.cur_vel_array = np.zeros((2, self.T))
        self.robot = robot(receding, step_time, **(robot_kwargs or {}))
        self.ipath = InitialPath(receding, step_time, ref_speed, self.robot, **(ipath_kwargs or {}))

        pan_kwargs = dict(pan_kwargs or {})
        pan_kwargs["adjust_kwargs"] = adjust_kwargs or {}
        pan_kwargs["train_kwargs"] = train_kwargs or {}
        self._train_kw = train_kwargs or {}
        self.pan = PAN(receding, step_time, self.robot, **pan_kwargs)

        self.info = {"stop": False, "arrive": False, "collision": False}

    @classmethod
    def init_from_yaml(cls, yaml_file, **kw):
        path = file_check(yaml_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cfg.update(kw)
        cfg["robot_kwargs"]  = cfg.pop("robot",  {})
        cfg["ipath_kwargs"]  = cfg.pop("ipath",  {})
        cfg["pan_kwargs"]    = cfg.pop("pan",    {})
        cfg["adjust_kwargs"] = cfg.pop("adjust", {})
        cfg["train_kwargs"]  = cfg.pop("train",  {})
        return cls(**cfg)

    # ------------------------------------------------------------------
    @time_it("planner forward")
    def forward(self, state, points, velocities=None):
        """One planning step → (action (2×1), info dict)."""
        assert state.shape[0] >= 3
        if self.ipath.check_arrive(state):
            self.info["arrive"] = True
            return np.zeros((2, 1)), self.info

        nom = self.ipath.generate_nom_ref_state(state, self.cur_vel_array, self.ref_speed)
        nom_t = [np_to_tensor(n) for n in nom]
        obs_t = np_to_tensor(points) if points is not None else None
        vel_t = np_to_tensor(velocities) if velocities is not None else None

        opt_s, opt_v, opt_d = self.pan(*nom_t, obs_t, vel_t)
        opt_s_np, opt_v_np = tensor_to_np(opt_s), tensor_to_np(opt_v)
        self.cur_vel_array = opt_v_np

        self.info.update(
            state_tensor=opt_s, vel_tensor=opt_v, distance_tensor=opt_d,
            ref_state_tensor=nom_t[2], ref_speed_tensor=nom_t[3],
            ref_state_list=[c[:, np.newaxis] for c in nom[2].T],
            opt_state_list=[c[:, np.newaxis] for c in opt_s_np.T],
        )

        if self.min_distance < self.collision_threshold:
            self.info["stop"] = True
            return np.zeros((2, 1)), self.info
        self.info["stop"] = False

        action = opt_v_np[:, 0:1]
        if self.robot.kinematics == "omni":
            v = opt_v_np[:, 0:1]
            action = np.array([[v[0, 0] * cos(v[1, 0])], [v[0, 0] * sin(v[1, 0])]])
            self.info["omni_linear_speed"] = v[0, 0]
            self.info["omni_orientation"] = v[1, 0]

        return action, self.info

    # ---- convenience API ----
    def train_dune(self):                          self.pan.dune_layer.train_dune(self._train_kw)
    def reset(self):                               self.ipath.reset(); self.info = {"stop": False, "arrive": False, "collision": False}; self.cur_vel_array[:] = 0
    def set_initial_path(self, path):              self.ipath.set_initial_path(path)
    def set_initial_path_from_state(self, state):  self.ipath.init_check(state)
    def set_reference_speed(self, speed):           self.ipath.ref_speed = self.ref_speed = speed
    def update_initial_path_from_goal(self, s, g): self.ipath.update_initial_path_from_goal(s, g)
    def update_initial_path_from_waypoints(self, w): self.ipath.set_ipath_with_waypoints(w)
    def update_adjust_parameters(self, **kw):      self.pan.nrmp_layer.update_adjust_parameters_value(**kw)

    # ---- properties ----
    @property
    def min_distance(self):      return self.pan.min_distance
    @property
    def dune_points(self):       return self.pan.dune_points
    @property
    def nrmp_points(self):       return self.pan.nrmp_points
    @property
    def initial_path(self):      return self.ipath.initial_path
    @property
    def adjust_parameters(self): return self.pan.nrmp_layer.adjust_parameters
    @property
    def waypoints(self):         return self.ipath.waypoints
    @property
    def opt_trajectory(self):    return self.info.get("opt_state_list", [])
    @property
    def ref_trajectory(self):    return self.info.get("ref_state_list", [])
