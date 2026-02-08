"""
PAN (Proximal Alternating-minimization Network): alternates DUNE + NRMP layers.
"""

import torch
from math import inf
from typing import Optional

from neupan.solvers.nrmp import NRMP
from neupan.models.dune import DUNE
from neupan.utils.transforms import to_device, tensor_to_np
from neupan.utils.math import downsample_decimation


class PAN(torch.nn.Module):
    """Core PAN network orchestrating DUNE (encoder) and NRMP (solver).

    Args:
        receding, step_time: MPC horizon parameters.
        robot:           Robot instance.
        iter_num:        Number of PAN alternating iterations.
        dune_max_num:    Max points for DUNE layer.
        nrmp_max_num:    Max points for NRMP solver.
        dune_checkpoint: Path to pretrained DUNE weights.
        iter_threshold:  Convergence threshold.
        adjust_kwargs:   Cost-weight parameters.
        train_kwargs:    DUNE training parameters.
    """

    def __init__(self, receding=10, step_time=0.1, robot=None, iter_num=2,
                 dune_max_num=100, nrmp_max_num=10, dune_checkpoint=None,
                 iter_threshold=0.1, adjust_kwargs=None, train_kwargs=None,
                 **kwargs) -> None:
        super().__init__()
        adjust_kwargs = adjust_kwargs or {}
        train_kwargs = train_kwargs or {}

        self.robot = robot
        self.T = receding
        self.dt = step_time
        self.iter_num = iter_num
        self.iter_threshold = iter_threshold
        self.nrmp_max_num = nrmp_max_num
        self.dune_max_num = dune_max_num
        self.no_obs = (nrmp_max_num == 0 or dune_max_num == 0)

        self.nrmp_layer = NRMP(
            receding, step_time, robot, nrmp_max_num,
            eta=adjust_kwargs.get("eta", 10.0), d_max=adjust_kwargs.get("d_max", 1.0),
            d_min=adjust_kwargs.get("d_min", 0.1), q_s=adjust_kwargs.get("q_s", 1.0),
            p_u=adjust_kwargs.get("p_u", 1.0), ro_obs=adjust_kwargs.get("ro_obs", 400),
            bk=adjust_kwargs.get("bk", 0.1), solver=adjust_kwargs.get("solver", "ECOS"),
        )

        self.dune_layer = (
            None if self.no_obs
            else DUNE(receding, dune_checkpoint, robot, dune_max_num, train_kwargs)
        )

        self._prev = [None, None, None, None]
        self._printed = False

    # ------------------------------------------------------------------
    def forward(self, nom_s, nom_u, ref_s, ref_us,
                obs_points=None, point_velocities=None):
        """Alternating forward: DUNE → NRMP → repeat.

        Returns (opt_state, opt_vel, opt_distance).
        """
        for _ in range(self.iter_num):
            if obs_points is not None and not self.no_obs:
                pf, Rl, opl = self._point_flow(nom_s, obs_points, point_velocities)
                mu_l, lam_l, sp_l = self.dune_layer(pf, Rl, opl)
            else:
                mu_l, lam_l, sp_l = [], [], []

            nom_s, nom_u, nom_d = self.nrmp_layer(
                nom_s, nom_u, ref_s, ref_us, mu_l, lam_l, sp_l
            )
            if self._converged(nom_s, nom_u, mu_l, lam_l):
                break

        return nom_s, nom_u, nom_d

    # ---- properties ----
    @property
    def min_distance(self):
        return inf if (self.dune_layer is None or self.no_obs) else self.dune_layer.min_distance

    @property
    def dune_points(self):
        return None if (self.dune_layer is None or self.no_obs) else tensor_to_np(self.dune_layer.points)

    @property
    def nrmp_points(self):
        return None if (self.nrmp_layer is None or self.no_obs) else tensor_to_np(self.nrmp_layer.points)

    # ---- private ----
    def _point_flow(self, nom_s, obs_points, point_vel=None):
        if point_vel is None:
            point_vel = torch.zeros_like(obs_points)
        if obs_points.shape[1] > self.dune_max_num:
            self._log_once(f"downsample obs {obs_points.shape[1]} → {self.dune_max_num}")
            obs_points = downsample_decimation(obs_points, self.dune_max_num)
            point_vel = downsample_decimation(point_vel, self.dune_max_num)

        opl, pfl, Rl = [], [], []
        for i in range(self.T + 1):
            pts = obs_points + i * point_vel * self.dt
            opl.append(pts)
            p0, R = self._to_robot_frame(nom_s[:, i], pts)
            pfl.append(p0)
            Rl.append(R)
        return pfl, Rl, opl

    @staticmethod
    def _to_robot_frame(state, obs_points):
        s = state.reshape(3, 1)
        trans, theta = s[0:2], s[2, 0]
        R = to_device(torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta),  torch.cos(theta)],
        ]))
        return R.T @ (obs_points - trans), R

    def _converged(self, ns, nu, mu_l, lam_l):
        if self._prev[0] is None:
            self._prev = [ns, nu, mu_l, lam_l]
            return False
        ds = torch.norm(ns - self._prev[0])
        du = torch.norm(nu - self._prev[1])
        if not mu_l or not self._prev[2]:
            diff = ds ** 2 + du ** 2
        else:
            k = min(mu_l[0].shape[1], self._prev[2][0].shape[1], self.nrmp_max_num)
            dm = torch.norm(torch.cat(mu_l)[:, :k] - torch.cat(self._prev[2])[:, :k]) / k
            dl = torch.norm(torch.cat(lam_l)[:, :k] - torch.cat(self._prev[3])[:, :k]) / k
            diff = dm ** 2 + dl ** 2
        self._prev = [ns, nu, mu_l, lam_l]
        return diff < self.iter_threshold

    def _log_once(self, msg):
        if not self._printed:
            print(msg)
            self._printed = True
