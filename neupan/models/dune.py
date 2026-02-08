"""
DUNE (Deep Unfolded Neural Encoder): maps obstacle point flow → latent (mu, lambda).
"""

import sys
import torch
from math import inf
from typing import Optional

from neupan.models.point_net import ObsPointNet
from neupan.training.dune_trainer import DUNETrain
from neupan.utils.transforms import np_to_tensor, to_device
from neupan.utils.io import time_it, file_check, repeat_mk_dirs


class DUNE(torch.nn.Module):
    """DUNE layer wrapping the trained ObsPointNet.

    Args:
        receding:     MPC receding horizon steps.
        checkpoint:   Path to pretrained .pth weights.
        robot:        Robot instance (provides G, h matrices).
        dune_max_num: Maximum number of obstacle points.
        train_kwargs: DUNE training configuration dict.
    """

    def __init__(self, receding: int = 10, checkpoint=None, robot=None,
                 dune_max_num: int = 100, train_kwargs: dict = dict()) -> None:
        super().__init__()

        self.T = receding
        self.max_num = dune_max_num
        self.robot = robot

        self.G = np_to_tensor(robot.G)
        self.h = np_to_tensor(robot.h)
        self.edge_dim = self.G.shape[0]
        self.state_dim = self.G.shape[1]

        self.model = to_device(ObsPointNet(2, self.edge_dim))
        self._load_model(checkpoint, train_kwargs)

        self.obstacle_points = None
        self.min_distance = inf

    # ------------------------------------------------------------------
    @time_it("- dune forward")
    def forward(self, point_flow, R_list, obs_points_list=None):
        """Map point flow to latent distance features (mu, lam).

        Returns (mu_list, lam_list, sort_point_list) sorted by ascending distance.
        """
        if obs_points_list is None:
            obs_points_list = []

        mu_list, lam_list, sort_point_list = [], [], []
        self.obstacle_points = obs_points_list[0]

        total_points = torch.hstack(point_flow)
        with torch.no_grad():
            total_mu = self.model(total_points.T).T

        for idx in range(self.T + 1):
            n_pts = point_flow[idx].shape[1]
            mu = total_mu[:, idx * n_pts:(idx + 1) * n_pts]
            R = R_list[idx]
            p0 = point_flow[idx]
            lam = -R @ self.G.T @ mu

            if mu.ndim == 1:
                mu, lam = mu.unsqueeze(1), lam.unsqueeze(1)

            dist = self._objective_distance(mu, p0)
            if idx == 0:
                self.min_distance = torch.min(dist)

            order = torch.argsort(dist)
            mu_list.append(mu[:, order])
            lam_list.append(lam[:, order])
            sort_point_list.append(obs_points_list[idx][:, order])

        return mu_list, lam_list, sort_point_list

    def train_dune(self, train_kwargs):
        """Launch DUNE training pipeline."""
        name = train_kwargs.get("model_name", self.robot.name)
        ckpt_dir = repeat_mk_dirs(sys.path[0] + "/model/" + name)
        self.train_model = DUNETrain(self.model, self.G, self.h, ckpt_dir)
        self.full_model_name = self.train_model.start(**train_kwargs)
        print("Complete Training. Model saved in " + self.full_model_name)

    @property
    def points(self):
        return self.obstacle_points

    # ---- private ----
    def _objective_distance(self, mu, p0):
        temp = (self.G @ p0 - self.h).T.unsqueeze(2)
        muT = mu.T.unsqueeze(1)
        d = torch.squeeze(torch.bmm(muT, temp))
        return d.unsqueeze(0) if d.ndim == 0 else d

    def _load_model(self, checkpoint=None, train_kwargs=None):
        try:
            if checkpoint is None:
                raise FileNotFoundError
            path = file_check(checkpoint)
            self.model.load_state_dict(torch.load(path, map_location="cpu"))
            to_device(self.model)
            self.model.eval()
        except FileNotFoundError:
            if not train_kwargs:
                train_kwargs = dict()
            if train_kwargs.get("direct_train", False):
                print("train or test the model directly.")
                return
            if self._ask("Train DUNE model now? (Y/N): "):
                self.train_dune(train_kwargs)
                if self._ask("Continue case running? (Y/N): "):
                    self.model.load_state_dict(torch.load(self.full_model_name, map_location="cpu"))
                    to_device(self.model)
                    self.model.eval()
            else:
                raise FileNotFoundError("DUNE checkpoint not found.")

    @staticmethod
    def _ask(prompt):
        while True:
            choice = input(prompt).upper()
            if choice == "Y":
                return True
            if choice == "N":
                sys.exit()
            print("Please input Y or N.")
