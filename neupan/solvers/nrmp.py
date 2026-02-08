"""
NRMP (Neural Regularized Motion Planner): differentiable convex optimization solver.
"""

import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from typing import Optional, List

from neupan.robot import robot
from neupan.utils.transforms import to_device, value_to_tensor, np_to_tensor
from neupan.utils.io import time_it


class NRMP(torch.nn.Module):
    """Differentiable MPC solver integrating DUNE distance features.

    Args:
        receding, step_time: MPC parameters.
        robot:        Robot instance.
        nrmp_max_num: Max obstacle points in QP.
        eta, d_max, d_min, q_s, p_u, ro_obs, bk: Cost / constraint tuning.
        solver:       Backend QP solver name (default "ECOS").
    """

    def __init__(self, receding, step_time, robot, nrmp_max_num=10,
                 eta=10.0, d_max=1.0, d_min=0.1, q_s=1.0, p_u=1.0,
                 ro_obs=400, bk=0.1, **kwargs) -> None:
        super().__init__()

        self.T, self.dt, self.robot = receding, step_time, robot
        self.G = np_to_tensor(robot.G)
        self.h = np_to_tensor(robot.h)
        self.max_num = nrmp_max_num
        self.no_obs = nrmp_max_num <= 0

        # tunable parameters
        self.eta   = value_to_tensor(eta, True)
        self.d_max = value_to_tensor(d_max, True)
        self.d_min = value_to_tensor(d_min, True)
        self.q_s   = value_to_tensor(q_s, True)
        self.p_u   = value_to_tensor(p_u, True)
        self.ro_obs, self.bk = ro_obs, bk

        self.adjust_parameters = (
            [self.q_s, self.p_u] if self.no_obs
            else [self.q_s, self.p_u, self.eta, self.d_max, self.d_min]
        )

        self._build_problem()
        self.obstacle_points = None
        self.solver = kwargs.get("solver", "ECOS")

    # ------------------------------------------------------------------
    @time_it("- nrmp forward")
    def forward(self, nom_s, nom_u, ref_s, ref_us,
                mu_list=None, lam_list=None, point_list=None):
        if point_list:
            self.obstacle_points = point_list[0][:, :self.max_num]

        params = self._param_values(nom_s, nom_u, ref_s, ref_us, mu_list, lam_list, point_list)
        sols = self._cvx_layer(*params, solver_args={"solve_method": self.solver})
        return sols[0], sols[1], (None if self.no_obs else sols[2])

    def update_adjust_parameters_value(self, **kw):
        """Update cost-weight parameters at runtime."""
        self.q_s   = value_to_tensor(kw.get("q_s",   self.q_s), True)
        self.p_u   = value_to_tensor(kw.get("p_u",   self.p_u), True)
        self.eta   = value_to_tensor(kw.get("eta",   self.eta), True)
        self.d_max = value_to_tensor(kw.get("d_max", self.d_max), True)
        self.d_min = value_to_tensor(kw.get("d_min", self.d_min), True)
        self.adjust_parameters = (
            [self.q_s, self.p_u] if self.no_obs
            else [self.q_s, self.p_u, self.eta, self.d_max, self.d_min]
        )

    @property
    def points(self):
        return self.obstacle_points

    # ---- CVXPY problem construction ----
    def _build_problem(self):
        # variables
        self._indep_dis = cp.Variable((1, self.T), name="distance", nonneg=True)
        self._indep_list = self.robot.define_variable(self.no_obs, self._indep_dis)
        # parameters
        self._para_list = (
            self.robot.state_parameter_define()
            + self.robot.coefficient_parameter_define(self.no_obs, self.max_num)
            + self._adjust_params()
        )
        # problem
        nav_c, nav_con = self._nav_cost()
        dune_c, dune_con = self._dune_cost()
        if self.no_obs:
            prob = cp.Problem(cp.Minimize(nav_c), nav_con)
        else:
            prob = cp.Problem(cp.Minimize(nav_c + dune_c), nav_con + dune_con)
        assert prob.is_dcp(dpp=True)
        self._cvx_layer = to_device(CvxpyLayer(prob, parameters=self._para_list, variables=self._indep_list))

    def _adjust_params(self):
        self._p_qs  = cp.Parameter(name="para_q_s", value=1.0)
        self._p_pu  = cp.Parameter(name="para_p_u", value=1.0)
        self._p_eta = cp.Parameter(value=8, nonneg=True, name="para_eta")
        self._p_dm  = cp.Parameter(name="para_d_max", value=1.0, nonneg=True)
        self._p_dn  = cp.Parameter(name="para_d_min", value=0.1, nonneg=True)
        return ([self._p_qs, self._p_pu] if self.no_obs
                else [self._p_qs, self._p_pu, self._p_eta, self._p_dm, self._p_dn])

    def _nav_cost(self):
        cost = self.robot.C0_cost(self._p_pu, self._p_qs)
        cost += 0.5 * self.bk * self.robot.proximal_cost()
        cons = self.robot.dynamics_constraint() + self.robot.bound_su_constraints()
        return cost, cons

    def _dune_cost(self):
        cost = -self._p_eta * cp.sum(self._indep_dis)
        cons = [self._indep_dis >= self._p_dn, self._indep_dis <= self._p_dm]
        if not self.no_obs:
            cost += self.robot.I_cost(self._indep_dis, self.ro_obs)
        return cost, cons

    # ---- parameter value assembly ----
    def _param_values(self, ns, nu, rs, rus, mu_l, lam_l, pt_l):
        sv = self.robot.generate_state_parameter_value(ns, nu, self.q_s * rs, self.p_u * rus)
        cv = self._coeff_values(mu_l, lam_l, pt_l)
        return sv + cv + self.adjust_parameters

    def _coeff_values(self, mu_l, lam_l, pt_l):
        if self.no_obs:
            return []
        fa = [to_device(torch.zeros(self.max_num, 2)) for _ in range(self.T)]
        fb = [to_device(torch.zeros(self.max_num, 1)) for _ in range(self.T)]
        if not mu_l:
            return fa + fb
        for t in range(self.T):
            mu, lam, pt = mu_l[t+1], lam_l[t+1], pt_l[t+1]
            a = lam.T
            b = torch.bmm(lam.T.unsqueeze(1), pt.T.unsqueeze(2)).squeeze(1) + mu.T @ self.h
            k = min(mu.shape[1], self.max_num)
            fa[t][:k] = a[:k];  fa[t][k:] = a[0]
            fb[t][:k] = b[:k];  fb[t][k:] = b[0]
        return fa + fb
