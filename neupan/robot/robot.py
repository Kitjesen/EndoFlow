"""
Robot class: combines geometry, kinematics, and CVXPY optimization interface.
"""

from math import inf
import numpy as np
from typing import Optional, Union
import cvxpy as cp

from neupan.utils.transforms import to_device
from neupan.robot.geometry import RobotGeometry
from neupan.robot.kinematics import linear_ackermann_model, linear_diff_model, linear_omni_model


class robot:
    def __init__(self, receding=10, step_time=0.1, kinematics=None,
                 vertices=None, max_speed=(inf, inf), max_acce=(inf, inf),
                 wheelbase=None, length=None, width=None, **kw):
        if kinematics is None: raise ValueError("kinematics is required")
        self._geo = RobotGeometry(vertices, length, width, wheelbase)
        self.T, self.dt, self.L = receding, step_time, wheelbase
        self.kinematics = kinematics
        self.max_speed = np.c_[list(max_speed)] if not isinstance(max_speed, np.ndarray) else max_speed
        self.max_acce  = np.c_[list(max_acce)]  if not isinstance(max_acce, np.ndarray)  else max_acce
        if kinematics == "acker" and self.max_speed[1] >= 1.57:
            print(f"Warning: acker max steering {self.max_speed[1]} rad → clamped to 1.57")
            self.max_speed[1] = 1.57
        self.speed_bound = self.max_speed
        self.acce_bound  = self.max_acce * self.dt
        self.name = kw.get("name", f"{kinematics}_robot_default")

    # geometry delegation
    @property
    def vertices(self): return self._geo.vertices
    @property
    def G(self): return self._geo.G
    @property
    def h(self): return self._geo.h
    @property
    def shape(self): return self._geo.shape

    # ---- CVXPY interface ----
    def define_variable(self, no_obs=False, indep_dis=None):
        self.indep_s = cp.Variable((3, self.T+1), name="state")
        self.indep_u = cp.Variable((2, self.T), name="vel")
        return [self.indep_s, self.indep_u] if no_obs else [self.indep_s, self.indep_u, indep_dis]

    def state_parameter_define(self):
        self.para_s       = cp.Parameter((3, self.T+1), name="para_state")
        self.para_gamma_a = cp.Parameter((3, self.T+1), name="para_gamma_a")
        self.para_gamma_b = cp.Parameter((self.T,),     name="para_gamma_b")
        self.para_A_list = [cp.Parameter((3, 3), name=f"para_A_{t}") for t in range(self.T)]
        self.para_B_list = [cp.Parameter((3, 2), name=f"para_B_{t}") for t in range(self.T)]
        self.para_C_list = [cp.Parameter((3, 1), name=f"para_C_{t}") for t in range(self.T)]
        return [self.para_s, self.para_gamma_a, self.para_gamma_b] + self.para_A_list + self.para_B_list + self.para_C_list

    def coefficient_parameter_define(self, no_obs=False, max_num=10):
        if no_obs:
            self.para_gamma_c = self.para_zeta_a = []
        else:
            self.para_gamma_c = [cp.Parameter((max_num, 2), value=np.zeros((max_num, 2)), name=f"gc{i}") for i in range(self.T)]
            self.para_zeta_a  = [cp.Parameter((max_num, 1), value=np.zeros((max_num, 1)), name=f"za{i}") for i in range(self.T)]
        return self.para_gamma_c + self.para_zeta_a

    def C0_cost(self, p_pu, p_qs):
        du = p_pu * self.indep_u[0, :] - self.para_gamma_b
        ds = p_qs * self.indep_s - self.para_gamma_a
        sc = cp.sum_squares(ds[0:2]) if self.kinematics == "omni" else cp.sum_squares(ds)
        return sc + cp.sum_squares(du)

    def proximal_cost(self):
        return cp.sum_squares(self.indep_s - self.para_s)

    def I_cost(self, indep_dis, ro_obs):
        Il = [self.para_gamma_c[t] @ self.indep_s[0:2, t+1:t+2] - self.para_zeta_a[t] - indep_dis[0, t] for t in range(self.T)]
        return 0.5 * ro_obs * cp.sum_squares(cp.neg(cp.vstack(Il)))

    def dynamics_constraint(self):
        tl = [self.para_A_list[t] @ self.indep_s[:, t:t+1] + self.para_B_list[t] @ self.indep_u[:, t:t+1] + self.para_C_list[t] for t in range(self.T)]
        return [self.indep_s[:, 1:] == cp.hstack(tl)]

    def bound_su_constraints(self):
        return [
            cp.abs(self.indep_u[:, 1:] - self.indep_u[:, :-1]) <= self.acce_bound,
            cp.abs(self.indep_u) <= self.speed_bound,
            self.indep_s[:, 0:1] == self.para_s[:, 0:1],
        ]

    def generate_state_parameter_value(self, nom_s, nom_u, qs_ref_s, pu_ref_us):
        sv = [nom_s, qs_ref_s, pu_ref_us]
        Al, Bl, Cl = [], [], []
        for t in range(self.T):
            st, ut = nom_s[:, t:t+1], nom_u[:, t:t+1]
            if self.kinematics == "acker":   A, B, C = linear_ackermann_model(st, ut, self.dt, self.L)
            elif self.kinematics == "diff":  A, B, C = linear_diff_model(st, ut, self.dt)
            elif self.kinematics == "omni":  A, B, C = linear_omni_model(ut, self.dt)
            else: raise ValueError(f"Unsupported kinematics: {self.kinematics}")
            Al.append(A); Bl.append(B); Cl.append(C)
        return sv + Al + Bl + Cl
