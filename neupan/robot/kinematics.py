"""
Linearized robot kinematics for MPC: x_{t+1} = A x_t + B u_t + C.
"""

import torch
from math import sin, cos, tan
from neupan.utils.transforms import to_device


def linear_ackermann_model(nom_st, nom_ut, dt, L):
    phi, v, psi = nom_st[2, 0], nom_ut[0, 0], nom_ut[1, 0]
    A = torch.Tensor([[1, 0, -v*dt*sin(phi)], [0, 1, v*dt*cos(phi)], [0, 0, 1]])
    B = torch.Tensor([[cos(phi)*dt, 0], [sin(phi)*dt, 0],
                       [tan(psi)*dt/L, v*dt/(L*cos(psi)**2)]])
    C = torch.Tensor([[phi*v*sin(phi)*dt], [-phi*v*cos(phi)*dt],
                       [-psi*v*dt/(L*cos(psi)**2)]])
    return to_device(A), to_device(B), to_device(C)


def linear_diff_model(nom_st, nom_ut, dt):
    phi, v = nom_st[2, 0], nom_ut[0, 0]
    A = torch.Tensor([[1, 0, -v*dt*sin(phi)], [0, 1, v*dt*cos(phi)], [0, 0, 1]])
    B = torch.Tensor([[cos(phi)*dt, 0], [sin(phi)*dt, 0], [0, dt]])
    C = torch.Tensor([[phi*v*sin(phi)*dt], [-phi*v*cos(phi)*dt], [0]])
    return to_device(A), to_device(B), to_device(C)


def linear_omni_model(nom_ut, dt):
    phi, v = nom_ut[1, 0], nom_ut[0, 0]
    A = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    B = torch.Tensor([[cos(phi)*dt, -v*sin(phi)*dt],
                       [sin(phi)*dt, v*cos(phi)*dt], [0, 0]])
    C = torch.Tensor([[phi*v*sin(phi)*dt], [-phi*v*cos(phi)*dt], [0]])
    return to_device(A), to_device(B), to_device(C)
