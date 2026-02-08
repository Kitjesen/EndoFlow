"""
Tensor / NumPy / coordinate conversion utilities.
"""

import torch
import numpy as np

device = torch.device("cpu")
time_print = False


def np_to_tensor(array):
    """Convert numpy array or scalar to float32 tensor on current device."""
    if np.isscalar(array):
        return torch.tensor(array).type(torch.float32).to(device)
    return torch.from_numpy(array).type(torch.float32).to(device)


def tensor_to_np(tensor):
    """Convert tensor to numpy array."""
    if tensor is None:
        return None
    return tensor.cpu().detach().numpy()


def value_to_tensor(value, requires_grad=False):
    """Convert a Python value to float32 tensor."""
    if value is None:
        return None
    return torch.tensor(value, dtype=torch.float32, requires_grad=requires_grad).to(device)


def to_device(tensor):
    """Move tensor/module to current device."""
    return tensor.to(device)
