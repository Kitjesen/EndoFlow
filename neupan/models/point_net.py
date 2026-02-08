"""
PointNet observation encoder: maps 2D obstacle points → latent distance features (mu).
"""

import torch
import torch.nn as nn


class ObsPointNet(nn.Module):
    """MLP encoder for obstacle point distance features.

    Architecture: 3 × (Linear → LayerNorm → Tanh → Linear → ReLU) + output Linear → ReLU

    Args:
        input_dim:  Input dimension (default 2: x, y).
        output_dim: Output dimension (number of polygon edges).
    """

    def __init__(self, input_dim: int = 2, output_dim: int = 4) -> None:
        super().__init__()
        h = 32
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, h), nn.LayerNorm(h), nn.Tanh(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.Tanh(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.Tanh(),
            nn.Linear(h, output_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.MLP(x)
