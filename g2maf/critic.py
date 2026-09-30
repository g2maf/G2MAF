"""Centralized behavior critic used by G²MAF."""

import torch
from torch import nn


class CentralizedBehaviorCritic(nn.Module):
    """MLP critic over a flattened joint observation and joint action."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)
