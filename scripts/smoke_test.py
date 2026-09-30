#!/usr/bin/env python
"""Minimal dependency and gradient check for the local G²MAF package."""

import sys
from pathlib import Path

import torch

# Allow `python scripts/smoke_test.py` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g2maf import CentralizedBehaviorCritic


def main():
    torch.manual_seed(0)
    critic = CentralizedBehaviorCritic(obs_dim=6, act_dim=4, hidden=32)
    obs = torch.randn(3, 6)
    action = torch.randn(3, 4, requires_grad=True)
    value = critic(obs, action).sum()
    (gradient,) = torch.autograd.grad(value, action)
    if gradient.shape != action.shape or not torch.isfinite(gradient).all():
        raise RuntimeError("G²MAF critic gradient check failed")
    print("G²MAF smoke test passed: critic forward and joint-action gradient are valid.")


if __name__ == "__main__":
    main()
