"""Loss builders."""

from __future__ import annotations

from torch import nn


def build_loss(name: str, huber_delta: float = 1.0) -> nn.Module:
    """Build a scalar regression loss."""
    name = name.lower()
    if name == "huber":
        return nn.HuberLoss(delta=float(huber_delta))
    if name == "mse":
        return nn.MSELoss()
    raise ValueError("loss must be huber or mse.")
