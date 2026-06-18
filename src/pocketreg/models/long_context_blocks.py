"""Parameter-efficient long-context convolution blocks."""

from __future__ import annotations


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise ImportError("PocketBorzoiV2 requires PyTorch.") from exc
    return torch, nn


class DepthwiseSeparableConv1d:
    def __new__(cls, channels: int, kernel_size: int = 7, dilation: int = 1, dropout: float = 0.0):
        torch, nn = require_torch()

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                pad = dilation * (kernel_size // 2)
                self.net = nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, groups=channels),
                    nn.Conv1d(channels, channels, 1),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )

            def forward(self, x):
                return x + self.net(x)

        return _Block()
