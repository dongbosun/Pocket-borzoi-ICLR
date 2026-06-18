"""Small CNN student for scalar track distillation.

The model requires PyTorch for real training. Importing this module stays cheap
on machines where PyTorch is not installed; construction raises an actionable
error instead.
"""

from __future__ import annotations


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise ImportError("SmallCNN requires PyTorch. Install torch for student training.") from exc
    return torch, nn


class SmallCNN:
    def __new__(
        cls,
        channels: int = 64,
        num_blocks: int = 6,
        stem_stride: int = 8,
        dropout: float = 0.1,
        head_hidden: int = 128,
    ):
        torch, nn = _require_torch()

        class _SmallCNN(nn.Module):
            def __init__(self):
                super().__init__()
                layers: list[nn.Module] = [
                    nn.Conv1d(4, channels, kernel_size=15, stride=stem_stride, padding=7),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                ]
                for _ in range(num_blocks):
                    layers.extend(
                        [
                            nn.Conv1d(channels, channels, kernel_size=7, padding=3),
                            nn.BatchNorm1d(channels),
                            nn.GELU(),
                            nn.MaxPool1d(2),
                        ]
                    )
                self.encoder = nn.Sequential(*layers)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Dropout(dropout),
                    nn.Linear(channels, head_hidden),
                    nn.GELU(),
                    nn.Linear(head_hidden, 1),
                )

            def forward(self, x):
                return self.head(self.encoder(x)).squeeze(-1)

        return _SmallCNN()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
