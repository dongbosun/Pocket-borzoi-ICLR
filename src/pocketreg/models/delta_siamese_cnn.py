"""Siamese CNN student for SNP delta distillation."""

from __future__ import annotations


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise ImportError("SiameseDeltaCNN requires PyTorch. Install torch for student training.") from exc
    return torch, nn


class SiameseDeltaCNN:
    def __new__(
        cls,
        metadata_dim: int = 0,
        channels: int = 64,
        num_blocks: int = 5,
        stem_stride: int = 4,
        dropout: float = 0.1,
        head_hidden: int = 128,
    ):
        torch, nn = _require_torch()

        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                layers: list[nn.Module] = [
                    nn.Conv1d(4, channels, kernel_size=11, stride=stem_stride, padding=5),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                ]
                for _ in range(num_blocks):
                    layers.extend(
                        [
                            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
                            nn.BatchNorm1d(channels),
                            nn.GELU(),
                            nn.MaxPool1d(2),
                        ]
                    )
                layers.extend([nn.AdaptiveAvgPool1d(1), nn.Flatten()])
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        class _SiameseDeltaCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _Encoder()
                feature_dim = channels * 4 + metadata_dim
                self.head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim, head_hidden),
                    nn.GELU(),
                    nn.Linear(head_hidden, 1),
                )

            def forward(self, seq_ref, seq_alt, metadata=None):
                h_ref = self.encoder(seq_ref)
                h_alt = self.encoder(seq_alt)
                parts = [h_alt - h_ref, torch.abs(h_alt - h_ref), h_ref, h_alt]
                if metadata is not None:
                    parts.append(metadata)
                features = torch.cat(parts, dim=-1)
                return self.head(features).squeeze(-1)

        return _SiameseDeltaCNN()
