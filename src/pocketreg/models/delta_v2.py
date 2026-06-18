"""Delta student with regression and effect-classification heads."""

from __future__ import annotations


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise ImportError("DeltaSiameseV2 requires PyTorch. Install torch for training.") from exc
    return torch, nn


class DeltaSiameseV2:
    """Siamese sequence encoder for sparse Borzoi delta distillation.

    The model intentionally stays small. Most of the v2 change is in target
    shaping and sampling, not a huge architecture jump.
    """

    def __new__(
        cls,
        metadata_dim: int = 0,
        channels: int = 96,
        num_blocks: int = 6,
        stem_stride: int = 8,
        dropout: float = 0.1,
        head_hidden: int = 192,
    ):
        torch, nn = _require_torch()

        class _Block(nn.Module):
            def __init__(self, dilation: int):
                super().__init__()
                pad = 3 * dilation
                self.net = nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=7, padding=pad, dilation=dilation, groups=channels),
                    nn.Conv1d(channels, channels, kernel_size=1),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )

            def forward(self, x):
                return x + self.net(x)

        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                layers: list[nn.Module] = [
                    nn.Conv1d(4, channels, kernel_size=17, stride=stem_stride, padding=8),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.MaxPool1d(2),
                ]
                for i in range(num_blocks):
                    layers.append(_Block(2 ** min(i, 7)))
                    if i % 2 == 1:
                        layers.append(nn.MaxPool1d(2))
                self.net = nn.Sequential(*layers)
                self.avg_pool = nn.AdaptiveAvgPool1d(1)
                self.max_pool = nn.AdaptiveMaxPool1d(1)

            def forward(self, x):
                h = self.net(x)
                return torch.cat([self.avg_pool(h).squeeze(-1), self.max_pool(h).squeeze(-1)], dim=-1)

        class _DeltaSiameseV2(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _Encoder()
                encoded = channels * 2
                feature_dim = encoded * 4 + metadata_dim
                self.shared = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim, head_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.delta_head = nn.Linear(head_hidden, 1)
                self.effect_head = nn.Linear(head_hidden, 1)

            def forward(self, seq_ref, seq_alt, metadata=None):
                h_ref = self.encoder(seq_ref)
                h_alt = self.encoder(seq_alt)
                parts = [h_alt - h_ref, torch.abs(h_alt - h_ref), h_ref, h_alt]
                if metadata is not None:
                    parts.append(metadata)
                h = self.shared(torch.cat(parts, dim=-1))
                return {
                    "delta": self.delta_head(h).squeeze(-1),
                    "effect_logit": self.effect_head(h).squeeze(-1),
                }

        return _DeltaSiameseV2()
