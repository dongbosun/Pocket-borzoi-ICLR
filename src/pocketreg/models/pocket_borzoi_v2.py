"""Pocket-Borzoi v2 long-context multi-head student."""

from __future__ import annotations

from pocketreg.models.long_context_blocks import DepthwiseSeparableConv1d, require_torch


class PocketBorzoiV2:
    def __new__(
        cls,
        input_channels: int = 5,
        channels: int = 96,
        num_blocks: int = 8,
        stem_stride: int = 16,
        dropout: float = 0.1,
        head_hidden: int = 192,
        fold_dim: int = 2,
        profile_dim: int = 16,
        aux_dim: int = 8,
        middle_dim: int = 27,
    ):
        torch, nn = require_torch()

        class _PocketBorzoiV2(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv1d(input_channels, channels, kernel_size=17, stride=stem_stride, padding=8),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.MaxPool1d(2),
                )
                dilations = [1, 2, 4, 8, 16, 32, 64, 128]
                blocks = []
                for i in range(num_blocks):
                    blocks.append(
                        DepthwiseSeparableConv1d(
                            channels=channels,
                            kernel_size=7,
                            dilation=dilations[i % len(dilations)],
                            dropout=dropout,
                        )
                    )
                    if i % 2 == 1:
                        blocks.append(nn.MaxPool1d(2))
                self.trunk = nn.Sequential(*blocks)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.max_pool = nn.AdaptiveMaxPool1d(1)
                pooled = channels * 2

                def head(out_dim: int):
                    return nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(pooled, head_hidden),
                        nn.GELU(),
                        nn.Linear(head_hidden, out_dim),
                    )

                self.primary_head = head(1)
                self.fold_head = head(fold_dim) if fold_dim > 0 else None
                self.profile_pca_head = head(profile_dim) if profile_dim > 0 else None
                self.aux_pca_head = head(aux_dim) if aux_dim > 0 else None
                self.middle_proj_head = head(middle_dim) if middle_dim > 0 else None

            def encode(self, x):
                z = self.trunk(self.stem(x))
                return torch.cat([self.pool(z).flatten(1), self.max_pool(z).flatten(1)], dim=1)

            def forward(self, x):
                h = self.encode(x)
                out = {"primary": self.primary_head(h).squeeze(-1)}
                if self.fold_head is not None:
                    out["fold"] = self.fold_head(h)
                if self.profile_pca_head is not None:
                    out["profile_pca"] = self.profile_pca_head(h)
                if self.aux_pca_head is not None:
                    out["aux_pca"] = self.aux_pca_head(h)
                if self.middle_proj_head is not None:
                    out["middle_proj"] = self.middle_proj_head(h)
                return out

            def inference_state_dict(self):
                return {
                    "stem": self.stem.state_dict(),
                    "trunk": self.trunk.state_dict(),
                    "primary_head": self.primary_head.state_dict(),
                }

        return _PocketBorzoiV2()
