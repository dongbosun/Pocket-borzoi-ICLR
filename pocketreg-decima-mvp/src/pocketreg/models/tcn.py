"""Lightweight long-context TCN models for Pocket-Decima v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from pocketreg.models.small_cnn import count_parameters, estimate_model_size_mb


def _norm(channels: int, norm: str) -> nn.Module:
    if norm == "batchnorm":
        return nn.BatchNorm1d(channels)
    if norm == "groupnorm":
        groups = min(8, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm in {"none", None}:
        return nn.Identity()
    raise ValueError("norm must be batchnorm, groupnorm, or none.")


def _cycle(values: Iterable[int], n: int) -> list[int]:
    values = list(values)
    return [int(values[i % len(values)]) for i in range(n)]


class DepthwiseTCNBlock(nn.Module):
    """Residual depthwise-separable dilated Conv1d block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, norm: str):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
            ),
            _norm(channels, norm),
            nn.GELU(),
            nn.Conv1d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            _norm(channels, norm),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


@dataclass
class V2TCNConfig:
    """Configuration for targeted distillation TCN."""

    input_channels: int = 5
    channels: int = 96
    num_blocks: int = 8
    kernel_size: int = 7
    stem_kernel_size: int = 17
    stem_stride: int = 8
    pool_every: int = 2
    dropout: float = 0.1
    head_hidden: int = 192
    norm: str = "batchnorm"
    dilation_cycle: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    n_targets: int = 1
    n_replicates: int = 0
    n_aux: int = 0
    n_mid: int = 0
    n_residual: int = 0


class TargetedDistillationTCN(nn.Module):
    """5-channel DNA+mask TCN with target inference head and training-only aux heads."""

    def __init__(self, config: V2TCNConfig):
        super().__init__()
        self.config = config
        c = int(config.channels)
        self.stem = nn.Sequential(
            nn.Conv1d(
                int(config.input_channels),
                c,
                kernel_size=int(config.stem_kernel_size),
                stride=int(config.stem_stride),
                padding=int(config.stem_kernel_size) // 2,
            ),
            _norm(c, config.norm),
            nn.GELU(),
        )
        blocks: list[nn.Module] = []
        for i, dilation in enumerate(_cycle(config.dilation_cycle, config.num_blocks)):
            blocks.append(
                DepthwiseTCNBlock(
                    c,
                    kernel_size=int(config.kernel_size),
                    dilation=dilation,
                    dropout=float(config.dropout),
                    norm=config.norm,
                )
            )
            if int(config.pool_every) > 0 and (i + 1) % int(config.pool_every) == 0:
                blocks.append(nn.AvgPool1d(kernel_size=2, stride=2))
        self.trunk = nn.Sequential(*blocks)
        feat_dim = 2 * c
        self.final_head = self._head(feat_dim, int(config.n_targets))
        self.rep_head = self._head(feat_dim, int(config.n_targets) * int(config.n_replicates))
        self.aux_head = self._head(feat_dim, int(config.n_aux))
        self.mid_head = self._head(feat_dim, int(config.n_mid))
        self.residual_head = self._head(feat_dim, int(config.n_residual))

    def _head(self, in_dim: int, out_dim: int) -> nn.Module:
        if out_dim <= 0:
            return nn.Identity()
        return nn.Sequential(
            nn.Linear(in_dim, int(self.config.head_hidden)),
            nn.GELU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(int(self.config.head_hidden), out_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.trunk(self.stem(x))
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        out = {"final": self.final_head(z)}
        if int(self.config.n_replicates) > 0:
            rep = self.rep_head(z)
            out["rep"] = rep.view(x.shape[0], int(self.config.n_targets), int(self.config.n_replicates))
        if int(self.config.n_aux) > 0:
            out["aux"] = self.aux_head(z)
        if int(self.config.n_mid) > 0:
            out["mid"] = self.mid_head(z)
        if int(self.config.n_residual) > 0:
            out["residual"] = self.residual_head(z)
        return out


def preset_v2_config(preset: str, **overrides: object) -> V2TCNConfig:
    """Return a v2 model preset with overrides."""
    if preset == "tcn_tiny":
        cfg = V2TCNConfig(channels=64, num_blocks=6, head_hidden=128, kernel_size=7)
    elif preset == "tcn_small":
        cfg = V2TCNConfig(channels=96, num_blocks=8, head_hidden=192, kernel_size=7)
    elif preset == "tcn_1m":
        cfg = V2TCNConfig(channels=160, num_blocks=10, head_hidden=320, kernel_size=7)
    else:
        raise ValueError(f"Unknown v2 preset {preset!r}; expected tcn_tiny, tcn_small, or tcn_1m.")
    for key, value in overrides.items():
        if key == "preset" or value is None:
            continue
        if key == "dilation_cycle" and isinstance(value, list):
            value = tuple(int(v) for v in value)
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown v2 model config field {key!r}.")
        setattr(cfg, key, value)
    return cfg


def build_v2_model(model_config: dict[str, object]) -> TargetedDistillationTCN:
    """Build a v2 targeted distillation TCN from config."""
    config = dict(model_config)
    preset = str(config.pop("preset", "tcn_tiny"))
    return TargetedDistillationTCN(preset_v2_config(preset, **config))


def v2_model_summary(model: nn.Module) -> str:
    """Return a compact model summary."""
    return "\n".join(
        [
            model.__class__.__name__,
            str(model),
            f"trainable_parameters: {count_parameters(model)}",
            f"model_size_mb: {estimate_model_size_mb(model):.3f}",
        ]
    )
