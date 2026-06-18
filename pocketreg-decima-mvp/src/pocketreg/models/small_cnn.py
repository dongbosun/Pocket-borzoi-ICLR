"""Configurable small 1D CNNs for DNA-to-scalar regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


def _norm_layer(channels: int, norm: str) -> nn.Module:
    if norm == "batchnorm":
        return nn.BatchNorm1d(channels)
    if norm == "groupnorm":
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm in {"none", None}:
        return nn.Identity()
    raise ValueError("norm must be batchnorm, groupnorm, or none.")


class ResidualBlock(nn.Module):
    """Dilation-aware residual Conv1d block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, norm: str):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            _norm_layer(channels, norm),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            _norm_layer(channels, norm),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


@dataclass
class SmallCNNConfig:
    """Configuration for SmallDNARegressor."""

    channels: int = 64
    num_blocks: int = 6
    kernel_size: int = 3
    stem_stride: int = 8
    pool_every: int = 2
    dropout: float = 0.1
    head_hidden: int = 128
    norm: str = "batchnorm"
    dilation_cycle: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)


class SmallDNARegressor(nn.Module):
    """Small CNN that maps [batch, 4, length] DNA one-hot input to [batch]."""

    def __init__(self, config: SmallCNNConfig):
        super().__init__()
        self.config = config
        c = config.channels
        self.stem = nn.Sequential(
            nn.Conv1d(4, c, kernel_size=15, stride=config.stem_stride, padding=7),
            _norm_layer(c, config.norm),
            nn.GELU(),
        )
        blocks: list[nn.Module] = []
        dilations = _cycle(config.dilation_cycle, config.num_blocks)
        for i, dilation in enumerate(dilations):
            blocks.append(
                ResidualBlock(
                    channels=c,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                    norm=config.norm,
                )
            )
            if config.pool_every > 0 and (i + 1) % config.pool_every == 0:
                blocks.append(nn.AvgPool1d(kernel_size=2, stride=2))
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(2 * c, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.stem(x)
        x = self.body(x)
        mean_pool = x.mean(dim=-1)
        max_pool = x.amax(dim=-1)
        return self.head(torch.cat([mean_pool, max_pool], dim=1)).squeeze(-1)


def _cycle(values: Iterable[int], n: int) -> list[int]:
    values = list(values)
    return [values[i % len(values)] for i in range(n)]


def preset_config(preset: str, **overrides: object) -> SmallCNNConfig:
    """Return a named model preset with optional field overrides."""
    if preset == "tiny_100k":
        cfg = SmallCNNConfig(
            channels=64,
            num_blocks=6,
            kernel_size=3,
            stem_stride=8,
            pool_every=2,
            dropout=0.1,
            head_hidden=128,
            norm="batchnorm",
        )
    elif preset == "small_1m":
        cfg = SmallCNNConfig(
            channels=128,
            num_blocks=8,
            kernel_size=3,
            stem_stride=8,
            pool_every=2,
            dropout=0.1,
            head_hidden=256,
            norm="batchnorm",
        )
    else:
        raise ValueError(f"Unknown model preset {preset!r}; expected tiny_100k or small_1m.")
    for key, value in overrides.items():
        if key == "preset" or value is None:
            continue
        if key == "dilation_cycle" and isinstance(value, list):
            value = tuple(int(v) for v in value)
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown model config field {key!r}.")
        setattr(cfg, key, value)
    return cfg


def build_model(model_config: dict[str, object]) -> SmallDNARegressor:
    """Build a SmallDNARegressor from a config dictionary."""
    config = dict(model_config)
    preset = str(config.pop("preset", "tiny_100k"))
    cfg = preset_config(preset, **config)
    return SmallDNARegressor(cfg)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size_mb(model: nn.Module) -> float:
    """Estimate parameter storage size in MiB."""
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    bytes_total += sum(b.numel() * b.element_size() for b in model.buffers())
    return bytes_total / (1024**2)


def model_summary(model: nn.Module) -> str:
    """Return a compact text model summary."""
    lines = [
        model.__class__.__name__,
        str(model),
        f"trainable_parameters: {count_parameters(model)}",
        f"model_size_mb: {estimate_model_size_mb(model):.3f}",
    ]
    return "\n".join(lines)
