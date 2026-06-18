"""Compact pooled teacher middle/head-input feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import numpy as np


def select_layer_name(report_path: str | Path, candidate: str = "auto") -> str:
    report = json.loads(Path(report_path).read_text())
    if candidate and candidate != "auto":
        return candidate
    candidates = report.get("candidates") or {}
    for key in ("head_input", "penultimate_spatial", "middle_spatial"):
        layer = candidates.get(key)
        if layer and layer.get("name"):
            return str(layer["name"])
    raise ValueError(f"Could not auto-select layer from {report_path}")


def pooled_spatial_features(tensor: np.ndarray, center_bins: int = 256) -> np.ndarray:
    """Pool [bins, channels] tensor into mean/max/center-mean features."""

    arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected [bins, channels] tensor, got {arr.shape}")
    mean = np.mean(arr, axis=0)
    maxv = np.max(arr, axis=0)
    bins = arr.shape[0]
    width = min(int(center_bins), bins)
    start = max(0, (bins - width) // 2)
    center = np.mean(arr[start : start + width], axis=0)
    return np.concatenate([mean, maxv, center], axis=0).astype(np.float16)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
