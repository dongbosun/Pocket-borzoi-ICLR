"""Target-relevant projection for pooled Borzoi middle features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import numpy as np


def build_target_matrix(label_frames: list[Any], id_col: str = "example_id") -> tuple[Any, list[str]]:
    import pandas as pd

    merged = None
    for frame in label_frames:
        keep = [id_col] + [col for col in frame.columns if col != id_col and np.issubdtype(frame[col].dtype, np.number)]
        sub = frame[keep].copy()
        merged = sub if merged is None else merged.merge(sub, on=id_col, how="inner")
    if merged is None:
        raise ValueError("No label frames supplied")
    cols = [col for col in merged.columns if col != id_col]
    return merged, cols


def fit_projection(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    n_components: int = 32,
    method: str = "pls",
    random_state: int = 42,
) -> tuple[Any, Any, np.ndarray, str]:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x[train_mask])
    x_all = x_scaler.transform(x)
    n = min(int(n_components), x_train.shape[0] - 1, x_train.shape[1], y.shape[1])
    if n <= 0:
        raise ValueError("Projection needs at least one component")
    if method == "pls":
        try:
            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(y[train_mask])
            model = PLSRegression(n_components=n, scale=False)
            model.fit(x_train, y_train)
            projected = model.transform(x_all).astype(np.float32)
            return x_scaler, {"method": "pls", "model": model, "y_scaler": y_scaler}, projected, "pls"
        except Exception:
            method = "pca"
    pca = PCA(n_components=n, random_state=random_state)
    pca.fit(x_train)
    projected = pca.transform(x_all).astype(np.float32)
    return x_scaler, {"method": "pca", "model": pca}, projected, "pca"


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
