"""Train-only PCA helpers for Borzoi v2 profile and auxiliary labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import numpy as np
import pandas as pd


def fit_standardized_pca(
    x: np.ndarray,
    train_mask: np.ndarray,
    n_components: int,
    *,
    random_state: int = 42,
) -> tuple[Any, Any, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {x.shape}")
    train = x[train_mask]
    if train.shape[0] < 2:
        raise ValueError("Need at least two train rows for PCA")
    n = min(int(n_components), train.shape[0], train.shape[1])
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train)
    pca = PCA(n_components=n, random_state=random_state)
    pca.fit(train_z)
    transformed = pca.transform(scaler.transform(x)).astype(np.float32)
    return scaler, pca, transformed


def save_component_labels(
    base: pd.DataFrame,
    values: np.ndarray,
    prefix: str,
    out_path: str | Path,
) -> None:
    frame = base[["example_id", "gene_id", "split"]].copy()
    for i in range(values.shape[1]):
        frame[f"{prefix}_{i}"] = values[:, i]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)


def aux_matrix_from_labels(labels: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = sorted(
        [col for col in labels.columns if col.startswith("aux_") and col.endswith("_q_mean")],
        key=lambda name: int(name.split("_")[1]),
    )
    if not cols:
        raise ValueError("No aux_<k>_q_mean columns found in rich labels")
    return labels[cols].to_numpy(dtype=np.float32), cols


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
