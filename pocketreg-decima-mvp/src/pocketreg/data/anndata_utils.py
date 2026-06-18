"""Utilities for reading Decima AnnData metadata and teacher labels."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

LOGGER = logging.getLogger(__name__)


def make_json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalar values into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [make_json_safe(v) for v in value]
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def row_to_metadata(row: pd.Series, obs_pos: int) -> dict[str, Any]:
    """Serialize one obs row with a stable integer position."""
    metadata = {"target_obs_idx": int(obs_pos), "obs_index": make_json_safe(row.name)}
    for key, value in row.items():
        metadata[str(key)] = make_json_safe(value)
    return metadata


def _casefold_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.casefold()


def _apply_exact_filter(df: pd.DataFrame, col: str, value: str | None) -> pd.DataFrame:
    if value is None or col not in df:
        return df
    mask = _casefold_series(df[col]) == value.casefold()
    return df.loc[mask]


def _apply_contains_filter(df: pd.DataFrame, col: str, value: str | None) -> pd.DataFrame:
    if value is None or col not in df:
        return df
    mask = df[col].astype(str).str.contains(value, case=False, na=False, regex=False)
    return df.loc[mask]


def _sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in ("test_pearson", "val_pearson", "n_cells") if col in df]
    if not sort_cols:
        return df
    return df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")


def select_pseudobulk(
    adata: Any,
    target_index: int | None = None,
    query: str | None = None,
    prefer_healthy_brain: bool = True,
    *,
    organ: str | None = None,
    tissue: str | None = None,
    disease: str | None = None,
    cell_type_contains: str | None = None,
    region_contains: str | None = None,
    subregion_contains: str | None = None,
    celltype_coarse_contains: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Select a Decima pseudobulk obs row and return its integer index plus metadata."""
    obs = adata.obs.copy()
    obs["_obs_pos"] = np.arange(adata.n_obs)

    if target_index is not None:
        if target_index < 0 or target_index >= adata.n_obs:
            raise IndexError(f"target_index={target_index} is outside [0, {adata.n_obs}).")
        row = adata.obs.iloc[int(target_index)]
        metadata = row_to_metadata(row, int(target_index))
        LOGGER.info("Selected pseudobulk by target index %s: %s", target_index, metadata)
        return int(target_index), metadata

    candidates = obs
    if query:
        try:
            candidates = candidates.query(query, engine="python")
        except Exception as exc:
            raise ValueError(
                f"Could not evaluate target query {query!r}. Use simple filter flags "
                "such as --organ, --disease, and --cell-type-contains if pandas query "
                f"syntax is not supported. Original error: {exc}"
            ) from exc

    candidates = _apply_exact_filter(candidates, "organ", organ)
    candidates = _apply_exact_filter(candidates, "tissue", tissue)
    candidates = _apply_exact_filter(candidates, "disease", disease)
    candidates = _apply_contains_filter(candidates, "cell_type", cell_type_contains)
    candidates = _apply_contains_filter(candidates, "region", region_contains)
    candidates = _apply_contains_filter(candidates, "subregion", subregion_contains)
    candidates = _apply_contains_filter(candidates, "celltype_coarse", celltype_coarse_contains)

    if candidates.empty:
        raise ValueError(
            "No pseudobulk rows matched the requested filters. Inspect obs columns with "
            "scripts/inspect_decima_data.py and relax the selection flags."
        )

    if prefer_healthy_brain and not any(
        [query, organ, tissue, disease, cell_type_contains, region_contains, celltype_coarse_contains]
    ):
        if "disease" in candidates:
            healthy = _casefold_series(candidates["disease"]).isin({"healthy", "control", "normal"})
            if healthy.any():
                candidates = candidates.loc[healthy]
            else:
                LOGGER.warning("No healthy/control/normal disease rows found; using all diseases.")
        brain_mask = pd.Series(False, index=candidates.index)
        for col in ("organ", "tissue"):
            if col in candidates:
                brain_mask = brain_mask | candidates[col].astype(str).str.contains(
                    "brain", case=False, na=False
                )
        if brain_mask.any():
            candidates = candidates.loc[brain_mask]
        else:
            LOGGER.warning("No brain organ/tissue rows found; using all tissues.")

    candidates = _sort_candidates(candidates)
    selected = candidates.iloc[0]
    obs_pos = int(selected["_obs_pos"])
    row = adata.obs.iloc[obs_pos]
    metadata = row_to_metadata(row, obs_pos)
    LOGGER.info("Selected pseudobulk obs position %s: %s", obs_pos, metadata)
    return obs_pos, metadata


def _extract_layer_row(layer: Any, row_idx: int) -> np.ndarray:
    row = layer[row_idx, :]
    if sparse.issparse(row):
        row = row.toarray()
    return np.asarray(row).reshape(-1).astype(np.float32)


def get_teacher_labels(adata: Any, pseudobulk_idx: int, label_layer: str) -> np.ndarray:
    """Extract one pseudobulk's Decima teacher labels from an AnnData layer."""
    available = list(adata.layers.keys())
    if label_layer == "ensemble":
        rep_layers = [name for name in [f"v1_rep{i}" for i in range(4)] if name in adata.layers]
        if not rep_layers:
            raise KeyError(
                "Requested label_layer='ensemble', but no v1_rep0-v1_rep3 layers were found. "
                f"Available layers: {available}"
            )
        rows = [_extract_layer_row(adata.layers[name], pseudobulk_idx) for name in rep_layers]
        labels = np.nanmean(np.stack(rows, axis=0), axis=0).astype(np.float32)
    else:
        if label_layer not in adata.layers:
            raise KeyError(
                f"Requested label layer {label_layer!r} was not found. Available layers: {available}"
            )
        labels = _extract_layer_row(adata.layers[label_layer], pseudobulk_idx)

    if labels.shape[0] != adata.n_vars:
        raise ValueError(
            f"Layer {label_layer!r} row length {labels.shape[0]} does not match n_vars={adata.n_vars}."
        )
    return labels.astype(np.float32, copy=False)


def summarize_labels(labels: np.ndarray) -> dict[str, float | int | None]:
    """Summarize finite teacher labels."""
    finite = np.asarray(labels, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0}
    quantiles = np.quantile(finite, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
    }
