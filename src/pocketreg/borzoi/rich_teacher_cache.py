"""Compact rich Mini-Borzoi reference teacher cache helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import numpy as np
import yaml

from pocketreg.borzoi.output_mapping import BorzoiOutputMapper


def load_selected_targets(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        data = yaml.safe_load(handle) or {}
    primary = data.get("primary_targets", []) or []
    aux = data.get("aux_targets", []) or []
    primary_indices = [int(item["index"]) for item in primary if item.get("index") is not None]
    aux_indices = [int(item["index"]) for item in aux if item.get("index") is not None]
    if not primary_indices:
        raise ValueError(f"No primary target indices in {path}")
    return {
        "path": str(path),
        "primary_targets": primary,
        "aux_targets": aux,
        "primary_indices": primary_indices,
        "aux_indices": aux_indices,
    }


def downsample_profile(values: np.ndarray, out_bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected one-dimensional profile, got {arr.shape}")
    if out_bins <= 0:
        raise ValueError("out_bins must be positive")
    if arr.shape[0] == out_bins:
        return arr.astype(np.float16)
    edges = np.linspace(0, arr.shape[0], out_bins + 1)
    pooled = np.empty(out_bins, dtype=np.float32)
    for i in range(out_bins):
        start = int(np.floor(edges[i]))
        end = int(np.floor(edges[i + 1]))
        if end <= start:
            end = min(arr.shape[0], start + 1)
        pooled[i] = float(np.mean(arr[start:end]))
    return pooled.astype(np.float16)


def aggregate_track_for_row(
    row: dict[str, Any],
    output: np.ndarray,
    *,
    input_len: int,
    output_num_bins: int,
    bin_size: int,
    target_index: int,
    aggregation: str = "gene_body_log1p_mean",
) -> Any:
    mapper = BorzoiOutputMapper(
        input_seq_start=int(row["seq_start"]),
        input_len=int(row.get("input_len", input_len)),
        output_num_bins=output_num_bins,
        bin_size=bin_size,
        target_index=int(target_index),
        output_core_start=int(row["output_core_start"]) if row.get("output_core_start") is not None else None,
    )
    if aggregation.startswith("gene_body"):
        return mapper.aggregate_gene_body(output, int(row["gene_start"]), int(row["gene_end"]), mode=aggregation)
    if aggregation.startswith("tss_window"):
        return mapper.aggregate_tss_window(output, int(row["tss"]), flank=1024, mode=aggregation)
    raise ValueError(f"Unsupported aggregation mode: {aggregation}")


def summarize_rich_labels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if row.get("status") == "success"]
    summary: dict[str, Any] = {
        "rows": len(rows),
        "success_rows": len(success),
        "failed_rows": len(rows) - len(success),
    }
    for key in ("q_old", "primary_0_q_mean", "primary_0_q_fold0", "primary_0_q_fold1"):
        vals = np.asarray([row.get(key, np.nan) for row in success], dtype=float)
        finite = np.isfinite(vals)
        if vals.size:
            summary[key] = {
                "finite_fraction": float(finite.mean()),
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
            }
    if success and "primary_0_q_fold0" in success[0] and "primary_0_q_fold1" in success[0]:
        x = np.asarray([row.get("primary_0_q_fold0", np.nan) for row in success], dtype=float)
        y = np.asarray([row.get("primary_0_q_fold1", np.nan) for row in success], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        summary["primary_0_fold_correlation"] = (
            float(np.corrcoef(x[mask], y[mask])[0, 1]) if int(mask.sum()) >= 2 else None
        )
    return summary


def write_summary(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
