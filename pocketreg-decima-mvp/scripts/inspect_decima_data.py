#!/usr/bin/env python
"""Inspect Decima AnnData metadata and layers."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketreg.data.anndata_utils import make_json_safe
from pocketreg.training.utils import setup_logging

LOGGER = logging.getLogger("inspect_decima_data")
OBS_VALUE_COLS = [
    "cell_type",
    "tissue",
    "organ",
    "disease",
    "region",
    "subregion",
    "celltype_coarse",
    "dataset",
]
OBS_NUMERIC_COLS = ["n_cells", "total_counts", "n_genes", "train_pearson", "val_pearson", "test_pearson"]
VAR_SUMMARY_COLS = [
    "gene_type",
    "chrom",
    "dataset",
    "fold",
    "frac_N",
    "frac_nan",
    "gene_length",
    "pearson",
]


def describe_numeric(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    out = {}
    for col in cols:
        if col not in df:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        out[col] = {k: make_json_safe(v) for k, v in series.describe().to_dict().items()}
    return out


def value_counts(df: pd.DataFrame, col: str, out_dir: Path) -> list[dict[str, Any]]:
    counts = df[col].astype(str).value_counts(dropna=False).head(50).reset_index()
    counts.columns = [col, "count"]
    out_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_dir / f"{col}.csv", index=False)
    return counts.to_dict(orient="records")


def layer_info(adata: ad.AnnData) -> tuple[list[dict[str, Any]], list[str]]:
    infos = []
    warnings = []
    for key in adata.layers.keys():
        layer = adata.layers[key]
        shape = getattr(layer, "shape", None)
        dtype = str(getattr(layer, "dtype", "unknown"))
        is_sparse = bool(sparse.issparse(layer))
        info = {"name": key, "shape": list(shape) if shape else None, "dtype": dtype, "sparse": is_sparse}
        if shape != adata.shape:
            warnings.append(f"Layer {key!r} shape {shape} does not match AnnData shape {adata.shape}.")
        infos.append(info)
    return infos, warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    setup_logging()
    if not args.adata.exists():
        raise FileNotFoundError(f"AnnData file not found: {args.adata}")

    args.out.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.adata)
    layer_infos, warnings = layer_info(adata)
    missing_obs = [col for col in OBS_VALUE_COLS + OBS_NUMERIC_COLS if col not in adata.obs]
    missing_var = [col for col in VAR_SUMMARY_COLS if col not in adata.var]
    warnings.extend([f"Missing obs column: {col}" for col in missing_obs])
    warnings.extend([f"Missing var column: {col}" for col in missing_var])

    adata.obs.head().to_csv(args.out / "obs_head.csv")
    adata.var.head().to_csv(args.out / "var_head.csv")

    obs_counts = {}
    for col in OBS_VALUE_COLS:
        if col in adata.obs:
            obs_counts[col] = value_counts(adata.obs, col, args.out / "obs_value_counts")
    var_counts = {}
    for col in ["gene_type", "chrom", "dataset", "fold"]:
        if col in adata.var:
            var_counts[col] = value_counts(adata.var, col, args.out / "var_value_counts")

    report = {
        "adata_path": str(args.adata),
        "shape": list(adata.shape),
        "obs_columns": list(map(str, adata.obs.columns)),
        "var_columns": list(map(str, adata.var.columns)),
        "layers": layer_infos,
        "obs_value_counts_top50": obs_counts,
        "obs_numeric_summary": describe_numeric(adata.obs, OBS_NUMERIC_COLS),
        "var_numeric_summary": describe_numeric(adata.var, VAR_SUMMARY_COLS),
        "warnings": warnings,
    }
    with (args.out / "inspection.json").open("w") as f:
        json.dump(report, f, indent=2, default=make_json_safe)
    with (args.out / "inspection.txt").open("w") as f:
        f.write(json.dumps(report, indent=2, default=make_json_safe))
        f.write("\n")

    LOGGER.info("AnnData shape: %s", adata.shape)
    LOGGER.info("Layers: %s", [info["name"] for info in layer_infos])
    for warning in warnings:
        LOGGER.warning(warning)
    LOGGER.info("Saved inspection to %s", args.out)


if __name__ == "__main__":
    main()
