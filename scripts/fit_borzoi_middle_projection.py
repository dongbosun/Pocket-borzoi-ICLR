#!/usr/bin/env python
"""Fit target-relevant projection labels from pooled Borzoi middle features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import _path  # noqa: F401

from pocketreg.borzoi.middle_projection import fit_projection, write_json  # noqa: E402
from pocketreg.training.metrics import pearsonr, r2_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--middle-index", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_middle_pooled.index.parquet", type=Path)
    parser.add_argument("--middle-features", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_middle_pooled.features.npz", type=Path)
    parser.add_argument("--rich-labels", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_ref_rich.labels.parquet", type=Path)
    parser.add_argument("--profile-pca", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_profile_pca_labels.parquet", type=Path)
    parser.add_argument("--aux-pca", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_aux_pca_labels.parquet", type=Path)
    parser.add_argument("--feature-key", default="features_mean")
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--method", default="pls", choices=["pls", "pca"])
    parser.add_argument("--out-labels", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_middle_projection_labels.parquet", type=Path)
    parser.add_argument("--out-model", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_middle_projection_model.joblib", type=Path)
    parser.add_argument("--report-dir", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/borzoi_middle_projection", type=Path)
    return parser.parse_args()


def _probe_metrics(x: np.ndarray, y: np.ndarray, split: np.ndarray, target_name: str) -> dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    train = split == "train"
    out = {}
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train])
    model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model.fit(x_train, y[train])
    for split_name in ("val", "test"):
        mask = split == split_name
        pred = model.predict(scaler.transform(x[mask]))
        out[split_name] = {
            "target": target_name,
            "pearson": pearsonr(y[mask].tolist(), pred.tolist()),
            "r2": r2_score(y[mask].tolist(), pred.tolist()),
        }
    return out


def main() -> None:
    args = parse_args()
    index = pd.read_parquet(args.middle_index)
    features_npz = np.load(args.middle_features)
    if args.feature_key not in features_npz:
        raise KeyError(f"{args.feature_key} missing; keys={features_npz.files}")
    x = np.asarray(features_npz[args.feature_key], dtype=np.float32)
    if x.shape[0] != len(index):
        raise ValueError(f"Feature rows {x.shape[0]} != index rows {len(index)}")
    rich = pd.read_parquet(args.rich_labels)
    profile = pd.read_parquet(args.profile_pca)
    aux = pd.read_parquet(args.aux_pca)
    base = index[["example_id", "gene_id", "split"]].copy()
    rich_value_cols = [
        col
        for col in rich.columns
        if col == "example_id"
        or col in {"primary_0_q_mean", "primary_0_q_fold0", "primary_0_q_fold1"}
    ]
    labels = base.merge(rich[rich_value_cols], on="example_id", how="inner")
    labels = labels.merge(
        profile.drop(columns=[col for col in ("gene_id", "split") if col in profile.columns]),
        on="example_id",
        how="inner",
    )
    labels = labels.merge(
        aux.drop(columns=[col for col in ("gene_id", "split") if col in aux.columns]),
        on="example_id",
        how="inner",
    )
    target_cols = [
        col
        for col in labels.columns
        if col in {"primary_0_q_mean", "primary_0_q_fold0", "primary_0_q_fold1"}
        or col.startswith("profile_pca_")
        or col.startswith("aux_pca_")
    ]
    y = labels[target_cols].to_numpy(dtype=np.float32)
    split = labels["split"].astype(str).to_numpy()
    train_mask = split == "train"
    x_scaler, proj_model, projected, method_used = fit_projection(
        x,
        y,
        train_mask,
        n_components=args.components,
        method=args.method,
    )
    out = labels[["example_id", "gene_id", "split"]].copy()
    for i in range(projected.shape[1]):
        out[f"middle_proj_{i}"] = projected[:, i]
    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_labels, index=False)
    joblib.dump(
        {
            "x_scaler": x_scaler,
            "projection": proj_model,
            "feature_key": args.feature_key,
            "target_cols": target_cols,
        },
        args.out_model,
    )
    probe = _probe_metrics(x, labels["primary_0_q_mean"].to_numpy(dtype=np.float32), split, "primary_0_q_mean")
    summary = {
        "middle_index": str(args.middle_index),
        "middle_features": str(args.middle_features),
        "feature_key": args.feature_key,
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "target_cols": target_cols,
        "method": method_used,
        "components": int(projected.shape[1]),
        "projected_shape": list(projected.shape),
        "probe_primary_q": probe,
        "outputs": {"labels": str(args.out_labels), "model": str(args.out_model)},
    }
    write_json(args.report_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
