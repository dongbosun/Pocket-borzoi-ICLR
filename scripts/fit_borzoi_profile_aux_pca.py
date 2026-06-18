#!/usr/bin/env python
"""Fit train-only PCA labels from rich Borzoi reference cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import _path  # noqa: F401

from pocketreg.borzoi.profile_pca import (  # noqa: E402
    aux_matrix_from_labels,
    fit_standardized_pca,
    save_component_labels,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich-labels", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_ref_rich.labels.parquet", type=Path)
    parser.add_argument("--profiles", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_ref_rich.profiles.npz", type=Path)
    parser.add_argument("--out-dir", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache", type=Path)
    parser.add_argument("--report-dir", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/borzoi_profile_aux_pca", type=Path)
    parser.add_argument("--profile-components", type=int, default=16)
    parser.add_argument("--aux-components", type=int, default=8)
    parser.add_argument("--profile-key", default="profiles_mean")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _plot_variance(values: np.ndarray, path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, len(values) + 1), np.cumsum(values), marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("components")
    plt.ylabel("cumulative explained variance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    labels = pd.read_parquet(args.rich_labels)
    data = np.load(args.profiles)
    if args.profile_key not in data:
        raise KeyError(f"{args.profile_key} not present in {args.profiles}; keys={data.files}")
    profiles = np.asarray(data[args.profile_key], dtype=np.float32)
    if profiles.shape[0] != len(labels):
        raise ValueError(f"Profile rows {profiles.shape[0]} != labels {len(labels)}")
    train_mask = labels["split"].astype(str).to_numpy() == "train"

    profile_scaler, profile_pca, profile_values = fit_standardized_pca(
        profiles,
        train_mask,
        args.profile_components,
        random_state=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    profile_labels = args.out_dir / "k562_profile_pca_labels.parquet"
    profile_model = args.out_dir / "k562_profile_pca_model.joblib"
    save_component_labels(labels, profile_values, "profile_pca", profile_labels)
    joblib.dump({"scaler": profile_scaler, "pca": profile_pca, "profile_key": args.profile_key}, profile_model)

    aux_x, aux_cols = aux_matrix_from_labels(labels)
    aux_scaler, aux_pca, aux_values = fit_standardized_pca(
        aux_x,
        train_mask,
        min(args.aux_components, aux_x.shape[1]),
        random_state=args.seed,
    )
    aux_labels = args.out_dir / "k562_aux_pca_labels.parquet"
    aux_model = args.out_dir / "k562_aux_pca_model.joblib"
    save_component_labels(labels, aux_values, "aux_pca", aux_labels)
    joblib.dump({"scaler": aux_scaler, "pca": aux_pca, "columns": aux_cols}, aux_model)

    summary = {
        "rich_labels": str(args.rich_labels),
        "profiles": str(args.profiles),
        "profile_key": args.profile_key,
        "num_rows": int(len(labels)),
        "split_counts": labels["split"].value_counts().to_dict(),
        "profile_shape": list(profiles.shape),
        "profile_components": int(profile_values.shape[1]),
        "profile_explained_variance_ratio": [float(v) for v in profile_pca.explained_variance_ratio_],
        "profile_explained_variance_sum": float(np.sum(profile_pca.explained_variance_ratio_)),
        "aux_columns": aux_cols,
        "aux_shape": list(aux_x.shape),
        "aux_components": int(aux_values.shape[1]),
        "aux_explained_variance_ratio": [float(v) for v in aux_pca.explained_variance_ratio_],
        "aux_explained_variance_sum": float(np.sum(aux_pca.explained_variance_ratio_)),
        "outputs": {
            "profile_labels": str(profile_labels),
            "profile_model": str(profile_model),
            "aux_labels": str(aux_labels),
            "aux_model": str(aux_model),
        },
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.report_dir / "summary.json", summary)
    _plot_variance(profile_pca.explained_variance_ratio_, args.report_dir / "profile_pca_variance.png", "profile PCA")
    _plot_variance(aux_pca.explained_variance_ratio_, args.report_dir / "aux_pca_variance.png", "aux PCA")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
