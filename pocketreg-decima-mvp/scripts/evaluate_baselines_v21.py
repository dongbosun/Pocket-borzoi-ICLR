#!/usr/bin/env python
"""Evaluate strict and oracle metadata baselines for Pocket-Decima v2.1."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketreg.data.fasta import FastaReader
from pocketreg.data.sequence import gc_content
from pocketreg.eval.plots import plot_parity, plot_residuals
from pocketreg.training.metrics import regression_metrics


LOGGER = logging.getLogger("evaluate_baselines_v21")
STRICT_FEATURES = ["gene_length", "frac_N", "GC", "CpG"]
EXPRESSION_ORACLE_FEATURES = ["mean_counts", "frac_nan", "n_tracks"]
DECIMA_EVAL_ORACLE_FEATURES = ["pearson", "size_factor_pearson"]
SINGLE_FEATURES = [
    "gene_length",
    "mean_counts",
    "n_tracks",
    "pearson",
    "size_factor_pearson",
    "GC",
    "CpG",
]
ALPHAS = np.logspace(-4, 4, 17)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/reports/baselines_v21_obs88")
    parser.add_argument("--target-col", default="y_final_t0")
    parser.add_argument("--max-cache", type=int, default=0, help="Reserved for future sequence-feature cache use.")
    return parser.parse_args()


def cpg_observed_expected(seq: str) -> float:
    seq = seq.upper()
    length = len(seq)
    if length == 0:
        return float("nan")
    c = seq.count("C")
    g = seq.count("G")
    cpg = seq.count("CG")
    denom = c * g
    if denom <= 0:
        return 0.0
    return float(cpg * length / denom)


def add_sequence_features(manifest: pd.DataFrame, fasta_path: Path) -> pd.DataFrame:
    if {"GC", "CpG"}.issubset(manifest.columns):
        return manifest
    reader = FastaReader(fasta_path)
    gc_values: list[float] = []
    cpg_values: list[float] = []
    for _, row in manifest.iterrows():
        chrom = row.get("fasta_chrom", row["chrom"])
        seq = reader.fetch(str(chrom), int(row["seq_start"]), int(row["seq_end"]), pad=True)
        gc_values.append(gc_content(seq))
        cpg_values.append(cpg_observed_expected(seq))
    frame = manifest.copy()
    frame["GC"] = np.asarray(gc_values, dtype=np.float32)
    frame["CpG"] = np.asarray(cpg_values, dtype=np.float32)
    return frame


def finite_feature_columns(frame: pd.DataFrame, candidates: list[str]) -> tuple[list[str], dict[str, str]]:
    cols = []
    missing = {}
    for col in candidates:
        if col not in frame.columns:
            missing[col] = "missing"
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().sum() == 0:
            missing[col] = "all_nan"
            continue
        cols.append(col)
    return cols, missing


def calibration(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2 or np.nanstd(y_pred[mask]) == 0:
        return {"calibration_slope": float("nan"), "calibration_intercept": float("nan")}
    reg = LinearRegression().fit(y_pred[mask].reshape(-1, 1), y_true[mask])
    return {"calibration_slope": float(reg.coef_[0]), "calibration_intercept": float(reg.intercept_)}


def per_chrom_metrics(preds: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for chrom, frame in preds.groupby("chrom"):
        if len(frame) < 3:
            continue
        metrics = regression_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
        out[str(chrom)] = {k: float(v) for k, v in metrics.items() if k in {"n", "pearson", "r2", "rmse"}}
    return out


def fit_predict_ridge(
    manifest: pd.DataFrame,
    features: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not features:
        raise ValueError("No features provided to Ridge baseline.")
    train = manifest["split"].eq("train").to_numpy()
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    x = manifest[features].apply(pd.to_numeric, errors="coerce")
    y = manifest[target_col].to_numpy(dtype=float)
    model.fit(x.loc[train], y[train])
    pred = model.predict(x)
    alpha = float(model.named_steps["ridge"].alpha_)
    preds = manifest[["gene_id", "chrom", "split"]].copy()
    preds["y_true"] = y
    preds["y_pred"] = pred.astype(float)
    return preds, {"alpha": alpha, "features": features}


def mean_predictions(manifest: pd.DataFrame, target_col: str) -> pd.DataFrame:
    train_mean = float(manifest.loc[manifest["split"].eq("train"), target_col].mean())
    preds = manifest[["gene_id", "chrom", "split"]].copy()
    preds["y_true"] = manifest[target_col].to_numpy(dtype=float)
    preds["y_pred"] = train_mean
    return preds


def summarize_predictions(preds: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    train_mean = float(preds.loc[preds["split"].eq("train"), "y_true"].mean())
    for split, frame in preds.groupby("split"):
        metrics = regression_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy(), train_mean=train_mean)
        metrics.update(calibration(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()))
        metrics["per_chrom"] = per_chrom_metrics(frame)
        metrics["y_true_quantiles"] = {
            str(q): float(np.quantile(frame["y_true"].to_numpy(), q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)
        }
        metrics["y_pred_quantiles"] = {
            str(q): float(np.quantile(frame["y_pred"].to_numpy(), q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)
        }
        out[str(split)] = metrics
    return out


def feature_correlations(manifest: pd.DataFrame, features: list[str], target_col: str) -> pd.DataFrame:
    train = manifest[manifest["split"].eq("train")]
    rows = []
    y = train[target_col].to_numpy(dtype=float)
    for feature in features:
        if feature not in train:
            continue
        x = pd.to_numeric(train[feature], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3 or np.nanstd(x[mask]) == 0:
            pearson = float("nan")
            spearman = float("nan")
        else:
            pearson = float(stats.pearsonr(x[mask], y[mask]).statistic)
            spearman = float(stats.spearmanr(x[mask], y[mask]).statistic)
        rows.append({"feature": feature, "n_train": int(mask.sum()), "pearson_train": pearson, "spearman_train": spearman})
    return pd.DataFrame(rows)


def save_plots(name: str, preds: pd.DataFrame, metrics: dict[str, Any], out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        frame = preds[preds["split"].eq(split)]
        if frame.empty:
            continue
        plot_parity(
            frame["y_true"].to_numpy(),
            frame["y_pred"].to_numpy(),
            metrics.get(split, {}),
            plot_dir / f"parity_{name}_{split}.png",
            f"{name} {split}",
        )
        if split == "test":
            plot_residuals(
                frame["y_true"].to_numpy(),
                frame["y_pred"].to_numpy(),
                plot_dir / f"residual_{name}_{split}.png",
                f"{name} {split} residuals",
            )


def write_report(metrics: dict[str, Any], metrics_table: pd.DataFrame, feature_groups: dict[str, Any], out_path: Path) -> None:
    def md_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        cols = [c for c in ["baseline", "group", "split", "pearson", "spearman", "r2", "mae", "rmse", "features"] if c in frame.columns]
        view = frame[cols].copy()
        for col in ("pearson", "spearman", "r2", "mae", "rmse"):
            if col in view:
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(rows)

    lines = [
        "# Baselines v2.1 obs88",
        "",
        "This report separates fair-ish metadata baselines from leakage-like/oracle baselines.",
        "",
        "Strict baseline uses only `gene_length`, `frac_N`, `GC`, and `CpG` when available.",
        "The fields `mean_counts`, `frac_nan`, `n_tracks`, `pearson`, and `size_factor_pearson` are reported only as oracle/leakage-like baselines, not as fair sequence-only baselines.",
        "",
        "## Feature Groups",
        "```json",
        json.dumps(feature_groups, indent=2),
        "```",
        "",
        "## Test Metrics",
        md_table(metrics_table[metrics_table["split"].eq("test")].sort_values("pearson", ascending=False)),
        "",
        "## All Metrics",
        md_table(metrics_table),
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    setup_logging()
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.fasta.exists():
        raise FileNotFoundError(f"FASTA not found: {args.fasta}")
    manifest = pd.read_parquet(args.manifest)
    if args.target_col not in manifest:
        raise ValueError(f"Target column {args.target_col!r} not found in manifest.")
    LOGGER.info("Computing/loading sequence features GC and CpG for %s genes", len(manifest))
    manifest = add_sequence_features(manifest, args.fasta)
    manifest.to_parquet(args.out / "manifest_with_baseline_features.parquet", index=False)

    strict, strict_missing = finite_feature_columns(manifest, STRICT_FEATURES)
    expression_oracle, expression_missing = finite_feature_columns(manifest, EXPRESSION_ORACLE_FEATURES)
    decima_oracle, decima_missing = finite_feature_columns(manifest, DECIMA_EVAL_ORACLE_FEATURES)
    combined = strict + [c for c in expression_oracle + decima_oracle if c not in strict]
    groups = {
        "mean": {"type": "train_label_mean"},
        "strict_metadata": {"features": strict, "missing": strict_missing, "fair_sequence_baseline": True},
        "expression_oracle": {"features": expression_oracle, "missing": expression_missing, "fair_sequence_baseline": False},
        "decima_eval_oracle": {"features": decima_oracle, "missing": decima_missing, "fair_sequence_baseline": False},
        "combined_oracle": {"features": combined, "fair_sequence_baseline": False},
    }

    jobs: dict[str, dict[str, Any]] = {
        "mean": {"kind": "mean", "group": "mean", "features": []},
        "strict_metadata": {"kind": "ridge", "group": "strict_metadata", "features": strict},
        "expression_oracle": {"kind": "ridge", "group": "expression_oracle", "features": expression_oracle},
        "decima_eval_oracle": {"kind": "ridge", "group": "decima_eval_oracle", "features": decima_oracle},
        "combined_oracle": {"kind": "ridge", "group": "combined_oracle", "features": combined},
    }
    for feature in SINGLE_FEATURES:
        cols, _ = finite_feature_columns(manifest, [feature])
        if cols:
            jobs[f"single_{feature}"] = {"kind": "ridge", "group": "single_feature", "features": cols}

    all_metrics: dict[str, Any] = {}
    rows = []
    for name, spec in jobs.items():
        LOGGER.info("Running baseline %s", name)
        if spec["kind"] == "mean":
            preds = mean_predictions(manifest, args.target_col)
            fit_info = {"features": [], "alpha": None}
        else:
            preds, fit_info = fit_predict_ridge(manifest, spec["features"], args.target_col)
        metrics = summarize_predictions(preds)
        all_metrics[name] = {"group": spec["group"], "fit": fit_info, "metrics": metrics}
        preds.to_parquet(args.out / f"predictions_{name}.parquet", index=False)
        save_plots(name, preds, metrics, args.out)
        for split, split_metrics in metrics.items():
            rows.append(
                {
                    "baseline": name,
                    "group": spec["group"],
                    "split": split,
                    "features": ",".join(spec["features"]),
                    "n": split_metrics.get("n"),
                    "pearson": split_metrics.get("pearson"),
                    "spearman": split_metrics.get("spearman"),
                    "r2": split_metrics.get("r2"),
                    "mae": split_metrics.get("mae"),
                    "rmse": split_metrics.get("rmse"),
                    "calibration_slope": split_metrics.get("calibration_slope"),
                    "calibration_intercept": split_metrics.get("calibration_intercept"),
                    "alpha": fit_info.get("alpha"),
                }
            )

    metrics_table = pd.DataFrame(rows)
    all_features = sorted(set(strict + expression_oracle + decima_oracle + [f for f in SINGLE_FEATURES if f in manifest.columns]))
    corr = feature_correlations(manifest, all_features, args.target_col)
    metrics_table.to_csv(args.out / "metrics.csv", index=False)
    corr.to_csv(args.out / "feature_correlations.csv", index=False)
    (args.out / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    (args.out / "feature_groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    write_report(all_metrics, metrics_table, groups, args.out / "baselines_v21_obs88.md")
    status = {
        "job": "job1_baselines_v21",
        "status": "completed",
        "out_dir": str(args.out),
        "metrics_json": str(args.out / "metrics.json"),
        "metrics_csv": str(args.out / "metrics.csv"),
        "report": str(args.out / "baselines_v21_obs88.md"),
    }
    (args.out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    (ROOT / "outputs/reports/job1_baselines_v21_status.json").write_text(json.dumps(status, indent=2) + "\n")
    LOGGER.info("Wrote baselines v2.1 outputs to %s", args.out)


if __name__ == "__main__":
    main()
