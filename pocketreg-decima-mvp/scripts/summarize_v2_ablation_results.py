#!/usr/bin/env python
"""Summarize Pocket-Decima v2 5k ablation runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketreg.training.utils import save_json


ABLATION_ORDER = [
    "A0_final_only_dna4",
    "A1_final_only_dna5_mask",
    "A2_mask_replicates",
    "A3_mask_aux_raw_pca",
    "A4_mask_residual",
    "A5_full_v2_reproduce",
    "A6_full_v2_no_aux_pc1",
    "A7_full_v2_aux_residualized",
]

PURPOSE = {
    "A0_final_only_dna4": "DNA-only v2 TCN, final loss only",
    "A1_final_only_dna5_mask": "Add gene-body mask",
    "A2_mask_replicates": "Add replicate soft-label head",
    "A3_mask_aux_raw_pca": "Add raw biological-neighborhood aux PCA",
    "A4_mask_residual": "Add strict-metadata residual head",
    "A5_full_v2_reproduce": "Full v2: mask + reps + raw aux PCA + residual",
    "A6_full_v2_no_aux_pc1": "Full v2 but raw aux PCA PC1 dropped",
    "A7_full_v2_aux_residualized": "Full v2 but aux PCA residualized by strict metadata",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/ablations/v2_5k_obs88_64kb")
    parser.add_argument("--v1-metrics", type=Path, default=ROOT / "outputs/runs/decima_astro_100k_64kb_5k_gpu/metrics.json")
    parser.add_argument("--v2-metrics", type=Path, default=ROOT / "outputs/runs/decima_v2_astro_100k_64kb_5k/metrics.json")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_metric(path: Path) -> dict[str, Any]:
    metrics = load_json(path)
    test = metrics.get("test", {})
    val = metrics.get("val", {})
    best = metrics.get("best_epoch", {})
    return {
        "test_pearson": test.get("pearson"),
        "test_spearman": test.get("spearman"),
        "test_r2": test.get("r2"),
        "test_mae": test.get("mae"),
        "test_rmse": test.get("rmse"),
        "val_pearson": val.get("pearson"),
        "val_r2": val.get("r2"),
        "val_rmse": val.get("rmse"),
        "best_epoch": best.get("epoch"),
        "best_score": best.get("score"),
    }


def status_for_run(run_dir: Path) -> str:
    if (run_dir / "metrics.json").exists() and (run_dir / "checkpoints/best_target_only.pt").exists():
        return "completed"
    if (run_dir / "metrics.json").exists():
        return "completed_missing_target_export"
    if (run_dir / "checkpoints/last.pt").exists():
        return "running_or_incomplete"
    if run_dir.exists():
        return "started_no_metrics"
    return "pending"


def row_for_run(root: Path, name: str) -> dict[str, Any]:
    run_dir = root / "runs" / name
    row: dict[str, Any] = {
        "name": name,
        "purpose": PURPOSE[name],
        "run_dir": str(run_dir),
        "status": status_for_run(run_dir),
    }
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        row.update(read_metric(metrics_path))
        row["metrics_path"] = str(metrics_path)
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        row["config_path"] = str(config_path)
    target_path = run_dir / "checkpoints/best_target_only.pt"
    if target_path.exists():
        row["target_only_checkpoint"] = str(target_path)
    return row


def reference_row(name: str, path: Path, purpose: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    row = {"name": name, "purpose": purpose, "run_dir": str(path.parent), "status": "completed"}
    row.update(read_metric(path))
    row["metrics_path"] = str(path)
    return row


def delta(a: dict[str, Any] | None, b: dict[str, Any] | None, key: str) -> float | None:
    if not a or not b:
        return None
    av = a.get(key)
    bv = b.get(key)
    if av is None or bv is None:
        return None
    return float(av) - float(bv)


def make_interpretation(rows_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "architecture_tcn_vs_v1_delta_test_pearson": delta(rows_by_name.get("A0_final_only_dna4"), rows_by_name.get("v1_4ch_cnn"), "test_pearson"),
        "architecture_tcn_vs_v1_delta_test_r2": delta(rows_by_name.get("A0_final_only_dna4"), rows_by_name.get("v1_4ch_cnn"), "test_r2"),
        "gene_mask_delta_test_pearson": delta(rows_by_name.get("A1_final_only_dna5_mask"), rows_by_name.get("A0_final_only_dna4"), "test_pearson"),
        "gene_mask_delta_test_r2": delta(rows_by_name.get("A1_final_only_dna5_mask"), rows_by_name.get("A0_final_only_dna4"), "test_r2"),
        "replicate_delta_test_pearson": delta(rows_by_name.get("A2_mask_replicates"), rows_by_name.get("A1_final_only_dna5_mask"), "test_pearson"),
        "raw_aux_pca_delta_test_pearson": delta(rows_by_name.get("A3_mask_aux_raw_pca"), rows_by_name.get("A1_final_only_dna5_mask"), "test_pearson"),
        "residual_branch_delta_test_pearson": delta(rows_by_name.get("A4_mask_residual"), rows_by_name.get("A1_final_only_dna5_mask"), "test_pearson"),
        "full_v2_vs_mask_only_delta_test_pearson": delta(rows_by_name.get("A5_full_v2_reproduce"), rows_by_name.get("A1_final_only_dna5_mask"), "test_pearson"),
        "drop_pc1_delta_test_pearson": delta(rows_by_name.get("A6_full_v2_no_aux_pc1"), rows_by_name.get("A5_full_v2_reproduce"), "test_pearson"),
        "residualized_aux_delta_test_pearson": delta(rows_by_name.get("A7_full_v2_aux_residualized"), rows_by_name.get("A5_full_v2_reproduce"), "test_pearson"),
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def save_bar(df: pd.DataFrame, metric: str, path: Path) -> None:
    complete = df.loc[df[metric].notna()].copy()
    if complete.empty:
        return
    complete = complete.sort_values(metric, ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    positions = range(len(complete))
    ax.bar(positions, complete[metric])
    ax.set_ylabel(metric)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(complete["name"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown(
    root: Path,
    rows: list[dict[str, Any]],
    interpretation: dict[str, Any],
    completed_all: bool,
) -> None:
    columns = ["name", "status", "test_pearson", "test_spearman", "test_r2", "test_rmse", "best_epoch"]
    ranked = sorted(rows, key=lambda r: (-1e9 if r.get("test_pearson") is None else -float(r["test_pearson"]), r["name"]))
    lines = [
        "# Pocket-Decima v2 5k Ablation Summary",
        "",
        f"Status: {'completed' if completed_all else 'partial'}",
        "",
        "## Ranked Results",
        markdown_table(ranked, columns),
        "",
        "## Component Effects",
        markdown_table(
            [{"effect": k, "delta": v} for k, v in interpretation.items()],
            ["effect", "delta"],
        ),
        "",
        "## Readout",
        "- `A0` vs old v1 estimates the architecture/trunk change under DNA-only input.",
        "- `A1 - A0` estimates gene-mask contribution.",
        "- `A2 - A1`, `A3 - A1`, and `A4 - A1` isolate replicate, aux PCA, and residual heads.",
        "- `A6 - A5` tests whether dropping raw aux PC1 hurts or helps.",
        "- `A7 - A5` tests whether strict-metadata residualized aux PCA helps or hurts.",
        "",
    ]
    (root / "ablation_summary.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    v1 = reference_row("v1_4ch_cnn", args.v1_metrics, "Previous v1 4-channel CNN baseline")
    v2 = reference_row("previous_full_v2", args.v2_metrics, "Previous full v2 reference run")
    if v1:
        rows.append(v1)
    if v2:
        rows.append(v2)
    for name in ABLATION_ORDER:
        rows.append(row_for_run(args.root, name))
    rows_by_name = {row["name"]: row for row in rows}
    completed_all = all(rows_by_name[name]["status"].startswith("completed") for name in ABLATION_ORDER)
    if not completed_all and not args.allow_partial:
        pending = [name for name in ABLATION_ORDER if not rows_by_name[name]["status"].startswith("completed")]
        raise RuntimeError(f"Ablations are not complete yet: {pending}. Pass --allow-partial to summarize current state.")
    df = pd.DataFrame(rows)
    df.to_csv(args.root / "ablation_summary.csv", index=False)
    interpretation = make_interpretation(rows_by_name)
    summary = {
        "job": "job2_v2_5k_ablation",
        "status": "completed" if completed_all else "partial",
        "root": str(args.root.resolve()),
        "completed_all": completed_all,
        "rows": rows,
        "interpretation": interpretation,
        "outputs": {
            "csv": str((args.root / "ablation_summary.csv").resolve()),
            "json": str((args.root / "ablation_summary.json").resolve()),
            "markdown": str((args.root / "ablation_summary.md").resolve()),
        },
    }
    save_json(summary, args.root / "ablation_summary.json")
    write_markdown(args.root, rows, interpretation, completed_all)
    save_bar(df, "test_pearson", args.root / "plots/ablation_bar_pearson.png")
    save_bar(df, "test_r2", args.root / "plots/ablation_bar_r2.png")
    save_json(summary, ROOT / "outputs/reports/job2_ablation_status.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
