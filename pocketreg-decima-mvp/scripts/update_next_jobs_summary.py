#!/usr/bin/env python
"""Write the integrated Pocket-Decima next-jobs status summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketreg.training.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-md", type=Path, default=ROOT / "outputs/reports/pocket_decima_next_jobs_summary.md")
    parser.add_argument("--out-json", type=Path, default=ROOT / "outputs/reports/pocket_decima_next_jobs_summary.json")
    parser.add_argument("--note", default="")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def metric_block(path: Path) -> dict[str, Any] | None:
    data = load_json(path)
    if not data:
        return None
    test = data.get("test", {})
    return {
        "metrics_path": str(path),
        "test_pearson": test.get("pearson"),
        "test_spearman": test.get("spearman"),
        "test_r2": test.get("r2"),
        "test_mae": test.get("mae"),
        "test_rmse": test.get("rmse"),
        "best_epoch": data.get("best_epoch", {}).get("epoch"),
    }


def path_status(path: Path) -> str:
    return "completed" if path.exists() else "pending"


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
    return "\n".join(out)


def collect() -> dict[str, Any]:
    reports = ROOT / "outputs/reports"
    ab_root = ROOT / "outputs/ablations/v2_5k_obs88_64kb"
    full_manifest = ROOT / "data/processed/decima_v2_astro_full/manifest.parquet"
    full_cache_summary = ROOT / "outputs/sequence_cache/obs88_full/context_65536/cache_summary.json"
    ctx128_manifest = ROOT / "data/processed/decima_v2_astro_full_128kb/manifest.parquet"
    ctx128_cache_summary = ROOT / "outputs/sequence_cache/obs88_full/context_131072/cache_summary.json"
    jobs: dict[str, Any] = {
        "0_repository_audit": {
            "status": path_status(reports / "job0_audit.json"),
            "outputs": [str(reports / "job0_audit.json"), str(reports / "job0_audit.md")],
        },
        "1_baselines_v21": {
            "status": path_status(reports / "baselines_v21_obs88/metrics.json"),
            "outputs": [str(reports / "baselines_v21_obs88/metrics.json"), str(reports / "baselines_v21_obs88/baselines_v21_obs88.md")],
        },
        "2_5k_ablation_queue": load_json(ab_root / "ablation_summary.json")
        or {"status": "pending", "outputs": [str(ab_root / "ablation_summary.json")]},
        "3_full_manifest_sequence_cache": {
            "status": "completed" if full_manifest.exists() and full_cache_summary.exists() else "queued_or_running",
            "manifest": str(full_manifest),
            "cache_summary": str(full_cache_summary),
        },
        "4_fullgenes_64kb_3seeds": {
            "status": "completed"
            if all((ROOT / f"outputs/runs/fullgenes_obs88_64kb_v2_seed{i}/metrics.json").exists() for i in (1, 2, 3))
            else "queued_or_running",
            "runs": {
                f"seed{i}": metric_block(ROOT / f"outputs/runs/fullgenes_obs88_64kb_v2_seed{i}/metrics.json")
                for i in (1, 2, 3)
            },
        },
        "5_context_scaling_128kb_seed1": {
            "status": "completed"
            if (ROOT / "outputs/runs/context_scaling_obs88_128kb_seed1/metrics.json").exists()
            else "queued_or_running",
            "manifest": str(ctx128_manifest),
            "cache_summary": str(ctx128_cache_summary),
            "run": metric_block(ROOT / "outputs/runs/context_scaling_obs88_128kb_seed1/metrics.json"),
        },
        "6_aux_pca_variants": {"status": "pending", "reason": "Not started; wait for Job2/5 signals."},
        "7_decima_middle_extraction": {"status": "pending", "reason": "Not started; official Decima availability not checked in this queue yet."},
        "8_true_middle_distillation": {"status": "blocked", "reason": "Requires Job7 middle targets."},
        "9_multitarget_quick": {"status": "pending"},
        "10_model_size_scaling": {"status": "pending"},
        "11_final_export_benchmark": {"status": "pending"},
        "12_integrated_report": {"status": "interim"},
    }
    baselines = load_json(reports / "baselines_v21_obs88/metrics.json")
    if baselines:
        jobs["1_baselines_v21"]["key_test_metrics"] = {
            name: values.get("test", {})
            for name, values in baselines.items()
            if name in {"mean", "strict_metadata", "expression_oracle", "decima_eval_oracle", "combined_oracle"}
        }
    return {
        "project": "Pocket-Decima Targeted Distillation v2 next experiment queue",
        "status": "interim",
        "jobs": jobs,
        "current_interpretation": {
            "strict_metadata_is_below_sequence_v2": True,
            "oracle_metadata_is_leakage_like": True,
            "a1_gene_mask_currently_explains_most_5k_gain": True,
        },
    }


def write_markdown(summary: dict[str, Any], out_md: Path, note: str) -> None:
    jobs = summary["jobs"]
    rows = [{"job": k, "status": v.get("status")} for k, v in jobs.items()]
    ab_rows = []
    ab = jobs.get("2_5k_ablation_queue", {})
    for row in ab.get("rows", []):
        if row.get("name", "").startswith("A") or row.get("name") in {"v1_4ch_cnn", "previous_full_v2"}:
            ab_rows.append(row)
    lines = [
        "# Pocket-Decima Next Jobs Summary",
        "",
        "Status: interim",
        "",
    ]
    if note:
        lines.extend(["## Note", note, ""])
    lines.extend(
        [
            "## Job Status",
            table(rows, ["job", "status"]),
            "",
            "## Current 5k Ablation Readout",
        ]
    )
    if ab_rows:
        lines.append(table(ab_rows, ["name", "status", "test_pearson", "test_r2", "test_rmse", "best_epoch"]))
    else:
        lines.append("No ablation rows available yet.")
    lines.extend(
        [
            "",
            "## Current Interpretation",
            "- Strict metadata is now separated from oracle/leakage-like metadata.",
            "- A0 shows the v2 TCN architecture alone improves over the older v1 CNN.",
            "- A1 shows the gene-body mask is currently the dominant observed 5k gain.",
            "- A2-A7, full-gene, and 128kb context jobs are queued/running; do not over-interpret until they finish.",
            "",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    summary = collect()
    summary["note"] = args.note
    save_json(summary, args.out_json)
    write_markdown(summary, args.out_md, args.note)
    print(json.dumps({"markdown": str(args.out_md), "json": str(args.out_json)}, indent=2))


if __name__ == "__main__":
    main()
