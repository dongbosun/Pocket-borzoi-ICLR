#!/usr/bin/env python
"""Audit current Pocket-Decima repo state and existing results."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = {
    "metadata_h5ad": ROOT / "data/raw/decima/metadata.h5ad",
    "hg38_fasta": ROOT.parent / "external/reference/hg38/hg38.fa",
    "manifest_5k_v2": ROOT / "data/processed/decima_v2_astro_5k/manifest.parquet",
    "run_v1": ROOT / "outputs/runs/decima_astro_100k_64kb_5k_gpu",
    "run_v2": ROOT / "outputs/runs/decima_v2_astro_100k_64kb_5k",
    "v2_metrics": ROOT / "outputs/runs/decima_v2_astro_100k_64kb_5k/metrics.json",
    "v2_signal_summary": ROOT / "outputs/runs/decima_v2_astro_100k_64kb_5k/v2_signal_summary.json",
    "target_only_checkpoint": ROOT / "outputs/runs/decima_v2_astro_100k_64kb_5k/checkpoints/best_target_only.pt",
}


def run_cmd(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:  # pragma: no cover - defensive audit path
        return {"cmd": cmd, "error": repr(exc)}


def package_versions() -> dict[str, str | None]:
    packages = ["torch", "numpy", "pandas", "scipy", "scikit-learn", "anndata", "h5py", "pyfaidx", "pysam"]
    versions: dict[str, str | None] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    summary: dict[str, Any] = {"path": str(run_dir), "exists": run_dir.exists(), "metrics": metrics}
    if metrics and "test" in metrics:
        test = metrics["test"]
        summary["test"] = {
            "pearson": test.get("pearson"),
            "spearman": test.get("spearman"),
            "r2": test.get("r2"),
            "mae": test.get("mae"),
            "rmse": test.get("rmse"),
        }
    return summary


def summarize_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    frame = pd.read_parquet(path)
    out: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "split_counts": {str(k): int(v) for k, v in frame["split"].value_counts().to_dict().items()},
        "context_lens": sorted([int(x) for x in frame["context_len"].dropna().unique().tolist()]),
    }
    for col in ("target_obs_idx", "target_cell_type", "target_organ", "target_disease"):
        if col in frame:
            out[col] = sorted([str(x) for x in frame[col].dropna().unique().tolist()])
    return out


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    v1 = audit["runs"]["v1"].get("test", {})
    v2 = audit["runs"]["v2"].get("test", {})
    manifest = audit["manifest_5k_v2"]
    lines = [
        "# Job 0 Audit",
        "",
        "## Environment",
        f"- Python: `{audit['python']['executable']}`",
        f"- Python version: `{audit['python']['version']}`",
        f"- CUDA available in audit process: `{audit['cuda']['available']}`",
        f"- GPU name: `{audit['cuda'].get('gpu_name')}`",
        "",
        "## Git",
        f"- Commit: `{audit['git']['commit'].get('stdout')}`",
        "",
        "```",
        audit["git"]["status"].get("stdout", ""),
        "```",
        "",
        "## Located Files",
    ]
    for name, entry in audit["paths"].items():
        lines.append(f"- {name}: `{entry['path']}` exists={entry['exists']} size={entry.get('size_bytes')}")
    lines.extend(
        [
            "",
            "## Existing 5k v2 Manifest",
            f"- shape: `{manifest.get('shape')}`",
            f"- split counts: `{manifest.get('split_counts')}`",
            f"- context lens: `{manifest.get('context_lens')}`",
            "",
            "## Existing Metrics",
            f"- v1 test: Pearson `{v1.get('pearson')}`, R2 `{v1.get('r2')}`, RMSE `{v1.get('rmse')}`",
            f"- v2 test: Pearson `{v2.get('pearson')}`, R2 `{v2.get('r2')}`, RMSE `{v2.get('rmse')}`",
            "",
            "## Confirmation",
            f"- v2 target obs idx 88: `{audit['confirmations']['v2_target_obs_idx_88']}`",
            f"- v2 context 65536: `{audit['confirmations']['v2_context_65536']}`",
            f"- v2 input channels 5: `{audit['confirmations']['v2_input_channels_5']}`",
            f"- v2 test Pearson about 0.648: `{audit['confirmations']['v2_test_pearson_about_0648']}`",
            f"- v2 test R2 about 0.246: `{audit['confirmations']['v2_test_r2_about_0246']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/reports")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, path in DEFAULT_PATHS.items():
        paths[name] = {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None}

    cuda_available = torch.cuda.is_available()
    cuda = {"available": bool(cuda_available), "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None}
    v2_metrics = load_json(DEFAULT_PATHS["v2_metrics"]) or {}
    v2_signal_summary = load_json(DEFAULT_PATHS["v2_signal_summary"]) or {}
    v2_config = load_json(DEFAULT_PATHS["target_only_checkpoint"].with_suffix(".json")) or {}
    manifest = summarize_manifest(DEFAULT_PATHS["manifest_5k_v2"])
    target_idx_values = manifest.get("target_obs_idx", [])
    context_lens = manifest.get("context_lens", [])
    input_channels = v2_config.get("model_config", {}).get("input_channels")
    test = v2_metrics.get("test", {})
    audit = {
        "job": "job0_audit",
        "status": "completed",
        "root": str(ROOT),
        "python": {"executable": sys.executable, "version": sys.version, "platform": platform.platform()},
        "packages": package_versions(),
        "cuda": cuda,
        "git": {"commit": run_cmd(["git", "rev-parse", "HEAD"]), "status": run_cmd(["git", "status", "--short"])},
        "paths": paths,
        "manifest_5k_v2": manifest,
        "runs": {"v1": summarize_run(DEFAULT_PATHS["run_v1"]), "v2": summarize_run(DEFAULT_PATHS["run_v2"])},
        "v2_signal_summary": v2_signal_summary,
        "confirmations": {
            "v2_target_obs_idx_88": "88" in target_idx_values or 88 in target_idx_values,
            "v2_context_65536": 65536 in context_lens,
            "v2_input_channels_5": input_channels == 5,
            "v2_test_pearson_about_0648": abs(float(test.get("pearson", float("nan"))) - 0.648) < 0.01,
            "v2_test_r2_about_0246": abs(float(test.get("r2", float("nan"))) - 0.246) < 0.01,
        },
    }
    json_path = args.out_dir / "job0_audit.json"
    md_path = args.out_dir / "job0_audit.md"
    json_path.write_text(json.dumps(audit, indent=2) + "\n")
    write_markdown(audit, md_path)
    status_path = args.out_dir / "job0_status.json"
    status_path.write_text(json.dumps({"job": "job0_audit", "status": "completed", "json": str(json_path), "markdown": str(md_path)}, indent=2) + "\n")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
