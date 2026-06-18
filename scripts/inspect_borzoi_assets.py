#!/usr/bin/env python
"""Inspect Borzoi/K562 asset metadata without running teacher inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _path  # noqa: F401

from pocketreg.borzoi.assets import inspect_assets, load_assets_config, load_params
from pocketreg.borzoi.targets import find_k562_rnaseq_candidates, parse_targets
from pocketreg.cluster.safety import assert_compute_context, print_cluster_context


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-config", required=True)
    parser.add_argument("--out", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/borzoi_assets_inspection")
    parser.add_argument("--test-model-load", action="store_true")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    print_cluster_context()
    if args.test_model_load:
        assert_compute_context(
            "inspect_borzoi_assets_model_load",
            allow_local=args.allow_local,
            toy=args.toy,
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_assets_config(args.assets_config)
    report = inspect_assets(config)

    targets = []
    candidates = []
    if config.k562_targets and config.k562_targets.exists():
        targets = parse_targets(config.k562_targets)
        candidates = find_k562_rnaseq_candidates(targets)
        write_csv(candidates, out / "targets_candidates.csv")

    if config.k562_params and config.k562_params.exists():
        params = load_params(config.k562_params)
        (out / "params_pretty.json").write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")

    report["selected_target"] = candidates[0] if candidates else None
    if args.test_model_load:
        try:
            import tensorflow as tf  # type: ignore

            report["model_load_test"] = {
                "status": "tensorflow_available_loader_deferred_to_phase2",
                "tensorflow_version": tf.__version__,
                "note": "Phase 1 validates metadata only; BorzoiK562Teacher is implemented in Phase 2.",
            }
        except Exception as exc:
            report["model_load_test"] = {
                "status": "failed_before_model_load",
                "error": str(exc),
                "note": "Install TensorFlow 2.15.x and official Borzoi/Baskerville repos.",
            }
    else:
        report["model_load_test"] = {"status": "skipped"}

    (out / "assets_inspection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report.get("params", {}).get("detected", {}), indent=2))
    print(f"targets={len(targets)} candidates={len(candidates)} out={out}")


if __name__ == "__main__":
    main()
