#!/usr/bin/env python
"""Select primary and auxiliary Mini-Borzoi K562 RNA-seq targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import _path  # noqa: F401

from pocketreg.borzoi.assets import load_assets_config  # noqa: E402
from pocketreg.borzoi.task_selection import select_targets, write_selection_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/borzoi_distill_v2.yaml", type=Path)
    parser.add_argument("--assets-config", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--ref-labels", type=Path)
    parser.add_argument("--out-dir", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/borzoi_task_selection", type=Path)
    parser.add_argument("--out-config", default="configs/borzoi_k562_targets.local.yaml", type=Path)
    parser.add_argument("--out-manifest", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_selected_targets.json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle) or {}
    assets_config = args.assets_config or Path(config.get("assets_config", "configs/borzoi_assets.local.yaml"))
    assets = load_assets_config(assets_config)
    targets = args.targets or assets.k562_targets
    if targets is None:
        raise ValueError("No targets file provided and assets config lacks k562_targets")
    ref_labels = args.ref_labels or Path(config.get("ref_labels_path", "/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_ref_labels.parquet"))
    selection = select_targets(targets, config=config, ref_labels_path=ref_labels)
    selection["config"] = str(args.config)
    selection["assets_config"] = str(assets_config)
    write_selection_outputs(selection, args.out_dir, args.out_config, args.out_manifest)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
