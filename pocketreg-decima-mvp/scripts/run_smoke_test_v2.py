#!/usr/bin/env python
"""Synthetic smoke test for Pocket-Decima targeted distillation v2."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_smoke_test import create_toy_data

from pocketreg.data.v2 import build_v2_manifest_from_anndata, save_v2_manifest_outputs
from pocketreg.training.train_loop_v2 import train_v2_from_config
from pocketreg.training.utils import load_yaml, setup_logging

LOGGER = logging.getLogger("run_smoke_test_v2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "smoke_v2")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    setup_logging()

    adata_path, fasta_path = create_toy_data(args.out_dir, args.context_len, args.seed)
    toy = ad.read_h5ad(adata_path)
    manifest, target_metadata, summary = build_v2_manifest_from_anndata(
        toy,
        primary_idx=0,
        target_indices=[0, 1],
        context_len=args.context_len,
        fasta_path=fasta_path,
        split_mode="chromosome",
        max_genes=120,
        skip_fasta_check=False,
        aux_pca_components=4,
        aux_max_obs=12,
        residual=True,
    )
    manifest_path = args.out_dir / "manifest.parquet"
    save_v2_manifest_outputs(manifest, manifest_path, target_metadata, summary)

    config = load_yaml(ROOT / "configs" / "decima_v2_toy.yaml")
    config["manifest_path"] = str(manifest_path)
    config["fasta_path"] = str(fasta_path)
    config["output_dir"] = str(ROOT / "outputs" / "runs" / "smoke_v2")
    run_dir = train_v2_from_config(config)
    required = [
        run_dir / "checkpoints" / "best.pt",
        run_dir / "metrics.json",
        run_dir / "predictions_test.parquet",
        run_dir / "label_spec.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"v2 smoke test missing expected outputs: {missing}")
    LOGGER.info("v2 smoke test completed: %s", run_dir)


if __name__ == "__main__":
    main()
