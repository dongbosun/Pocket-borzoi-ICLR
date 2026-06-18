#!/usr/bin/env python
"""Evaluate a trained Decima student checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketreg.training.train_loop import evaluate_checkpoint
from pocketreg.training.utils import setup_logging

LOGGER = logging.getLogger("evaluate_decima_student")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    setup_logging()
    out_dir = evaluate_checkpoint(
        args.checkpoint,
        args.manifest,
        args.fasta,
        args.out,
        device_name=args.device,
        split=args.split,
        batch_size=args.batch_size,
    )
    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    LOGGER.info("Saved evaluation to %s", out_dir)
    for split, vals in metrics.items():
        LOGGER.info(
            "%s: n=%s pearson=%s spearman=%s r2=%s mae=%s rmse=%s",
            split,
            vals.get("n"),
            vals.get("pearson"),
            vals.get("spearman"),
            vals.get("r2"),
            vals.get("mae"),
            vals.get("rmse"),
        )


if __name__ == "__main__":
    main()
