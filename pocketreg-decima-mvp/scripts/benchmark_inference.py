#!/usr/bin/env python
"""Benchmark model-only and end-to-end Decima student inference."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketreg.eval.benchmark import (
    attach_common_fields,
    benchmark_end_to_end,
    benchmark_model_only,
    memory_report,
)
from pocketreg.training.train_loop import load_checkpoint_model
from pocketreg.training.utils import resolve_device, save_json, setup_logging

LOGGER = logging.getLogger("benchmark_inference")


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    setup_logging()

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    manifest = pd.read_parquet(args.manifest)
    context_len = int(checkpoint.get("context_len") or manifest["context_len"].iloc[0])
    y_mean = float(checkpoint.get("y_mean", 0.0))
    y_std = float(checkpoint.get("y_std", 1.0))
    results = []
    for batch_size in args.batch_sizes:
        before_rss = rss_mb()
        model_only = benchmark_model_only(
            model,
            device=device,
            context_len=context_len,
            batch_size=batch_size,
            num_warmup=args.num_warmup,
            num_steps=args.num_steps,
        )
        after_rss = rss_mb()
        row = attach_common_fields(
            model_only,
            mode="model_only",
            device=device,
            batch_size=batch_size,
            context_len=context_len,
            num_warmup=args.num_warmup,
            num_steps=args.num_steps,
            model=model,
        )
        row.update(memory_report(device, before_rss, after_rss))
        results.append(row)

        before_rss = rss_mb()
        end_to_end = benchmark_end_to_end(
            model,
            args.manifest,
            args.fasta,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            batch_size=batch_size,
            num_warmup=args.num_warmup,
            num_steps=args.num_steps,
        )
        after_rss = rss_mb()
        row = attach_common_fields(
            end_to_end,
            mode="end_to_end",
            device=device,
            batch_size=batch_size,
            context_len=context_len,
            num_warmup=args.num_warmup,
            num_steps=args.num_steps,
            model=model,
        )
        row.update(memory_report(device, before_rss, after_rss))
        results.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_json({"results": results}, args.out)
    header = "mode\tdevice\tbatch\tmean_ms_batch\tmean_ms_gene\tgenes_per_second"
    print(header)
    for row in results:
        print(
            f"{row['mode']}\t{row['device']}\t{row['batch_size']}\t"
            f"{row['mean_ms_per_batch']:.3f}\t{row['mean_ms_per_gene']:.3f}\t"
            f"{row['genes_per_second']:.2f}"
        )
    LOGGER.info("Saved benchmark JSON to %s", args.out)


if __name__ == "__main__":
    main()
