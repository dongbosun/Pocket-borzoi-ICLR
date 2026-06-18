#!/usr/bin/env python
"""Run official Mini-Borzoi K562 selected-track teacher inference for one shard."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

import _path  # noqa: F401

from pocketreg.borzoi.assets import load_assets_config  # noqa: E402
from pocketreg.borzoi.output_mapping import BorzoiOutputMapper  # noqa: E402
from pocketreg.borzoi.teacher import BorzoiK562Teacher  # noqa: E402
from pocketreg.cluster.safety import assert_compute_context, print_cluster_context  # noqa: E402
from pocketreg.data.fasta import FastaReader  # noqa: E402
from pocketreg.data.manifest import atomic_write_table, read_table  # noqa: E402
from pocketreg.data.sequence import one_hot_encode  # noqa: E402


LOGGER = logging.getLogger("run_borzoi_teacher_ref")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional run config for provenance.")
    parser.add_argument("--assets-config", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--aggregation", default="gene_body_log1p_mean")
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-failure-rate", type=float, default=0.05)
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _row_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    if hasattr(value, "item"):
        return value.item()
    return value


def _failure_row(row: dict[str, Any], shard_id: str, error: Exception) -> dict[str, Any]:
    return {
        "example_id": row.get("example_id"),
        "gene_id": row.get("gene_id"),
        "q_teacher": np.nan,
        "q_teacher_raw_mean": np.nan,
        "q_teacher_raw_sum": np.nan,
        "q_teacher_log1p_mean": np.nan,
        "aggregation_mode": row.get("aggregation_mode"),
        "n_bins_used": 0,
        "teacher_model": "mini_borzoi_k562",
        "teacher_weights": None,
        "target_index": row.get("target_index"),
        "shard_id": shard_id,
        "status": "failed",
        "error_message": str(error),
    }


def _aggregate_row(
    row: dict[str, Any],
    output: np.ndarray,
    teacher: BorzoiK562Teacher,
    aggregation: str,
) -> dict[str, Any]:
    mapper = BorzoiOutputMapper(
        input_seq_start=int(_row_value(row, "seq_start")),
        input_len=int(_row_value(row, "input_len", teacher.input_len)),
        output_num_bins=teacher.output_num_bins,
        bin_size=teacher.bin_size,
        target_index=teacher.mapper_target_index,
        output_core_start=int(_row_value(row, "output_core_start", None))
        if _row_value(row, "output_core_start", None) is not None
        else None,
    )
    if aggregation.startswith("gene_body"):
        result = mapper.aggregate_gene_body(
            output,
            int(_row_value(row, "gene_start")),
            int(_row_value(row, "gene_end")),
            mode=aggregation,
        )
    elif aggregation.startswith("tss_window"):
        result = mapper.aggregate_tss_window(
            output,
            int(_row_value(row, "tss")),
            flank=1024,
            mode=aggregation,
        )
    else:
        raise ValueError(f"Unsupported aggregation mode for teacher ref: {aggregation}")
    if result is None:
        raise ValueError("No Borzoi output bins overlap target gene interval")
    return {
        "q_teacher": result.q,
        "q_teacher_raw_mean": result.raw_mean,
        "q_teacher_raw_sum": result.raw_sum,
        "q_teacher_log1p_mean": result.log1p_mean,
        "n_bins_used": result.n_bins_used,
    }


def _flush_batch(
    batch_rows: list[dict[str, Any]],
    batch_onehots: list[np.ndarray],
    teacher: BorzoiK562Teacher,
    aggregation: str,
    shard_id: str,
    fail_fast: bool,
    results: list[dict[str, Any]],
) -> None:
    if not batch_rows:
        return
    try:
        preds = teacher.predict(np.stack(batch_onehots, axis=0), batch_size=len(batch_onehots))
    except Exception as exc:
        if fail_fast:
            raise
        for row in batch_rows:
            results.append(_failure_row(row, shard_id, exc))
        return
    for row, pred in zip(batch_rows, preds):
        try:
            values = _aggregate_row(row, pred, teacher=teacher, aggregation=aggregation)
            results.append(
                {
                    "example_id": row.get("example_id"),
                    "gene_id": row.get("gene_id"),
                    **values,
                    "aggregation_mode": aggregation,
                    "teacher_model": "mini_borzoi_k562",
                    "teacher_weights": str(teacher.weights_path),
                    "target_index": teacher.selected_target_index,
                    "shard_id": shard_id,
                    "status": "success",
                    "error_message": "",
                }
            )
        except Exception as exc:
            if fail_fast:
                raise
            results.append(_failure_row(row, shard_id, exc))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    assert_compute_context(
        task_name="run_borzoi_teacher_ref",
        allow_local=args.allow_local,
        toy=args.toy,
    )
    print_cluster_context()
    if args.out.exists() and not args.overwrite:
        LOGGER.info("Output exists and --overwrite not set; skipping %s", args.out)
        return
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    started = time.time()
    shard_id = args.shard.stem.replace("shard_", "")
    rows = read_table(args.shard)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    if not rows:
        raise ValueError(f"Shard has no rows: {args.shard}")

    assets = load_assets_config(args.assets_config)
    teacher = BorzoiK562Teacher(assets, target_index=args.target_index, fold=args.fold)
    fasta = FastaReader(args.fasta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    teacher.write_metadata_json(args.out.parent / "teacher_metadata.json")

    results: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    batch_onehots: list[np.ndarray] = []
    for row in rows:
        try:
            seq_start = int(_row_value(row, "seq_start"))
            seq_end = int(_row_value(row, "seq_end"))
            seq = fasta.fetch(str(row["chrom"]), seq_start, seq_end, pad=True)
            if len(seq) != teacher.input_len:
                raise ValueError(
                    f"Sequence length {len(seq)} does not match teacher input {teacher.input_len}"
                )
            batch_rows.append(row)
            batch_onehots.append(one_hot_encode(seq, channels_first=False))
            if len(batch_rows) >= args.batch_size:
                _flush_batch(
                    batch_rows,
                    batch_onehots,
                    teacher,
                    args.aggregation,
                    shard_id,
                    args.fail_fast,
                    results,
                )
                batch_rows = []
                batch_onehots = []
        except Exception as exc:
            if args.fail_fast:
                raise
            results.append(_failure_row(row, shard_id, exc))

    _flush_batch(
        batch_rows,
        batch_onehots,
        teacher,
        args.aggregation,
        shard_id,
        args.fail_fast,
        results,
    )

    n_total = len(results)
    n_success = sum(row.get("status") == "success" for row in results)
    n_failed = n_total - n_success
    failure_rate = n_failed / max(1, n_total)
    atomic_write_table([{k: _json_safe(v) for k, v in row.items()} for row in results], args.out)
    runtime = time.time() - started
    summary = {
        "task": "run_borzoi_teacher_ref",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "shard": str(args.shard),
        "out": str(args.out),
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
        "failure_rate": failure_rate,
        "runtime_seconds": runtime,
        "examples_per_second": n_total / runtime if runtime > 0 else None,
    }
    log_path = args.out.with_suffix(".summary.json")
    log_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failure_rate > args.max_failure_rate:
        raise RuntimeError(
            f"Failure rate {failure_rate:.3f} exceeds --max-failure-rate {args.max_failure_rate}"
        )


if __name__ == "__main__":
    main()
