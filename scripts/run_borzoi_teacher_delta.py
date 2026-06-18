#!/usr/bin/env python
"""Run official Mini-Borzoi K562 ref-alt SNV delta teacher inference."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import _path  # noqa: F401
import numpy as np

from pocketreg.borzoi.assets import load_assets_config  # noqa: E402
from pocketreg.borzoi.output_mapping import BorzoiOutputMapper  # noqa: E402
from pocketreg.borzoi.teacher import BorzoiK562Teacher  # noqa: E402
from pocketreg.cluster.safety import assert_compute_context, print_cluster_context  # noqa: E402
from pocketreg.data.fasta import FastaReader  # noqa: E402
from pocketreg.data.manifest import atomic_write_table, read_table  # noqa: E402
from pocketreg.data.sequence import apply_snv, one_hot_encode  # noqa: E402

LOGGER = logging.getLogger("run_borzoi_teacher_delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--assets-config", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--ref-cache", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--aggregation", default="gene_body_log1p_mean")
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max-variants", type=int)
    parser.add_argument("--max-failure-rate", type=float, default=0.05)
    parser.add_argument("--reuse-ref-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-ref-check", action="store_true")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _row_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    if hasattr(value, "item"):
        return value.item()
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _failure_row(variant: dict[str, Any], shard_id: str, error: Exception, status: str = "failed") -> dict[str, Any]:
    return {
        "variant_example_id": variant.get("variant_example_id"),
        "example_id": variant.get("example_id"),
        "gene_id": variant.get("gene_id"),
        "chrom": variant.get("chrom"),
        "pos_1based": variant.get("pos_1based"),
        "pos_0based": variant.get("pos_0based"),
        "ref": variant.get("ref"),
        "alt": variant.get("alt"),
        "q_ref_teacher": np.nan,
        "q_alt_teacher": np.nan,
        "delta_teacher": np.nan,
        "abs_delta_teacher": np.nan,
        "sign_teacher": np.nan,
        "q_ref_source": "",
        "target_index": variant.get("target_index"),
        "aggregation_mode": "",
        "n_bins_used": 0,
        "shard_id": shard_id,
        "status": status,
        "error_message": str(error),
    }


def _mapper_for(row: dict[str, Any], teacher: BorzoiK562Teacher) -> BorzoiOutputMapper:
    return BorzoiOutputMapper(
        input_seq_start=int(_row_value(row, "seq_start")),
        input_len=int(_row_value(row, "input_len", teacher.input_len)),
        output_num_bins=teacher.output_num_bins,
        bin_size=teacher.bin_size,
        target_index=teacher.mapper_target_index,
        output_core_start=int(_row_value(row, "output_core_start", None))
        if _row_value(row, "output_core_start", None) is not None
        else None,
    )


def _aggregate(row: dict[str, Any], output: np.ndarray, teacher: BorzoiK562Teacher, aggregation: str):
    mapper = _mapper_for(row, teacher)
    if aggregation.startswith("gene_body"):
        result = mapper.aggregate_gene_body(
            output,
            int(_row_value(row, "gene_start")),
            int(_row_value(row, "gene_end")),
            mode=aggregation,
        )
    else:
        raise ValueError(f"Unsupported delta aggregation mode: {aggregation}")
    if result is None:
        raise ValueError("No Borzoi output bins overlap target gene interval")
    return result


def _success_row(
    variant: dict[str, Any],
    ref_row: dict[str, Any],
    q_ref: float,
    q_alt: float,
    n_bins_used: int,
    teacher: BorzoiK562Teacher,
    aggregation: str,
    shard_id: str,
    q_ref_source: str,
) -> dict[str, Any]:
    delta = q_alt - q_ref
    return {
        "variant_example_id": variant.get("variant_example_id"),
        "example_id": variant.get("example_id"),
        "gene_id": variant.get("gene_id"),
        "chrom": variant.get("chrom"),
        "pos_1based": int(_row_value(variant, "pos_1based")),
        "pos_0based": int(_row_value(variant, "pos_0based")),
        "ref": variant.get("ref"),
        "alt": variant.get("alt"),
        "distance_to_tss": variant.get("distance_to_tss"),
        "split": variant.get("split"),
        "q_ref_teacher": q_ref,
        "q_alt_teacher": q_alt,
        "delta_teacher": delta,
        "abs_delta_teacher": abs(delta),
        "sign_teacher": 1 if delta > 0 else (-1 if delta < 0 else 0),
        "q_ref_source": q_ref_source,
        "target_index": teacher.selected_target_index,
        "aggregation_mode": aggregation,
        "n_bins_used": n_bins_used,
        "variant_in_input": int(ref_row["seq_start"]) <= int(variant["pos_0based"]) < int(ref_row["seq_end"]),
        "variant_in_output_core": int(ref_row["output_core_start"]) <= int(variant["pos_0based"]) < int(ref_row["output_core_end"]),
        "shard_id": shard_id,
        "status": "success",
        "error_message": "",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    assert_compute_context("run_borzoi_teacher_delta", allow_local=args.allow_local, toy=args.toy)
    print_cluster_context()
    if args.out.exists() and not args.overwrite:
        LOGGER.info("Output exists and --overwrite not set; skipping %s", args.out)
        return
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    started = time.time()
    shard_id = args.shard.stem.replace("shard_", "")
    variants = read_table(args.shard)
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    if not variants:
        raise ValueError(f"No variants in shard: {args.shard}")

    ref_cache_rows = read_table(args.ref_cache)
    ref_by_id = {row["example_id"]: row for row in ref_cache_rows if row.get("status") == "success"}
    assets = load_assets_config(args.assets_config)
    teacher = BorzoiK562Teacher(assets, target_index=args.target_index, fold=args.fold)
    fasta = FastaReader(args.fasta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    teacher.write_metadata_json(args.out.parent / "teacher_metadata.json")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for variant in variants:
        groups[str(variant["example_id"])].append(variant)

    results: list[dict[str, Any]] = []
    for example_id, group in groups.items():
        ref_row = ref_by_id.get(example_id)
        if ref_row is None:
            error = KeyError(f"Missing successful ref-cache row for example_id={example_id}")
            for variant in group:
                results.append(_failure_row(variant, shard_id, error, status="failed_missing_ref"))
            continue
        try:
            seq_start = int(ref_row["seq_start"])
            seq_end = int(ref_row["seq_end"])
            ref_seq = fasta.fetch(str(ref_row["chrom"]), seq_start, seq_end, pad=True)
            if len(ref_seq) != teacher.input_len:
                raise ValueError(f"Reference sequence length {len(ref_seq)} != {teacher.input_len}")
            if args.reuse_ref_cache:
                q_ref = float(ref_row["q_teacher"])
                q_ref_source = "ref_cache"
            else:
                ref_pred = teacher.predict(
                    one_hot_encode(ref_seq, channels_first=False)[None, ...],
                    batch_size=1,
                )[0]
                q_ref_result = _aggregate(ref_row, ref_pred, teacher, args.aggregation)
                q_ref = float(q_ref_result.q)
                q_ref_source = "teacher_recomputed"
        except Exception as exc:
            if args.fail_fast:
                raise
            for variant in group:
                results.append(_failure_row(variant, shard_id, exc))
            continue

        batch_variants: list[dict[str, Any]] = []
        batch_onehots: list[np.ndarray] = []
        for variant in group:
            try:
                applied = apply_snv(
                    ref_seq,
                    genomic_pos_0based=int(_row_value(variant, "pos_0based")),
                    seq_start_0based=seq_start,
                    ref=str(variant["ref"]),
                    alt=str(variant["alt"]),
                    skip_ref_check=args.skip_ref_check,
                )
                batch_variants.append(variant)
                batch_onehots.append(one_hot_encode(applied.alt_sequence, channels_first=False))
            except Exception as exc:
                if args.fail_fast:
                    raise
                results.append(_failure_row(variant, shard_id, exc, status="failed_ref_mismatch"))

            if len(batch_variants) >= args.batch_size:
                try:
                    preds = teacher.predict(np.stack(batch_onehots, axis=0), batch_size=len(batch_onehots))
                    for alt_variant, pred in zip(batch_variants, preds):
                        alt_result = _aggregate(ref_row, pred, teacher, args.aggregation)
                        results.append(
                            _success_row(
                                alt_variant,
                                ref_row,
                                q_ref=q_ref,
                                q_alt=float(alt_result.q),
                                n_bins_used=alt_result.n_bins_used,
                                teacher=teacher,
                                aggregation=args.aggregation,
                                shard_id=shard_id,
                                q_ref_source=q_ref_source,
                            )
                        )
                except Exception as exc:
                    if args.fail_fast:
                        raise
                    for alt_variant in batch_variants:
                        results.append(_failure_row(alt_variant, shard_id, exc))
                batch_variants = []
                batch_onehots = []

        if batch_variants:
            try:
                preds = teacher.predict(np.stack(batch_onehots, axis=0), batch_size=len(batch_onehots))
                for alt_variant, pred in zip(batch_variants, preds):
                    alt_result = _aggregate(ref_row, pred, teacher, args.aggregation)
                    results.append(
                        _success_row(
                            alt_variant,
                            ref_row,
                            q_ref=q_ref,
                            q_alt=float(alt_result.q),
                            n_bins_used=alt_result.n_bins_used,
                            teacher=teacher,
                            aggregation=args.aggregation,
                            shard_id=shard_id,
                            q_ref_source=q_ref_source,
                        )
                    )
            except Exception as exc:
                if args.fail_fast:
                    raise
                for alt_variant in batch_variants:
                    results.append(_failure_row(alt_variant, shard_id, exc))

    n_total = len(results)
    n_success = sum(row.get("status") == "success" for row in results)
    n_failed = n_total - n_success
    failure_rate = n_failed / max(1, n_total)
    atomic_write_table([{k: _json_safe(v) for k, v in row.items()} for row in results], args.out)
    runtime = time.time() - started
    deltas = [abs(float(row["delta_teacher"])) for row in results if row.get("status") == "success"]
    summary = {
        "task": "run_borzoi_teacher_delta",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "shard": str(args.shard),
        "out": str(args.out),
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
        "failure_rate": failure_rate,
        "runtime_seconds": runtime,
        "variants_per_second": n_total / runtime if runtime > 0 else None,
        "abs_delta_lt_1e-6": sum(v < 1e-6 for v in deltas) / max(1, len(deltas)),
        "abs_delta_lt_1e-4": sum(v < 1e-4 for v in deltas) / max(1, len(deltas)),
        "abs_delta_lt_1e-3": sum(v < 1e-3 for v in deltas) / max(1, len(deltas)),
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failure_rate > args.max_failure_rate:
        raise RuntimeError(
            f"Failure rate {failure_rate:.3f} exceeds --max-failure-rate {args.max_failure_rate}"
        )


if __name__ == "__main__":
    main()
