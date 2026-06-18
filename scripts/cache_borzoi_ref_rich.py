#!/usr/bin/env python
"""Cache compact rich reference Mini-Borzoi K562 teacher outputs."""

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
from pocketreg.borzoi.rich_teacher_cache import (  # noqa: E402
    aggregate_track_for_row,
    downsample_profile,
    load_selected_targets,
    summarize_rich_labels,
    write_summary,
)
from pocketreg.borzoi.teacher import BorzoiK562Teacher  # noqa: E402
from pocketreg.cluster.safety import assert_compute_context, print_cluster_context  # noqa: E402
from pocketreg.data.fasta import FastaReader  # noqa: E402
from pocketreg.data.manifest import atomic_write_table, read_table  # noqa: E402
from pocketreg.data.sequence import one_hot_encode  # noqa: E402

LOGGER = logging.getLogger("cache_borzoi_ref_rich")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-config", default="configs/borzoi_assets.local.yaml", type=Path)
    parser.add_argument("--distill-config", default="configs/borzoi_distill_v2.yaml", type=Path)
    parser.add_argument("--target-config", default="configs/borzoi_k562_targets.local.yaml", type=Path)
    parser.add_argument("--manifest", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_gene_manifest.parquet", type=Path)
    parser.add_argument("--shard", type=Path)
    parser.add_argument("--fasta", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa", type=Path)
    parser.add_argument("--out-prefix", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_ref_rich", type=Path)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1], choices=[0, 1])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n", type=int, help="Limit examples for smoke/debug.")
    parser.add_argument("--aggregation", default="gene_body_log1p_mean")
    parser.add_argument("--profile-downsample-bins", type=int, default=384)
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _row_base(row: dict[str, Any], profile_row: int) -> dict[str, Any]:
    return {
        "profile_row": profile_row,
        "example_id": row.get("example_id"),
        "gene_id": row.get("gene_id"),
        "gene_name": row.get("gene_name"),
        "chrom": row.get("chrom"),
        "gene_start": row.get("gene_start"),
        "gene_end": row.get("gene_end"),
        "strand": row.get("strand"),
        "split": row.get("split"),
        "seq_start": row.get("seq_start"),
        "seq_end": row.get("seq_end"),
    }


def _failure(base: dict[str, Any], error: Exception) -> dict[str, Any]:
    return {**base, "status": "failed", "error_message": str(error)}


def _flush_batch(
    *,
    teacher: BorzoiK562Teacher,
    fold: int,
    rows: list[dict[str, Any]],
    onehots: list[np.ndarray],
    labels_by_example: dict[str, dict[str, Any]],
    profile_accumulators: dict[str, list[np.ndarray]],
    selected: dict[str, Any],
    aggregation: str,
    profile_bins: int,
) -> None:
    if not rows:
        return
    preds = teacher.predict(np.stack(onehots, axis=0), batch_size=len(onehots))
    for row, pred in zip(rows, preds):
        example_id = str(row["example_id"])
        label = labels_by_example[example_id]
        for p_i, target_index in enumerate(selected["primary_indices"]):
            result = aggregate_track_for_row(
                row,
                pred,
                input_len=teacher.input_len,
                output_num_bins=teacher.output_num_bins,
                bin_size=teacher.bin_size,
                target_index=target_index,
                aggregation=aggregation,
            )
            if result is None:
                raise ValueError(f"No bins overlap gene for {example_id}")
            label[f"primary_{p_i}_target_index"] = target_index
            label[f"primary_{p_i}_q_fold{fold}"] = float(result.q)
            label[f"primary_{p_i}_raw_mean_fold{fold}"] = float(result.raw_mean)
            if p_i == 0:
                label[f"q_old_fold{fold}"] = float(result.q)
                profile_accumulators[f"fold{fold}"].append(
                    downsample_profile(pred[:, target_index], profile_bins)
                )
        for a_i, target_index in enumerate(selected["aux_indices"]):
            result = aggregate_track_for_row(
                row,
                pred,
                input_len=teacher.input_len,
                output_num_bins=teacher.output_num_bins,
                bin_size=teacher.bin_size,
                target_index=target_index,
                aggregation=aggregation,
            )
            if result is not None:
                label[f"aux_{a_i}_target_index"] = target_index
                label[f"aux_{a_i}_q_fold{fold}"] = float(result.q)


def _finalize_labels(rows: list[dict[str, Any]], folds: list[int], selected: dict[str, Any]) -> None:
    for row in rows:
        if row.get("status") == "failed":
            continue
        for p_i, _ in enumerate(selected["primary_indices"]):
            vals = [float(row[f"primary_{p_i}_q_fold{fold}"]) for fold in folds if f"primary_{p_i}_q_fold{fold}" in row]
            if vals:
                row[f"primary_{p_i}_q_mean"] = float(np.mean(vals))
                row[f"primary_{p_i}_q_std"] = float(np.std(vals))
        for a_i, _ in enumerate(selected["aux_indices"]):
            vals = [float(row[f"aux_{a_i}_q_fold{fold}"]) for fold in folds if f"aux_{a_i}_q_fold{fold}" in row]
            if vals:
                row[f"aux_{a_i}_q_mean"] = float(np.mean(vals))
                row[f"aux_{a_i}_q_std"] = float(np.std(vals))
        q_vals = [float(row[f"q_old_fold{fold}"]) for fold in folds if f"q_old_fold{fold}" in row]
        if q_vals:
            row["q_old"] = float(np.mean(q_vals))
            row["q_teacher"] = row["q_old"]
        row["status"] = "success"
        row["error_message"] = ""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, default=str))
        return
    assert_compute_context("cache_borzoi_ref_rich", allow_local=args.allow_local, toy=False)
    print_cluster_context()

    label_path = args.out_prefix.with_suffix(".labels.parquet")
    index_path = args.out_prefix.with_suffix(".index.parquet")
    profiles_path = args.out_prefix.with_suffix(".profiles.npz")
    summary_path = args.out_prefix.with_suffix(".summary.json")
    if label_path.exists() and profiles_path.exists() and not args.overwrite:
        LOGGER.info("Outputs exist and --overwrite not set; skipping %s", args.out_prefix)
        return

    started = time.time()
    assets = load_assets_config(args.assets_config)
    selected = load_selected_targets(args.target_config)
    rows = read_table(args.shard or args.manifest)
    if args.n is not None:
        rows = rows[: args.n]
    if not rows:
        raise ValueError("No rows to process")

    fasta = FastaReader(args.fasta)
    labels = [_row_base(row, i) for i, row in enumerate(rows)]
    labels_by_example = {str(row["example_id"]): label for row, label in zip(rows, labels)}
    profile_accumulators: dict[str, list[np.ndarray]] = {f"fold{fold}": [] for fold in args.folds}

    for fold in args.folds:
        LOGGER.info("Loading fold %s full-output teacher", fold)
        teacher = BorzoiK562Teacher(assets, fold=fold, slice_to_target=False)
        batch_rows: list[dict[str, Any]] = []
        batch_onehots: list[np.ndarray] = []
        for row in rows:
            try:
                seq = fasta.fetch(str(row["chrom"]), int(row["seq_start"]), int(row["seq_end"]), pad=True)
                if len(seq) != teacher.input_len:
                    raise ValueError(f"Sequence length {len(seq)} != teacher input {teacher.input_len}")
                batch_rows.append(row)
                batch_onehots.append(one_hot_encode(seq, channels_first=False))
                if len(batch_rows) >= args.batch_size:
                    _flush_batch(
                        teacher=teacher,
                        fold=fold,
                        rows=batch_rows,
                        onehots=batch_onehots,
                        labels_by_example=labels_by_example,
                        profile_accumulators=profile_accumulators,
                        selected=selected,
                        aggregation=args.aggregation,
                        profile_bins=args.profile_downsample_bins,
                    )
                    batch_rows = []
                    batch_onehots = []
            except Exception as exc:
                labels_by_example[str(row["example_id"])].update(_failure(labels_by_example[str(row["example_id"])], exc))
        if batch_rows:
            _flush_batch(
                teacher=teacher,
                fold=fold,
                rows=batch_rows,
                onehots=batch_onehots,
                labels_by_example=labels_by_example,
                profile_accumulators=profile_accumulators,
                selected=selected,
                aggregation=args.aggregation,
                profile_bins=args.profile_downsample_bins,
            )

    _finalize_labels(labels, args.folds, selected)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_table(labels, label_path)
    index_rows = [
        {
            "profile_row": row["profile_row"],
            "example_id": row["example_id"],
            "gene_id": row["gene_id"],
            "split": row.get("split"),
            "status": row.get("status"),
        }
        for row in labels
    ]
    atomic_write_table(index_rows, index_path)
    arrays: dict[str, np.ndarray] = {}
    for fold in args.folds:
        key = f"profiles_fold{fold}"
        values = profile_accumulators[f"fold{fold}"]
        arrays[key] = np.stack(values, axis=0).astype(np.float16) if values else np.empty((0, args.profile_downsample_bins), dtype=np.float16)
    if arrays:
        present = [arrays[f"profiles_fold{fold}"].astype(np.float32) for fold in args.folds if arrays[f"profiles_fold{fold}"].shape[0] == len(labels)]
        if present:
            arrays["profiles_mean"] = np.mean(np.stack(present, axis=0), axis=0).astype(np.float16)
    tmp_npz = profiles_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_npz, **arrays)
    tmp_npz.replace(profiles_path)

    summary = {
        "task": "cache_borzoi_ref_rich",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "input": str(args.shard or args.manifest),
        "label_path": str(label_path),
        "index_path": str(index_path),
        "profiles_path": str(profiles_path),
        "folds": args.folds,
        "selected_targets": selected,
        "profile_shape": {key: list(value.shape) for key, value in arrays.items()},
        "runtime_seconds": time.time() - started,
        "examples_per_second": len(rows) / max(1e-6, time.time() - started),
        **summarize_rich_labels(labels),
    }
    write_summary(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
