#!/usr/bin/env python
"""CPU-only toy smoke test for Phase 0 wiring."""

from __future__ import annotations

import json
import random
from pathlib import Path

import _path  # noqa: F401
import numpy as np

from pocketreg.data.fasta import FastaReader
from pocketreg.data.genes import build_gene_manifest_rows, manifest_summary
from pocketreg.data.gtf import GeneRecord
from pocketreg.data.manifest import write_table
from pocketreg.data.sequence import apply_snv, gc_content
from pocketreg.data.variants import generate_snvs_for_manifest_row
from pocketreg.eval.plots import save_parity_plot
from pocketreg.training.metrics import mae, pearsonr, r2_score, rmse
from pocketreg.training.utils import set_seed


MOTIFS = ("AATAAA", "GATA", "CGCG")


def motif_counts(seq: str) -> np.ndarray:
    return np.array([seq.count(motif) for motif in MOTIFS] + [gc_content(seq), 1.0], dtype=float)


def fit_linear(features: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    xtx = features.T @ features + ridge * np.eye(features.shape[1])
    return np.linalg.solve(xtx, features.T @ y)


def write_toy_fasta(path: Path, rng: random.Random) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for chrom_i in range(1, 23):
        bases = [rng.choice("ACGT") for _ in range(24000)]
        for pos in range(1000, 23000, 3000):
            bases[pos : pos + 6] = list("AATAAA")
        for pos in range(1700, 23000, 3500):
            bases[pos : pos + 4] = list("GATA")
        seq = "".join(bases)
        chunks.append(f">chr{chrom_i}\n")
        chunks.extend(seq[i : i + 80] + "\n" for i in range(0, len(seq), 80))
    path.write_text("".join(chunks))


def make_toy_genes() -> list[GeneRecord]:
    genes = []
    for chrom_i in range(1, 23):
        for j, start in enumerate((3500, 9500, 15500)):
            genes.append(
                GeneRecord(
                    gene_id=f"TOY{chrom_i:02d}{j}",
                    gene_name=f"ToyGene{chrom_i:02d}_{j}",
                    chrom=f"chr{chrom_i}",
                    start_0based=start,
                    end_0based=start + 1800,
                    strand="+" if j % 2 == 0 else "-",
                    gene_type="protein_coding",
                )
            )
    return genes


def fake_q(seq: str, rng: random.Random) -> float:
    return float(seq.count("AATAAA") + 0.5 * seq.count("GATA") - 0.25 * seq.count("CGCG") + rng.gauss(0, 0.02))


def main() -> None:
    set_seed(42)
    rng = random.Random(42)
    out = Path("/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/runs/toy_smoke")
    checkpoint_dir = Path("/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/checkpoints/toy_smoke")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = Path("/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/raw/toy.fa")
    write_toy_fasta(fasta_path, rng)
    fasta = FastaReader(fasta_path)

    rows = build_gene_manifest_rows(
        genes=make_toy_genes(),
        fasta=fasta,
        input_len=4096,
        output_num_bins=64,
        bin_size=32,
        target_index=0,
        target_identifier="toy_k562_rna",
        target_description="toy K562 RNA motif target",
        max_genes=None,
        min_gene_overlap_fraction=0.1,
        source="toy",
    )
    write_table(rows, Path("/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/toy_gene_manifest.parquet"))
    (out / "manifest_summary.json").write_text(json.dumps(manifest_summary(rows), indent=2) + "\n")

    label_rows = []
    for row in rows:
        seq = fasta.fetch(row["chrom"], int(row["seq_start"]), int(row["seq_end"]))
        label_rows.append({**row, "q_teacher": fake_q(seq, rng), "status": "success"})
    write_table(label_rows, Path("/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/toy_ref_labels.parquet"))

    train = [row for row in label_rows if row["split"] == "train"]
    test = [row for row in label_rows if row["split"] == "test"]
    x_train = np.vstack([motif_counts(fasta.fetch(r["chrom"], int(r["seq_start"]), int(r["seq_end"]))) for r in train])
    y_train = np.array([r["q_teacher"] for r in train])
    weights = fit_linear(x_train, y_train)
    predictions = {}
    metrics = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in label_rows if row["split"] == split]
        x = np.vstack([motif_counts(fasta.fetch(r["chrom"], int(r["seq_start"]), int(r["seq_end"]))) for r in split_rows])
        y = np.array([r["q_teacher"] for r in split_rows])
        pred = x @ weights
        metrics[split] = {
            "pearson": pearsonr(y, pred),
            "r2": r2_score(y, pred),
            "mae": mae(y, pred),
            "rmse": rmse(y, pred),
        }
        predictions[split] = [{"example_id": r["example_id"], "y": float(a), "pred": float(b)} for r, a, b in zip(split_rows, y, pred)]
        write_table(predictions[split], out / f"predictions_{split}.parquet")
        save_parity_plot(y, pred, out / "plots" / f"track_parity_{split}.png", title=f"toy track {split}")

    variants = []
    for row in rows:
        variants.extend(
            snv.__dict__
            for snv in generate_snvs_for_manifest_row(
                row,
                fasta=fasta,
                snvs_per_gene=4,
                rng=rng,
                tss_flank=2048,
            )
        )
    write_table(variants, Path("/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/variants/toy_synthetic_snvs.parquet"))

    ref_by_example = {row["example_id"]: row for row in rows}
    delta_rows = []
    for var in variants:
        row = ref_by_example[var["example_id"]]
        ref_seq = fasta.fetch(row["chrom"], int(row["seq_start"]), int(row["seq_end"]))
        alt_seq = apply_snv(ref_seq, int(var["pos_0based"]), int(row["seq_start"]), var["ref"], var["alt"]).alt_sequence
        delta_rows.append(
            {
                **var,
                "delta_teacher": fake_q(alt_seq, rng) - fake_q(ref_seq, rng),
                "status": "success",
            }
        )
    write_table(delta_rows, Path("/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/toy_delta_labels.parquet"))

    split_delta_train = [row for row in delta_rows if row["split"] == "train"]
    x_delta_train = np.vstack(
        [
            np.array(
                [
                    row["distance_to_tss"] / 2048,
                    1.0 if row["ref"] in "AG" and row["alt"] in "CT" else 0.0,
                    1.0,
                ]
            )
            for row in split_delta_train
        ]
    )
    y_delta_train = np.array([row["delta_teacher"] for row in split_delta_train])
    delta_weights = fit_linear(x_delta_train, y_delta_train)
    delta_metrics = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in delta_rows if row["split"] == split]
        x = np.vstack(
            [
                np.array(
                    [
                        row["distance_to_tss"] / 2048,
                        1.0 if row["ref"] in "AG" and row["alt"] in "CT" else 0.0,
                        1.0,
                    ]
                )
                for row in split_rows
            ]
        )
        y = np.array([row["delta_teacher"] for row in split_rows])
        pred = x @ delta_weights
        delta_metrics[split] = {
            "pearson": pearsonr(y, pred),
            "r2": r2_score(y, pred),
            "mae": mae(y, pred),
            "rmse": rmse(y, pred),
        }
        write_table(
            [{"variant_example_id": r["variant_example_id"], "y": float(a), "pred": float(b)} for r, a, b in zip(split_rows, y, pred)],
            out / f"delta_predictions_{split}.parquet",
        )
        save_parity_plot(y, pred, out / "plots" / f"delta_parity_{split}.png", title=f"toy delta {split}")

    all_metrics = {"track": metrics, "delta": delta_metrics}
    (out / "metrics.json").write_text(json.dumps(all_metrics, indent=2, sort_keys=True) + "\n")
    (checkpoint_dir / "checkpoint_best.pt").write_text("toy linear checkpoint placeholder\n")
    required = [
        out / "metrics.json",
        checkpoint_dir / "checkpoint_best.pt",
        out / "predictions_test.parquet",
        out / "delta_predictions_test.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Toy smoke outputs missing: {missing}")
    print(json.dumps({"status": "ok", "out": str(out), "metrics": all_metrics}, indent=2))


if __name__ == "__main__":
    main()
