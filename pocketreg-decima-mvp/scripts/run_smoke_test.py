#!/usr/bin/env python
"""End-to-end synthetic smoke test for the Decima MVP path."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketreg.data.manifest import build_manifest_from_anndata, save_manifest_outputs
from pocketreg.training.train_loop import evaluate_checkpoint, train_from_config
from pocketreg.training.utils import setup_logging

LOGGER = logging.getLogger("run_smoke_test")
BASES = np.array(list("ACGT"))


def motif_count(seq: str, motif: str = "ACGT") -> int:
    return sum(1 for i in range(0, len(seq) - len(motif) + 1) if seq[i : i + len(motif)] == motif)


def padded_window(genome: dict[str, str], chrom: str, start: int, end: int) -> str:
    chrom_seq = genome[chrom]
    left = max(0, -start)
    right = max(0, end - len(chrom_seq))
    body = chrom_seq[max(0, start) : min(end, len(chrom_seq))]
    return ("N" * left) + body + ("N" * right)


def write_fasta(genome: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for chrom, seq in genome.items():
            f.write(f">{chrom}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def create_toy_data(out_dir: Path, context_len: int, seed: int) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "toy.fa"
    adata_path = out_dir / "toy_decima.h5ad"
    chroms = [f"chr{i}" for i in range(1, 23)]
    genome = {chrom: "".join(rng.choice(BASES, size=180_000)) for chrom in chroms}

    n_genes = 300
    gene_rows = []
    labels_base = []
    for gene_idx in range(n_genes):
        chrom = chroms[gene_idx % len(chroms)]
        tss = int(rng.integers(context_len, len(genome[chrom]) - context_len))
        # Inject a simple motif signal into a subset of genes.
        insert_n = gene_idx % 8
        seq_list = list(genome[chrom])
        for j in range(insert_n):
            pos = min(len(seq_list) - 4, tss - 128 + j * 16)
            seq_list[pos : pos + 4] = list("ACGT")
        genome[chrom] = "".join(seq_list)
        window = padded_window(genome, chrom, tss - context_len // 2, tss + context_len // 2)
        y = motif_count(window) + 2.0 * (window.count("G") + window.count("C")) / len(window)
        labels_base.append(y)
        gene_rows.append(
            {
                "chrom": chrom,
                "start": tss - 500,
                "end": tss + 500,
                "strand": "+" if gene_idx % 2 == 0 else "-",
                "gene_type": "protein_coding",
                "gene_id": f"ENSGTOY{gene_idx:06d}",
                "gene_name": f"TOY{gene_idx}",
                "gene_start": tss - 500,
                "gene_end": tss + 500,
                "gene_length": 1000,
                "ensembl_canonical_tss": tss,
                "frac_N": 0.0,
                "frac_nan": 0.0,
                "mean_counts": float(10 + y),
                "n_tracks": 1,
                "dataset": "toy",
                "fold": gene_idx % 5,
                "pearson": 0.5,
                "size_factor_pearson": 0.4,
            }
        )
    write_fasta(genome, fasta_path)

    obs = pd.DataFrame(
        {
            "cell_type": ["astrocyte"] + [f"celltype_{i}" for i in range(1, 20)],
            "tissue": ["brain"] * 20,
            "organ": ["brain"] * 20,
            "disease": ["healthy"] * 20,
            "study": ["toy"] * 20,
            "dataset": ["toy"] * 20,
            "region": ["cortex"] * 20,
            "subregion": ["toy_subregion"] * 20,
            "celltype_coarse": ["glia"] * 20,
            "n_cells": np.linspace(1000, 100, 20),
            "total_counts": np.linspace(1e5, 5e4, 20),
            "n_genes": [n_genes] * 20,
            "train_pearson": np.linspace(0.8, 0.2, 20),
            "val_pearson": np.linspace(0.75, 0.15, 20),
            "test_pearson": np.linspace(0.7, 0.1, 20),
        }
    )
    var = pd.DataFrame(gene_rows)
    base = np.asarray(labels_base, dtype=np.float32)
    preds = np.stack([base + i * 0.05 + rng.normal(0, 0.05, n_genes) for i in range(20)]).astype(
        np.float32
    )
    toy = ad.AnnData(X=np.zeros((20, n_genes), dtype=np.float32), obs=obs, var=var)
    toy.layers["preds"] = preds
    toy.layers["v1_rep0"] = preds + rng.normal(0, 0.02, preds.shape).astype(np.float32)
    toy.layers["v1_rep1"] = preds + rng.normal(0, 0.02, preds.shape).astype(np.float32)
    toy.write_h5ad(adata_path)
    return adata_path, fasta_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "smoke")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    setup_logging()

    adata_path, fasta_path = create_toy_data(args.out_dir, args.context_len, args.seed)
    toy = ad.read_h5ad(adata_path)
    manifest, target_metadata, summary = build_manifest_from_anndata(
        toy,
        pseudobulk_idx=0,
        label_layer="preds",
        context_len=args.context_len,
        split_mode="chromosome",
        max_genes=120,
        fasta_path=fasta_path,
        skip_fasta_check=False,
    )
    manifest_path = args.out_dir / "manifest.parquet"
    save_manifest_outputs(manifest, manifest_path, target_metadata, summary)

    config = {
        "seed": args.seed,
        "manifest_path": str(manifest_path),
        "fasta_path": str(fasta_path),
        "output_dir": str(ROOT / "outputs" / "runs" / "smoke"),
        "context_len": args.context_len,
        "model": {
            "preset": "tiny_100k",
            "channels": 16,
            "num_blocks": 2,
            "kernel_size": 3,
            "stem_stride": 8,
            "pool_every": 2,
            "dropout": 0.05,
            "head_hidden": 32,
            "norm": "groupnorm",
        },
        "train": {
            "batch_size": 16,
            "num_workers": 0,
            "max_epochs": 2,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "loss": "huber",
            "huber_delta": 1.0,
            "scheduler": "cosine",
            "early_stopping_patience": 3,
            "grad_clip_norm": 1.0,
            "amp": False,
            "device": "cpu",
            "target_standardize": True,
            "cache_size": 64,
        },
        "logging": {"save_plots": True, "save_predictions": True},
    }
    run_dir = train_from_config(config)
    eval_dir = ROOT / "outputs" / "reports" / "smoke_eval"
    evaluate_checkpoint(
        run_dir / "checkpoints" / "best.pt",
        manifest_path,
        fasta_path,
        eval_dir,
        device_name="cpu",
        batch_size=16,
    )

    # Keep the benchmark tiny for smoke mode; use benchmark_inference.py for real timing.
    from subprocess import run

    bench_json = ROOT / "outputs" / "reports" / "smoke_benchmark.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_inference.py"),
        "--checkpoint",
        str(run_dir / "checkpoints" / "best.pt"),
        "--manifest",
        str(manifest_path),
        "--fasta",
        str(fasta_path),
        "--device",
        "cpu",
        "--batch-sizes",
        "4",
        "--num-warmup",
        "1",
        "--num-steps",
        "2",
        "--out",
        str(bench_json),
    ]
    run(cmd, check=True)

    required = [
        run_dir / "checkpoints" / "best.pt",
        run_dir / "metrics.json",
        run_dir / "predictions_test.parquet",
        run_dir / "plots" / "parity_test.png",
        eval_dir / "metrics.json",
        bench_json,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Smoke test did not produce required outputs: {missing}")
    LOGGER.info("Smoke test complete. Run outputs: %s", run_dir)


if __name__ == "__main__":
    main()
