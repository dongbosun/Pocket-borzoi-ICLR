# pocketreg-decima-mvp

Week-1 MVP for distilling one Decima pseudobulk gene-expression prediction into a small local DNA CNN.

The core experiment is:

```text
student(sequence around gene g) ~= DecimaPrediction[pseudobulk p, gene g]
```

The input is gene-centered hg38 DNA around the canonical TSS. The target is a scalar from a precomputed Decima prediction layer in `Genentech/decima-data` `metadata.h5ad`.

## What This Does

- Selects one Decima pseudobulk / cell-type-condition row from AnnData `.obs`.
- Extracts teacher labels from `.layers["preds"]`, a replicate layer, or an average of `v1_rep0` through `v1_rep3`.
- Builds a one-row-per-gene manifest with TSS-centered sequence windows and chromosome-held-out splits.
- Trains a small 1D CNN student on raw DNA sequence only.
- Reports Pearson, Spearman, R2, MAE, RMSE, parameter count, model size, predictions, plots, and CPU/MPS/CUDA latency.

## What This Does Not Do Yet

- No Borzoi.
- No variant effect prediction.
- No Decima teacher inference.
- No real expression supervised training.
- No clinical prediction.

## Installation

```bash
conda create -n pocketreg-decima python=3.10 -y
conda activate pocketreg-decima
pip install -r requirements.txt
pip install -e .
```

Or with venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Cluster Rule

Do not run training, smoke training, evaluation, or benchmark jobs on the head node. Submit them with `sbatch` so they run on compute nodes.

## Quick Smoke Test

On the cluster:

```bash
sbatch scripts/slurm_smoke_test.sbatch
```

The underlying command is:

```bash
python scripts/run_smoke_test.py
```

Unit tests are lightweight utility tests:

```bash
pytest -q
```

If your cluster treats all Torch execution as compute work, submit tests too:

```bash
make test-sbatch
```

## Real Decima MVP

Download Decima metadata:

```bash
python scripts/download_decima_data.py --out data/raw/decima/metadata.h5ad
```

Inspect the AnnData object:

```bash
python scripts/inspect_decima_data.py \
  --adata data/raw/decima/metadata.h5ad \
  --out outputs/reports/decima_inspection
```

Build a 64 kb chromosome-split manifest:

```bash
python scripts/build_decima_manifest.py \
  --adata data/raw/decima/metadata.h5ad \
  --fasta /path/to/hg38.fa \
  --out data/processed/decima_mvp/manifest.parquet \
  --context-len 65536 \
  --label-layer preds \
  --organ brain \
  --disease healthy \
  --cell-type-contains astro \
  --split-mode chromosome \
  --max-genes 5000
```

Submit 100k-parameter training:

```bash
sbatch --export=ALL,MANIFEST=data/processed/decima_mvp/manifest.parquet,FASTA=/path/to/hg38.fa,RUN_NAME=decima_astro_100k_64kb scripts/slurm_train100k.sbatch
```

Equivalent Python command, for a compute node shell only:

```bash
python scripts/train_decima_student.py \
  --config configs/decima_mvp_100k.yaml \
  --manifest data/processed/decima_mvp/manifest.parquet \
  --fasta /path/to/hg38.fa \
  --run-name decima_astro_100k_64kb
```

Submit evaluation:

```bash
sbatch --export=ALL,CHECKPOINT=outputs/runs/decima_astro_100k_64kb/checkpoints/best.pt,MANIFEST=data/processed/decima_mvp/manifest.parquet,FASTA=/path/to/hg38.fa,OUT=outputs/reports/eval_decima_astro_100k_64kb scripts/slurm_evaluate.sbatch
```

Submit CPU benchmark:

```bash
sbatch --export=ALL,CHECKPOINT=outputs/runs/decima_astro_100k_64kb/checkpoints/best.pt,MANIFEST=data/processed/decima_mvp/manifest.parquet,FASTA=/path/to/hg38.fa,DEVICE=cpu,OUT=outputs/reports/bench_cpu.json scripts/slurm_benchmark.sbatch
```

## Selecting Another Pseudobulk

Use an explicit AnnData obs integer position:

```bash
python scripts/build_decima_manifest.py \
  --adata data/raw/decima/metadata.h5ad \
  --fasta /path/to/hg38.fa \
  --out data/processed/decima_mvp/manifest.parquet \
  --target-index 123
```

Or use filters:

```bash
python scripts/build_decima_manifest.py \
  --adata data/raw/decima/metadata.h5ad \
  --fasta /path/to/hg38.fa \
  --out data/processed/decima_mvp/manifest.parquet \
  --organ brain \
  --disease healthy \
  --cell-type-contains astro \
  --region-contains cortex
```

You can also pass `--target-query`, but the simple filter flags are more robust across pandas versions.

## Output Files

Training writes to `outputs/runs/{run_name}/`:

- `config.yaml`: resolved run config.
- `target_metadata.json`: selected pseudobulk metadata.
- `manifest_summary.json`: gene counts, split counts, label stats.
- `checkpoints/best.pt` and `checkpoints/last.pt`: model checkpoints.
- `metrics.json`: student metrics by split.
- `metrics_baselines.json`: train-mean and metadata Ridge baselines.
- `train_log.csv`: epoch-level loss and validation metrics.
- `predictions_{train,val,test}.parquet`: raw-scale predictions.
- `plots/*.png`: training curves, parity plots, residuals, distributions.
- `model_summary.txt`: architecture, parameter count, model size.

Benchmark JSON reports model-only and end-to-end FASTA-plus-model latency by batch size.

## Makefile

```bash
make install
make test
make smoke
make inspect ADATA=data/raw/decima/metadata.h5ad
make manifest ADATA=data/raw/decima/metadata.h5ad FASTA=/path/to/hg38.fa
make train100k MANIFEST=data/processed/decima_mvp/manifest.parquet FASTA=/path/to/hg38.fa RUN_NAME=decima_astro_100k_64kb
make bench CHECKPOINT=outputs/runs/decima_astro_100k_64kb/checkpoints/best.pt MANIFEST=data/processed/decima_mvp/manifest.parquet FASTA=/path/to/hg38.fa DEVICE=cpu OUT=outputs/reports/bench_cpu.json
```

`make smoke`, `make train100k`, `make eval`, and `make bench` submit Slurm jobs.

## Known Caveats

- This is teacher-output distillation, not validation against observed expression.
- Coordinate conventions should be checked carefully against the FASTA and Decima metadata.
- Chromosome split is the default to reduce leakage.
- Performance depends on selected pseudobulk, gene filters, context length, and model size.
- Later stages should use Decima's own inference pipeline for exact teacher behavior and variant-effect tutorials for delta prediction.
