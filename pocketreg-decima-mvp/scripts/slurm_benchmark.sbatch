#!/bin/bash
#SBATCH --job-name=pocketreg-bench
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/reports/slurm_bench_%j.out
#SBATCH --error=outputs/reports/slurm_bench_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${CHECKPOINT:?Set CHECKPOINT=/path/to/best.pt}"
: "${MANIFEST:?Set MANIFEST=/path/to/manifest.parquet}"
: "${FASTA:?Set FASTA=/path/to/hg38.fa}"
DEVICE="${DEVICE:-cpu}"
OUT="${OUT:-outputs/reports/benchmark_${DEVICE}.json}"
PYTHON_BIN="${PYTHON:-/home/dongbos/miniconda3/envs/doge/bin/python}"
"${PYTHON_BIN}" scripts/benchmark_inference.py \
  --checkpoint "${CHECKPOINT}" \
  --manifest "${MANIFEST}" \
  --fasta "${FASTA}" \
  --device "${DEVICE}" \
  --batch-sizes 1 8 32 \
  --out "${OUT}"
