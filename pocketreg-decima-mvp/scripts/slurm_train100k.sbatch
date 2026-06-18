#!/bin/bash
#SBATCH --job-name=pocketreg-train100k
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/reports/slurm_train100k_%j.out
#SBATCH --error=outputs/reports/slurm_train100k_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${MANIFEST:?Set MANIFEST=/path/to/manifest.parquet}"
: "${FASTA:?Set FASTA=/path/to/hg38.fa}"
RUN_NAME="${RUN_NAME:-decima_mvp_100k}"
PYTHON_BIN="${PYTHON:-/home/dongbos/miniconda3/envs/doge/bin/python}"
"${PYTHON_BIN}" scripts/train_decima_student.py \
  --config configs/decima_mvp_100k.yaml \
  --manifest "${MANIFEST}" \
  --fasta "${FASTA}" \
  --run-name "${RUN_NAME}"
