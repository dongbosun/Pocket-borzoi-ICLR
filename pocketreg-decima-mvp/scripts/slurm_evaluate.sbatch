#!/bin/bash
#SBATCH --job-name=pocketreg-eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/reports/slurm_eval_%j.out
#SBATCH --error=outputs/reports/slurm_eval_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${CHECKPOINT:?Set CHECKPOINT=/path/to/best.pt}"
: "${MANIFEST:?Set MANIFEST=/path/to/manifest.parquet}"
: "${FASTA:?Set FASTA=/path/to/hg38.fa}"
OUT="${OUT:-outputs/reports/eval_decima_mvp}"
DEVICE="${DEVICE:-cpu}"
PYTHON_BIN="${PYTHON:-/home/dongbos/miniconda3/envs/doge/bin/python}"
"${PYTHON_BIN}" scripts/evaluate_decima_student.py \
  --checkpoint "${CHECKPOINT}" \
  --manifest "${MANIFEST}" \
  --fasta "${FASTA}" \
  --device "${DEVICE}" \
  --out "${OUT}"
