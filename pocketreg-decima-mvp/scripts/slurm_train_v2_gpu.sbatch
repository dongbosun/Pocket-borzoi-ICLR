#!/bin/bash
#SBATCH --job-name=pocketreg-v2-gpu
#SBATCH --partition=zhanglab.p
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/reports/slurm_train_v2_gpu_%j.out
#SBATCH --error=outputs/reports/slurm_train_v2_gpu_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${MANIFEST:?Set MANIFEST=/path/to/v2_manifest.parquet}"
: "${FASTA:?Set FASTA=/path/to/hg38.fa}"
RUN_NAME="${RUN_NAME:-decima_v2_gpu}"
CONFIG="${CONFIG:-configs/decima_v2_100k.yaml}"
PYTHON_BIN="${PYTHON:-/home/dongbos/miniconda3/envs/doge/bin/python}"
"${PYTHON_BIN}" scripts/train_decima_student_v2.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --fasta "${FASTA}" \
  --run-name "${RUN_NAME}"
"${PYTHON_BIN}" scripts/export_v2_target_only.py \
  --checkpoint "outputs/runs/${RUN_NAME}/checkpoints/best.pt" \
  --out "outputs/runs/${RUN_NAME}/checkpoints/best_target_only.pt"
