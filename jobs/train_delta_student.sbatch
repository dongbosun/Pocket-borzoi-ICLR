#!/usr/bin/env bash
#SBATCH --job-name=train_delta_k562
#SBATCH --partition=zhanglab.p
#SBATCH --output=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A.out
#SBATCH --error=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
echo "date=$(date)"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
mkdir -p /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
"$PYTHON_BIN" scripts/train_borzoi_delta_student.py \
  --config "${CONFIG:-configs/borzoi_k562_delta_100k.yaml}" \
  --delta-labels "${DELTA_LABELS:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_delta_labels.parquet}" \
  --fasta "${FASTA:?FASTA required}" \
  --run-name "${RUN_NAME:-k562_delta_100k_8kb}"
