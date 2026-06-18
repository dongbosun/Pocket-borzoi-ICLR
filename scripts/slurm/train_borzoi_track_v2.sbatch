#!/usr/bin/env bash
#SBATCH --job-name=train_track_v2
#SBATCH --partition=zhanglab.p
#SBATCH --output=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%j.out
#SBATCH --error=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%j.err
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
mkdir -p /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
"$PYTHON_BIN" scripts/train_borzoi_track_student_v2.py \
  --config "${CONFIG:-configs/borzoi_track_v2_full_128kb.yaml}" \
  --run-name "${RUN_NAME:-}"
