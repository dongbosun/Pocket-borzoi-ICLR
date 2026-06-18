#!/usr/bin/env bash
#SBATCH --job-name=eval_delta_k562
#SBATCH --output=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A.out
#SBATCH --error=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
python scripts/evaluate_borzoi_delta_student.py \
  --checkpoint "${CHECKPOINT:?CHECKPOINT required}" \
  --delta-labels "${DELTA_LABELS:-/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_delta_labels.parquet}" \
  --fasta "${FASTA:?FASTA required}" \
  --out "${OUT:-/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/eval_k562_delta_100k_8kb}"
