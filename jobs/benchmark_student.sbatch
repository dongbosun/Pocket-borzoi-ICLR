#!/usr/bin/env bash
#SBATCH --job-name=bench_student
#SBATCH --output=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A.out
#SBATCH --error=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
python scripts/benchmark_borzoi_student.py \
  --checkpoint "${CHECKPOINT:?CHECKPOINT required}" \
  --labels "${LABELS:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_ref_labels.parquet}" \
  --fasta "${FASTA:?FASTA required}" \
  --device "${DEVICE:-cpu}" \
  --out "${OUT:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/results/benchmarks/k562_track_100k_cpu.json}"
