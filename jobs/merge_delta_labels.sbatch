#!/usr/bin/env bash
#SBATCH --job-name=merge_delta_k562
#SBATCH --partition=zhanglab.p
#SBATCH --output=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A.out
#SBATCH --error=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
echo "date=$(date)"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
CMD=("$PYTHON_BIN" scripts/merge_teacher_delta_cache.py \
  --cache-dir "${CACHE_DIR:-/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/delta_k562}" \
  --out "${OUT:-/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_delta_labels.parquet}" \
  --variants "${VARIANTS:-/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/variants/k562_synthetic_snvs.parquet}")
if [[ -n "${REQUIRE_COMPLETE:-}" ]]; then
  CMD+=(--require-complete)
fi
"${CMD[@]}"
