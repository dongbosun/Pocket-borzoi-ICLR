#!/usr/bin/env bash
#SBATCH --job-name=merge_ref_rich
#SBATCH --partition=zhanglab.p
#SBATCH --output=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%j.out
#SBATCH --error=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
echo "date=$(date)"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
mkdir -p /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
"$PYTHON_BIN" scripts/merge_borzoi_ref_rich.py \
  --cache-dir "${CACHE_DIR:-/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/ref_rich_shards}" \
  --out-prefix "${OUT_PREFIX:-/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_ref_rich}" \
  --manifest "${MANIFEST:-/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_gene_manifest.parquet}" \
  --overwrite
