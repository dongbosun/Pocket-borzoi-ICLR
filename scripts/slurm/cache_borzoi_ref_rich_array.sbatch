#!/usr/bin/env bash
#SBATCH --job-name=cache_ref_rich
#SBATCH --partition=zhanglab.p
#SBATCH --output=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A_%a.out
#SBATCH --error=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
echo "date=$(date)"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
mkdir -p /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm "${CACHE_DIR:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/ref_rich_shards}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
SITE_PACKAGES=$("$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/curand/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PADDED=$(printf "%03d" "$TASK_ID")
SHARD="${SHARD_DIR:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/processed/shards/ref_k562}/shard_${PADDED}.parquet"
OUT_PREFIX="${CACHE_DIR:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/ref_rich_shards}/shard_${PADDED}"
"$PYTHON_BIN" scripts/cache_borzoi_ref_rich.py \
  --assets-config "${ASSETS_CONFIG:-configs/borzoi_assets.local.yaml}" \
  --distill-config "${DISTILL_CONFIG:-configs/borzoi_distill_v2.yaml}" \
  --target-config "${TARGET_CONFIG:-configs/borzoi_k562_targets.local.yaml}" \
  --shard "$SHARD" \
  --fasta "${FASTA:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa}" \
  --out-prefix "$OUT_PREFIX" \
  --batch-size "${BATCH_SIZE:-1}" \
  --profile-downsample-bins "${PROFILE_BINS:-384}"
