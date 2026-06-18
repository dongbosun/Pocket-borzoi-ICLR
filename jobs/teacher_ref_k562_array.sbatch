#!/usr/bin/env bash
#SBATCH --job-name=teacher_ref_k562
#SBATCH --partition=zhanglab.p
#SBATCH --output=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A_%a.out
#SBATCH --error=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A_%a.err
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
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-pocketreg_borzoi}"
if [[ -n "$CONDA_SH" && -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi
cd "$REPO_ROOT"
mkdir -p /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/cache/cuda "${CACHE_DIR:?CACHE_DIR required}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/cache/cuda}"
TASK_ID_PADDED=$(printf "%03d" "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}")
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$REPO_ROOT/src:/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/repos/baskerville/src:/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/repos/borzoi/src:${PYTHONPATH:-}"
PY_SITE=$("$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])')
export PATH="$PY_SITE/nvidia/cuda_nvcc/bin:${PATH:-}"
if [[ -z "${XLA_FLAGS:-}" ]]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=$PY_SITE/nvidia/cuda_nvcc"
fi
CMD=("$PYTHON_BIN" scripts/run_borzoi_teacher_ref.py \
  --config "${CONFIG:?CONFIG required}" \
  --assets-config "${ASSETS_CONFIG:?ASSETS_CONFIG required}" \
  --fasta "${FASTA:?FASTA required}" \
  --shard "${SHARD_DIR:?SHARD_DIR required}/shard_${TASK_ID_PADDED}.parquet" \
  --out "${CACHE_DIR}/shard_${TASK_ID_PADDED}.parquet" \
  --batch-size "${BATCH_SIZE:-4}")
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  CMD+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ -n "${OVERWRITE:-}" ]]; then
  CMD+=(--overwrite)
fi
"${CMD[@]}"
