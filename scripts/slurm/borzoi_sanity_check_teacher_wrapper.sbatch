#!/usr/bin/env bash
#SBATCH --job-name=borzoi_sanity
#SBATCH --partition=zhanglab.p
#SBATCH --output=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%j.out
#SBATCH --error=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

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
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/home/dongbos/miniconda3/envs/pocketreg_borzoi_py310}/lib:${CONDA_PREFIX:-/home/dongbos/miniconda3/envs/pocketreg_borzoi_py310}/lib/python3.10/site-packages/nvidia/cudnn/lib:${CONDA_PREFIX:-/home/dongbos/miniconda3/envs/pocketreg_borzoi_py310}/lib/python3.10/site-packages/nvidia/cublas/lib:${CONDA_PREFIX:-/home/dongbos/miniconda3/envs/pocketreg_borzoi_py310}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
"$PYTHON_BIN" scripts/borzoi_sanity_check_teacher_wrapper.py \
  --assets-config "${ASSETS_CONFIG:-configs/borzoi_assets.local.yaml}" \
  --manifest "${MANIFEST:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_gene_manifest.parquet}" \
  --fasta "${FASTA:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa}" \
  --ref-cache "${REF_CACHE:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_ref_labels.parquet}" \
  --rich-cache "${RICH_CACHE:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_ref_rich.labels.parquet}" \
  --delta-labels "${DELTA_LABELS:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_delta_labels.parquet}" \
  --out "${OUT:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/borzoi_sanity}"
