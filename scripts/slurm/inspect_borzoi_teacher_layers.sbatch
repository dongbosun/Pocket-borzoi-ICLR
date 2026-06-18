#!/usr/bin/env bash
#SBATCH --job-name=inspect_borzoi_layers
#SBATCH --partition=zhanglab.p
#SBATCH --output=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%j.out
#SBATCH --error=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

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
SITE_PACKAGES=$("$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/curand/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
"$PYTHON_BIN" scripts/inspect_borzoi_teacher_layers.py \
  --assets "${ASSETS_CONFIG:-configs/borzoi_assets.local.yaml}" \
  --manifest "${MANIFEST:-/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_gene_manifest.parquet}" \
  --fasta "${FASTA:-/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa}" \
  --out "${OUT:-/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/borzoi_teacher_layers}"
