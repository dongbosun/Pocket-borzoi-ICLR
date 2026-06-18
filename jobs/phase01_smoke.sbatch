#!/usr/bin/env bash
#SBATCH --job-name=borzoi_phase01_smoke
#SBATCH --partition=zhanglab.p
#SBATCH --output=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%j.out
#SBATCH --error=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail
echo "date=$(date)"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$REPO_ROOT"
mkdir -p /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/borzoi_assets_inspection_sbatch

"$PYTHON_BIN" --version
"$PYTHON_BIN" -m compileall -q src scripts tests
make test PYTHON="$PYTHON_BIN"
"$PYTHON_BIN" scripts/run_smoke_test.py
"$PYTHON_BIN" scripts/inspect_borzoi_assets.py \
  --assets-config configs/borzoi_assets_example.yaml \
  --out /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/borzoi_assets_inspection_sbatch

echo "phase01_smoke_done=$(date)"
