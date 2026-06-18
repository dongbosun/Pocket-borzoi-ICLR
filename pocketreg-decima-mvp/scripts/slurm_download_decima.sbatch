#!/bin/bash
#SBATCH --job-name=pocketreg-dl-decima
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/reports/slurm_download_decima_%j.out
#SBATCH --error=outputs/reports/slurm_download_decima_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON:-/home/dongbos/miniconda3/envs/doge/bin/python}"
OUT="${DECIMA_H5AD:-/lv_scratch/tmp/pocketreg-decima-mvp/data/raw/decima/metadata.h5ad}"
mkdir -p "$(dirname "${OUT}")"
"${PYTHON_BIN}" scripts/download_decima_data.py --out "${OUT}"
