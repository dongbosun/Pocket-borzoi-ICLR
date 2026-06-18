#!/bin/bash
#SBATCH --job-name=pocketreg-dl-hg38
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/reports/slurm_download_hg38_%j.out
#SBATCH --error=outputs/reports/slurm_download_hg38_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON:-/home/dongbos/miniconda3/envs/doge/bin/python}"
OUT_DIR="${HG38_DIR:-/lv_scratch/tmp/pocketreg-decima-mvp/data/raw/hg38}"
mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"
wget -c -O hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gzip -df hg38.fa.gz
"${PYTHON_BIN}" -c "from pyfaidx import Fasta; Fasta('hg38.fa'); print('indexed hg38.fa')"
