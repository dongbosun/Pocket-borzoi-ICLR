#!/bin/bash
#SBATCH --job-name=pocketreg-smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/reports/slurm_smoke_%j.out
#SBATCH --error=outputs/reports/slurm_smoke_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON:-/home/dongbos/miniconda3/envs/doge/bin/python}"
"${PYTHON_BIN}" scripts/run_smoke_test.py
