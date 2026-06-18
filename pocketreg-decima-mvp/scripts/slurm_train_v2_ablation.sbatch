#!/bin/bash
#SBATCH --job-name=pocketreg-v2-abl
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/ablations/v2_5k_obs88_64kb/logs/slurm_%x_%j.out
#SBATCH --error=outputs/ablations/v2_5k_obs88_64kb/logs/slurm_%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${CONFIG:?Set CONFIG=/path/to/ablation.yaml}"
: "${FASTA:?Set FASTA=/path/to/hg38.fa}"
: "${RUN_NAME:?Set RUN_NAME=ablation_name}"
PYTHON_BIN="${PYTHON:-/home/dongbos/miniconda3/envs/doge/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_${SLURM_JOB_ID}}"
mkdir -p outputs/ablations/v2_5k_obs88_64kb/logs
"${PYTHON_BIN}" scripts/train_decima_student_v2.py \
  --config "${CONFIG}" \
  --fasta "${FASTA}" \
  --run-name "${RUN_NAME}"
"${PYTHON_BIN}" scripts/export_v2_target_only.py \
  --checkpoint "outputs/ablations/v2_5k_obs88_64kb/runs/${RUN_NAME}/checkpoints/best.pt" \
  --out "outputs/ablations/v2_5k_obs88_64kb/runs/${RUN_NAME}/checkpoints/best_target_only.pt"
