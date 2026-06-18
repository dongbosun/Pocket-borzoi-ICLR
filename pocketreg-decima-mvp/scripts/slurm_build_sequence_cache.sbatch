#!/usr/bin/env bash
#SBATCH --job-name=pocketreg-cache
#SBATCH --partition=zhanglab.p
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/reports/slurm_sequence_cache_%j.out
#SBATCH --error=outputs/reports/slurm_sequence_cache_%j.err

set -euo pipefail

cd /home/dongbos/Pocket-borzoi-ICLR/pocketreg-decima-mvp
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" outputs/reports

PYTHON="${PYTHON:-/home/dongbos/miniconda3/envs/doge/bin/python}"
MANIFEST="${MANIFEST:?set MANIFEST=/path/to/manifest.parquet}"
FASTA="${FASTA:?set FASTA=/path/to/hg38.fa}"
OUT_DIR="${OUT_DIR:?set OUT_DIR=/path/to/cache_dir}"
CONTEXT_LEN="${CONTEXT_LEN:-}"
CHANNELS="${CHANNELS:-5}"
VALIDATE_N="${VALIDATE_N:-32}"

ARGS=(
  scripts/build_sequence_cache.py
  --manifest "$MANIFEST"
  --fasta "$FASTA"
  --out-dir "$OUT_DIR"
  --channels "$CHANNELS"
  --validate-n "$VALIDATE_N"
)

if [[ -n "$CONTEXT_LEN" ]]; then
  ARGS+=(--context-len "$CONTEXT_LEN")
fi

"$PYTHON" "${ARGS[@]}"
