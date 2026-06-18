#!/bin/bash
#SBATCH --job-name=pocketreg-v2-manifest
#SBATCH --partition=zhanglab.p
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/reports/slurm_manifest_v2_%j.out
#SBATCH --error=outputs/reports/slurm_manifest_v2_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${ADATA:?Set ADATA=/path/to/metadata.h5ad}"
: "${FASTA:?Set FASTA=/path/to/hg38.fa}"
: "${OUT:?Set OUT=/path/to/v2_manifest.parquet}"
PYTHON_BIN="${PYTHON:-/home/dongbos/miniconda3/envs/doge/bin/python}"
ARGS=(
  scripts/build_decima_manifest_v2.py
  --adata "${ADATA}"
  --fasta "${FASTA}"
  --out "${OUT}"
  --context-len "${CONTEXT_LEN:-65536}"
  --final-layer "${FINAL_LAYER:-preds}"
  --organ "${ORGAN:-CNS}"
  --disease "${DISEASE:-healthy}"
  --cell-type-contains "${CELL_TYPE_CONTAINS:-astro}"
  --split-mode "${SPLIT_MODE:-chromosome}"
  --aux-pca-components "${AUX_PCA_COMPONENTS:-8}"
  --aux-max-obs "${AUX_MAX_OBS:-64}"
)

if [[ -n "${TARGET_INDEX:-}" ]]; then
  ARGS+=(--target-index "${TARGET_INDEX}")
fi

if [[ -n "${MAX_GENES:-}" ]]; then
  ARGS+=(--max-genes "${MAX_GENES}")
fi

"${PYTHON_BIN}" "${ARGS[@]}"
