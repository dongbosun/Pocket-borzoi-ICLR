#!/usr/bin/env bash
#SBATCH --job-name=generate_k562_snvs
#SBATCH --partition=zhanglab.p
#SBATCH --output=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A.out
#SBATCH --error=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%A.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
echo "date=$(date)"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
CMD=("$PYTHON_BIN" scripts/generate_synthetic_snvs.py \
  --manifest "${MANIFEST:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_gene_manifest.parquet}" \
  --fasta "${FASTA:?FASTA required}" \
  --out "${OUT:-/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/variants/k562_synthetic_snvs.parquet}" \
  --snvs-per-gene "${SNVS_PER_GENE:-50}" \
  --seed "${SEED:-42}")
if [[ -n "${MAX_GENES:-}" ]]; then
  CMD+=(--max-genes "$MAX_GENES")
fi
"${CMD[@]}"
