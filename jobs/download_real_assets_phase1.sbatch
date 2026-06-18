#!/usr/bin/env bash
#SBATCH --job-name=borzoi_real_assets_p1
#SBATCH --partition=zhanglab.p
#SBATCH --output=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%j.out
#SBATCH --error=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
echo "date=$(date)"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$REPO_ROOT"
mkdir -p /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/repos /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/assets/k562_mini /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38

if [[ ! -d /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/repos/baskerville/.git ]]; then
  git clone --depth 1 https://github.com/calico/baskerville.git /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/repos/baskerville
fi
if [[ ! -d /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/repos/borzoi/.git ]]; then
  git clone --depth 1 https://github.com/calico/borzoi.git /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/repos/borzoi
fi

wget -c -O /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/assets/k562_mini/fold0_model0_best.h5 \
  https://storage.googleapis.com/seqnn-share/borzoi/mini/k562_rna/f0/model0_best.h5
wget -c -O /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/assets/k562_mini/fold1_model0_best.h5 \
  https://storage.googleapis.com/seqnn-share/borzoi/mini/k562_rna/f1/model0_best.h5
wget -c -O /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/assets/k562_mini/targets.txt \
  https://storage.googleapis.com/seqnn-share/borzoi/mini/k562_rna/hg38/targets.txt
wget -c -O /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/assets/k562_mini/params.json \
  https://storage.googleapis.com/seqnn-share/borzoi/mini/k562_rna/params.json

if [[ ! -f /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa ]]; then
  wget -c -O /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa.gz \
    https://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
  gunzip -c /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa.gz > /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa.tmp
  mv /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa.tmp /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa
fi

if [[ ! -f /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/gencode41_basic_nort_protein.gtf ]]; then
  wget -c -O /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/gencode41_basic_nort_protein.gtf.gz \
    https://storage.googleapis.com/seqnn-share/helper/gencode41_basic_nort_protein.gtf.gz
  gunzip -c /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/gencode41_basic_nort_protein.gtf.gz \
    > /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/gencode41_basic_nort_protein.gtf.tmp
  mv /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/gencode41_basic_nort_protein.gtf.tmp \
    /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/gencode41_basic_nort_protein.gtf
fi

cat > configs/borzoi_assets.local.yaml <<EOF
borzoi_repo: /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/repos/borzoi
baskerville_repo: /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/repos/baskerville
westminster_repo:
k562_weights_fold0: /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/assets/k562_mini/fold0_model0_best.h5
k562_weights_fold1: /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/assets/k562_mini/fold1_model0_best.h5
k562_targets: /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/assets/k562_mini/targets.txt
k562_params: /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/assets/k562_mini/params.json
hg38_fasta: /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa
gencode_gtf: /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/gencode41_basic_nort_protein.gtf
EOF

"$PYTHON_BIN" scripts/inspect_borzoi_assets.py \
  --assets-config configs/borzoi_assets.local.yaml \
  --out /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/borzoi_assets_inspection_real

"$PYTHON_BIN" scripts/build_k562_gene_manifest.py \
  --assets-config configs/borzoi_assets.local.yaml \
  --fasta /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa \
  --gtf /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/gencode41_basic_nort_protein.gtf \
  --out /extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_gene_manifest.parquet \
  --input-len auto \
  --output-num-bins auto \
  --bin-size auto \
  --target-index auto \
  --aggregation gene_body_log1p_mean \
  --max-genes 5000

echo "download_real_assets_phase1_done=$(date)"
