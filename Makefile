.PHONY: test smoke inspect-assets inspect-processed build-manifest ref-shards submit-ref-teacher merge-ref train-track eval-track bench-track snvs delta-shards submit-delta-teacher merge-delta train-delta eval-delta

PYTHON ?= python
ASSETS_CONFIG ?= configs/borzoi_assets.local.yaml
DATA_DIR ?= /path/to/processed_k562
FASTA ?= /path/to/hg38.fa
GTF ?= /path/to/gencode.vXX.annotation.gtf.gz
MANIFEST ?= /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_gene_manifest.parquet

test:
	@if command -v pytest >/dev/null 2>&1; then pytest -q; else $(PYTHON) -m unittest discover -s tests; fi

smoke:
	$(PYTHON) scripts/run_smoke_test.py

inspect-assets:
	$(PYTHON) scripts/inspect_borzoi_assets.py --assets-config $(ASSETS_CONFIG)

inspect-processed:
	$(PYTHON) scripts/inspect_borzoi_processed_data.py --data-dir $(DATA_DIR)

build-manifest:
	$(PYTHON) scripts/build_k562_gene_manifest.py --assets-config $(ASSETS_CONFIG) --fasta $(FASTA) --gtf $(GTF) --out $(MANIFEST)

ref-shards:
	$(PYTHON) scripts/make_teacher_ref_shards.py --manifest $(MANIFEST) --out-dir /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/processed/shards/ref_k562 --num-shards 32

submit-ref-teacher:
	$(PYTHON) scripts/submit_sbatch.py --job jobs/teacher_ref_k562_array.sbatch --array 0-31 --export ALL,REPO_ROOT=$(PWD),CONFIG=configs/borzoi_k562_track_100k.yaml,SHARD_DIR=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/processed/shards/ref_k562,CACHE_DIR=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/ref_k562,FASTA=$(FASTA),ASSETS_CONFIG=$(ASSETS_CONFIG),BATCH_SIZE=4

merge-ref:
	sbatch jobs/merge_ref_labels.sbatch

train-track:
	sbatch jobs/train_track_student.sbatch

eval-track:
	sbatch jobs/evaluate_track_student.sbatch

bench-track:
	sbatch jobs/benchmark_student.sbatch

snvs:
	$(PYTHON) scripts/generate_synthetic_snvs.py --manifest $(MANIFEST) --fasta $(FASTA)

delta-shards:
	$(PYTHON) scripts/make_teacher_delta_shards.py --variants /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/variants/k562_synthetic_snvs.parquet --out-dir /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/processed/shards/delta_k562 --num-shards 64

submit-delta-teacher:
	$(PYTHON) scripts/submit_sbatch.py --job jobs/teacher_delta_k562_array.sbatch --array 0-63 --export ALL,REPO_ROOT=$(PWD),CONFIG=configs/borzoi_k562_delta_100k.yaml,SHARD_DIR=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/processed/shards/delta_k562,CACHE_DIR=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/delta_k562,FASTA=$(FASTA),ASSETS_CONFIG=$(ASSETS_CONFIG),REF_CACHE=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_ref_labels.parquet,BATCH_SIZE=4

merge-delta:
	sbatch jobs/merge_delta_labels.sbatch

train-delta:
	sbatch jobs/train_delta_student.sbatch

eval-delta:
	sbatch jobs/evaluate_delta_student.sbatch
