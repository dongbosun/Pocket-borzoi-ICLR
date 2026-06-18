PYTHON ?= /home/dongbos/miniconda3/envs/doge/bin/python
PIP ?= $(PYTHON) -m pip
SBATCH ?= /pkg/slurm/22.05.3/bin/sbatch
PARTITION ?= zhanglab.p
RUN_NAME ?= decima_mvp_100k
DEVICE ?= cpu
OUT ?= outputs/reports/benchmark_$(DEVICE).json

.PHONY: install test test-sbatch smoke smoke-local inspect manifest train100k bench eval

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

test-sbatch:
	$(SBATCH) -p $(PARTITION) --job-name=pocketreg-tests --cpus-per-task=2 --mem=8G --time=00:10:00 --output=outputs/reports/slurm_tests_%j.out --error=outputs/reports/slurm_tests_%j.err --wrap='$(PYTHON) -m pytest -q'

smoke:
	$(SBATCH) -p $(PARTITION) --export=ALL,PYTHON=$(PYTHON) scripts/slurm_smoke_test.sbatch

smoke-local:
	$(PYTHON) scripts/run_smoke_test.py

inspect:
	$(PYTHON) scripts/inspect_decima_data.py --adata $(ADATA) --out outputs/reports/decima_inspection

manifest:
	$(PYTHON) scripts/build_decima_manifest.py \
		--adata $(ADATA) \
		--fasta $(FASTA) \
		--out data/processed/decima_mvp/manifest.parquet \
		--context-len 65536 \
		--label-layer preds \
		--split-mode chromosome \
		--max-genes 5000

train100k:
	$(SBATCH) -p $(PARTITION) --export=ALL,PYTHON=$(PYTHON),MANIFEST=$(MANIFEST),FASTA=$(FASTA),RUN_NAME=$(RUN_NAME) scripts/slurm_train100k.sbatch

eval:
	$(SBATCH) -p $(PARTITION) --export=ALL,PYTHON=$(PYTHON),CHECKPOINT=$(CHECKPOINT),MANIFEST=$(MANIFEST),FASTA=$(FASTA),DEVICE=$(DEVICE),OUT=$(OUT) scripts/slurm_evaluate.sbatch

bench:
	$(SBATCH) -p $(PARTITION) --export=ALL,PYTHON=$(PYTHON),CHECKPOINT=$(CHECKPOINT),MANIFEST=$(MANIFEST),FASTA=$(FASTA),DEVICE=$(DEVICE),OUT=$(OUT) scripts/slurm_benchmark.sbatch
