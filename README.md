# pocketreg-borzoi-mvp

This repo is an MVP scaffold for distilling one Borzoi / Mini-Borzoi K562 RNA-seq endpoint into small local DNA-sequence student models.

Current delivery status: Phase 0 and Phase 1 only. Real Borzoi teacher inference, real student training, cache merging, and full evaluation are intentionally guarded placeholders for later phases.

## What This Repo Does

Week 2 target:

`S_track(sequence around gene g) ~= q(Borzoi_K562(sequence around gene g))`

where `q` defaults to `log1p(mean selected K562 RNA-seq coverage over gene body bins)`.

Week 3 target:

`S_delta(ref_seq, alt_seq, gene metadata) ~= q(Borzoi_K562(x_alt)) - q(Borzoi_K562(x_ref))`

Phase 0/1 currently provides:

- repo/package skeleton
- cluster safety guard
- toy CPU smoke pipeline
- unit tests
- Borzoi asset config/setup scaffolding
- Borzoi/K562 asset inspection
- official processed-data inspection
- K562 gene manifest builder
- SLURM job templates for later phases

## What This Repo Does Not Do

- no full Borzoi training
- no full multi-track distillation
- no all-tissue model
- no real personal-genome inference
- no clinical phenotype prediction
- no Decima implementation
- no AlphaGenome implementation
- no full multi-TB Borzoi training data download

## Cluster Rule

Heavy jobs must run through SLURM, not on a login/head node. The guard raises this exact error:

```text
Refusing to run heavy task {task_name} on login/head node. Submit via sbatch or pass --allow-local only for debugging.
```

Heavy scripts call `assert_compute_context(...)`. During Phase 0/1, the future heavy scripts are placeholders, but they already enforce the guard.

## Install

```bash
conda create -n pocketreg_borzoi python=3.10
conda activate pocketreg_borzoi
pip install -r requirements.txt
pip install -e .
```

For Phase 0 toy smoke tests, only the light dependencies are needed: NumPy, PyYAML, matplotlib, and psutil. Real Borzoi and student training later require PyTorch, TensorFlow 2.15.x, pandas/pyarrow, pyfaidx, and the official Calico repos.

## Official Assets

Create a local asset config from explicit paths:

```bash
python scripts/setup_borzoi_assets.py \
  --work-dir external \
  --k562-fold0-weights /path/to/fold0 \
  --k562-fold1-weights /path/to/fold1 \
  --k562-targets /path/to/targets.txt \
  --k562-params /path/to/params.json \
  --hg38-fasta /path/to/hg38.fa \
  --gencode-gtf /path/to/gencode.gtf.gz \
  --out-config configs/borzoi_assets.local.yaml
```

Clone official repos when needed:

```bash
python scripts/setup_borzoi_assets.py \
  --work-dir external \
  --download-official-repos \
  --out-config configs/borzoi_assets.local.yaml
```

Inspect K562 metadata:

```bash
python scripts/inspect_borzoi_assets.py \
  --assets-config configs/borzoi_assets.local.yaml \
  --out /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/borzoi_assets_inspection
```

Inspect already processed official K562 data:

```bash
python scripts/inspect_borzoi_processed_data.py \
  --data-dir /path/to/borzoi/tutorials/latest/make_data_or_processed \
  --out /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/processed_k562_inspection
```

## Phase 0 Smoke

The toy smoke path does not need Borzoi, TensorFlow, hg38, GPU, SLURM, pandas, or PyTorch.

```bash
python scripts/run_smoke_test.py
```

It creates toy FASTA/genes, fake motif teacher labels, fake delta labels, toy predictions, plots, and metrics under `/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/runs/toy_smoke`.

## Phase 1 Manifest

Build a real gene manifest after assets have been inspected:

```bash
python scripts/build_k562_gene_manifest.py \
  --assets-config configs/borzoi_assets.local.yaml \
  --fasta /path/to/hg38.fa \
  --gtf /path/to/gencode.vXX.annotation.gtf.gz \
  --out /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_gene_manifest.parquet \
  --input-len auto \
  --output-num-bins auto \
  --bin-size auto \
  --target-index auto \
  --aggregation gene_body_log1p_mean \
  --max-genes 5000
```

If model dimensions cannot be detected from params, pass `--input-len`, `--output-num-bins`, and `--bin-size` explicitly. This repo intentionally does not hard-code full-Borzoi dimensions.

## Later Week 2 Commands

These commands are documented now, but the implementation is reserved for Phase 2/3.

```bash
python scripts/make_teacher_ref_shards.py ...
python scripts/submit_sbatch.py --job jobs/teacher_ref_k562_array.sbatch --array 0-31 --export ...
sbatch jobs/merge_ref_labels.sbatch
sbatch jobs/train_track_student.sbatch
sbatch jobs/evaluate_track_student.sbatch
sbatch jobs/benchmark_student.sbatch
```

## Later Week 3 Commands

Synthetic SNV generation is available as a light utility. Teacher delta inference and delta training are later-phase placeholders.

```bash
sbatch jobs/generate_snvs.sbatch
python scripts/make_teacher_delta_shards.py ...
python scripts/submit_sbatch.py --job jobs/teacher_delta_k562_array.sbatch --array 0-63 --export ...
sbatch jobs/merge_delta_labels.sbatch
sbatch jobs/train_delta_student.sbatch
sbatch jobs/evaluate_delta_student.sbatch
```

## Output Files

- manifests: `/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/`
- synthetic variants: `/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/variants/`
- teacher cache: `/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/`
- run checkpoints/metrics/plots: `/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/runs/`
- reports: `/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/`
- benchmarks: `/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/benchmarks/`
- SLURM logs: `/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm/`

## Known Caveats

- K562 tutorial / Mini-Borzoi models may have different input and output lengths from full Borzoi.
- Teacher pseudo-labels are not real coverage labels.
- Processed K562 coverage-label mode is optional and must be inspected before use.
- Synthetic SNP delta is teacher fidelity only, not biological validation.
- Later stages need eQTL or MPRA validation.
