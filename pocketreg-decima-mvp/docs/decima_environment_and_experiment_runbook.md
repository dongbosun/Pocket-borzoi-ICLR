# Decima Environment And Experiment Runbook

Last updated: 2026-06-19, America/Los_Angeles

Project root:

```text
/home/dongbos/Pocket-borzoi-ICLR/pocketreg-decima-mvp
```

External artifact root:

```text
/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR
```

This is the operational runbook for the Decima mainline of Pocket-borzoi-ICLR. It records the exact local environments, storage layout, official Decima teacher setup, teacher-cache artifacts, SLURM workflow, and current experimental recommendations.

The main rule: keep the repo code-first and lightweight. Source code, configs, scripts, tests, and docs can live in `$HOME`; generated datasets, checkpoints, teacher caches, plots, logs, and large result folders must live under `/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR`.

## 1. What Is Safe To Commit

Commit to Git:

- `pocketreg-decima-mvp/src`
- `pocketreg-decima-mvp/scripts`
- `pocketreg-decima-mvp/configs`
- `pocketreg-decima-mvp/tests`
- `pocketreg-decima-mvp/docs`
- small metadata or hand-written summaries

Do not commit to Git:

- `*.pt`, `*.pth`, `*.ckpt`
- `*.h5`, `*.hdf5`
- `*.npz`
- large `*.parquet`
- downloaded FASTA/GTF/reference files
- teacher labels, teacher hidden-state caches, sequence caches
- CUDA/Hugging Face/genomepy caches
- SLURM logs
- run folders, plots, benchmark outputs

The repo-local Decima symlinks point into external storage:

```text
pocketreg-decima-mvp/data    -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/pocketreg-decima-mvp/data
pocketreg-decima-mvp/outputs -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/pocketreg-decima-mvp/outputs
```

Before staging anything:

```bash
cd /home/dongbos/Pocket-borzoi-ICLR
git status --short
```

Stage only explicit lightweight paths. Do not use broad `git add -A` or `git add .` in this project unless the status output has been reviewed and contains only safe source/doc files.

## 2. Canonical Paths

Use these variables in shell sessions:

```bash
export PROJECT_ROOT=/home/dongbos/Pocket-borzoi-ICLR/pocketreg-decima-mvp
export STORAGE_ROOT=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR
export POCKET_BORZOI_STORAGE_ROOT=/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR
export STUDENT_PY=/home/dongbos/miniconda3/envs/doge/bin/python
export OFFICIAL_PY=${PROJECT_ROOT}/outputs/official_decima/environment/decima_official_072/bin/python
cd "${PROJECT_ROOT}"
```

Canonical external subdirectories:

```text
dataset      -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset
results      -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results
checkpoints  -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/checkpoints
interim      -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim
logs         -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs
plots_tables -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/plots
```

For Python code shared with the larger repo, prefer `src/pocketreg/paths.py` helpers instead of hard-coding `data/`, `outputs/`, `checkpoints/`, `logs/`, or `plots/`. Inside the Decima MVP folder, existing scripts use `PROJECT_ROOT/data` and `PROJECT_ROOT/outputs`, which are symlinks into the external tree.

## 3. Environment Matrix

There are two important Python environments.

### 3.1 Student / Development Environment

Use this for student training, manifest manipulation, result summaries, and lightweight local analysis:

```text
/home/dongbos/miniconda3/envs/doge/bin/python
```

Observed version snapshot:

| Package | Version |
|---|---:|
| Python | 3.9.25 |
| torch | 2.1.0+cu118 |
| huggingface_hub | 0.36.2 |
| safetensors | 0.7.0 |
| anndata | 0.9.2 |
| h5py | 3.14.0 |
| scikit-learn | 1.6.1 |
| pyarrow | 21.0.0 |
| pandas | 2.3.3 |
| scipy | 1.13.1 |
| PyYAML | 6.0.3 |
| matplotlib | 3.9.4 |

Important negative facts:

- `decima` is not installed in `doge`.
- `grelu` is not installed in `doge`.
- `lightning` is not installed in `doge`.
- `zarr` was not present in the `doge` import probe.

Do not try to force official Decima into `doge`. Official teacher inference uses the isolated official environment below.

### 3.2 Official Decima Teacher Environment

Use this only for official Decima imports, official prediction sanity, hidden-state extraction, and hook discovery:

```text
/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/pocketreg-decima-mvp/outputs/official_decima/environment/decima_official_072
```

Official Python:

```text
/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/pocketreg-decima-mvp/outputs/official_decima/environment/decima_official_072/bin/python
```

Status files:

```text
pocketreg-decima-mvp/outputs/official_decima/environment/environment_status.json
pocketreg-decima-mvp/outputs/official_decima/environment/environment_status.md
pocketreg-decima-mvp/outputs/official_decima/environment/pip_freeze.txt
pocketreg-decima-mvp/outputs/official_decima/environment/install_log.txt
```

Observed official environment freeze includes:

| Package | Version |
|---|---:|
| torch | 2.1.0+cu118 |
| pytorch-lightning | 1.9.5 |
| torchmetrics | 1.8.2 |
| anndata | 0.9.2 |
| h5py | 3.14.0 |
| huggingface_hub | 0.36.2 |
| pandas | 2.3.3 |
| pyarrow | 21.0.0 |
| safetensors | 0.7.0 |
| scikit-learn | 1.6.1 |
| scipy | 1.13.1 |

The install log contains attempted metadata resolution for newer CUDA 12 packages, but the final freeze used for the successful run shows `torch==2.1.0+cu118`. Trust `pip_freeze.txt` and the status JSON over transient pip log lines.

Official Decima package source:

```text
git+https://github.com/Genentech/decima.git
commit 5e2439a63effe12b33e9b624dabf8b577a409d75
package decima==0.7.2
```

Official gReLU package source:

```text
git+https://github.com/Genentech/gReLU.git
package gReLU==1.1.0.post1.dev27
```

Cache paths for official jobs:

```bash
export XDG_CACHE_HOME=${STORAGE_ROOT}/interim/phase3_decima_official_cache/xdg
export HF_HOME=${STORAGE_ROOT}/interim/phase3_decima_official_cache/huggingface
export GENOMEPY_CACHE_DIR=${STORAGE_ROOT}/interim/phase3_decima_official_cache/genomepy
export MPLCONFIGDIR=/tmp/mpl_phase3_${SLURM_JOB_ID:-manual}
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${GENOMEPY_CACHE_DIR}" "${MPLCONFIGDIR}"
```

## 4. Rebuilding Or Checking The Official Environment

Setup/check script:

```text
pocketreg-decima-mvp/scripts/phase3_official_env_setup.py
```

The official environment was created by the Phase-3 J7A setup workflow. The script is conservative: it records imports, versions, checkpoint discovery, optional Hugging Face downloads, and optional isolated venv installation. It does not write into the student environment.

Typical status/check command:

```bash
cd /home/dongbos/Pocket-borzoi-ICLR/pocketreg-decima-mvp
/home/dongbos/miniconda3/envs/doge/bin/python scripts/phase3_official_env_setup.py \
  --project-root "${PROJECT_ROOT}" \
  --out-dir "${PROJECT_ROOT}/outputs/official_decima/environment" \
  --target-obs-idx 88
```

Only use the install/download flags when intentionally rebuilding:

```bash
/home/dongbos/miniconda3/envs/doge/bin/python scripts/phase3_official_env_setup.py \
  --project-root "${PROJECT_ROOT}" \
  --out-dir "${PROJECT_ROOT}/outputs/official_decima/environment" \
  --target-obs-idx 88 \
  --install-env \
  --attempt-download
```

Network installs/downloads may be slow. Keep the created venv and downloaded checkpoints under external storage, not under `$HOME`.

## 5. Official Teacher Checkpoints

Checkpoint snapshot directory:

```text
pocketreg-decima-mvp/outputs/official_decima/environment/hf_snapshot
```

The Decima official checkpoint files are stored through the `outputs` symlink, so their real location is under:

```text
/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/pocketreg-decima-mvp/outputs/official_decima/environment/hf_snapshot
```

Downloaded checkpoint inventory:

| File | Size bytes | SHA256 |
|---|---:|---|
| `rep0.safetensors` | 755,007,760 | `7b4a4ddd20f4c4ee8792459ee2f006576e255adc633c9347b6d3d8378f9173e0` |
| `rep1.safetensors` | 755,007,760 | `245bf6f4b4357f39ee15d8e38e32f79e6976ef64936a276d37bd03dbc266a7e3` |
| `rep2.safetensors` | 755,007,760 | `9200c7640fe6774b3a3c033c50009a4b002f439f6be10515c66bd727ad4328a0` |
| `rep3.safetensors` | 755,007,760 | `9a390d8be850879d8a533e6826dafa0329f6a4d26c612f280244ce7c19ee1992` |
| `rep0.ckpt` | 2,260,600,894 | checksum skipped because file is larger than the 2 GiB local checksum limit |
| `rep1.ckpt` | 2,260,600,894 | checksum skipped because file is larger than the 2 GiB local checksum limit |
| `rep2.ckpt` | 2,260,600,894 | checksum skipped because file is larger than the 2 GiB local checksum limit |
| `rep3.ckpt` | 2,260,600,894 | checksum skipped because file is larger than the 2 GiB local checksum limit |

Teacher parameter context:

- Official Decima teacher `rep0` parameter-like tensor count previously verified: `188,292,024`.
- Exported student target-only parameter count: `126,337`.
- One-rep teacher vs exported target-only student compression ratio: `188,292,024 / 126,337 = about 1490x`.
- Four-rep official ensemble vs exported target-only student approximate ratio: about `5960x`.

Use `rep0.safetensors` as the default teacher unless a new experiment explicitly studies the 4-rep ensemble.

## 6. Core Data And Reference Inputs

Metadata:

```text
pocketreg-decima-mvp/data/raw/decima/metadata.h5ad
```

Important metadata fact:

- Shape is `(8856, 18457)`.
- Available layers include `preds`, `v1_rep0`, `v1_rep1`, `v1_rep2`, `v1_rep3`.
- The metadata file did not expose a backed `X` observed-expression matrix for the sanity workflow.
- Current student metrics are therefore Decima-teacher distillation metrics, not direct measured-expression validation.

Reference genome:

```text
/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa
```

Main manifests:

```text
pocketreg-decima-mvp/data/processed/decima_v2_astro_full_128kb/manifest.parquet
pocketreg-decima-mvp/data/processed/decima_v2_astro_full_262kb/manifest.parquet
```

Sequence caches:

```text
pocketreg-decima-mvp/outputs/sequence_cache/obs88_full/context_131072
pocketreg-decima-mvp/outputs/sequence_cache/obs88_full/context_262144
```

## 7. SLURM Rules

Heavy compute must run through SLURM, not directly on the login/head node. This includes:

- official Decima teacher inference
- hidden-state extraction
- feature-cache generation
- full student training
- large evaluation

On the cluster, first load SLURM if commands are absent:

```bash
module load slurm/22.05.3
```

Common queue checks:

```bash
squeue -u "$USER"
sacct -j <job_ids> --format=JobIDRaw,State,ExitCode,Elapsed,Start,End -P -n
```

Use project SLURM scripts when available:

```bash
cd /home/dongbos/Pocket-borzoi-ICLR/pocketreg-decima-mvp
sbatch scripts/slurm_phase3_j7b_rep0_sanity.sh
sbatch scripts/slurm_phase3_j7e_feature_pilot_500.sh
sbatch scripts/slurm_phase3_j7f_feature_cache_5k.sh
```

SLURM logs are written through the project `outputs` symlink, for example:

```text
pocketreg-decima-mvp/outputs/slurm_logs/phase3_bio
pocketreg-decima-mvp/outputs/slurm_logs/phase3_bio/j7j_middle_screening_5k
```

These logs are generated artifacts and must not be committed.

## 8. Official Sanity Validation

Official sanity script:

```text
pocketreg-decima-mvp/scripts/official_decima_sanity.py
```

SLURM wrapper:

```text
pocketreg-decima-mvp/scripts/slurm_phase3_j7b_rep0_sanity.sh
```

Primary output:

```text
pocketreg-decima-mvp/outputs/official_decima/sanity/sanity_status.json
pocketreg-decima-mvp/outputs/official_decima/sanity/sanity_status.md
pocketreg-decima-mvp/outputs/official_decima/sanity/official_predictions_long.parquet
```

The successful sanity run used:

- official Decima API: `decima.tools.inference.predict_gene_expression(...)`
- model: `rep0.safetensors`
- target observation index: `88`
- gene sample: `99` genes returned from a request of `100`
- genome: local hg38 FASTA
- device: CUDA

Key sanity metrics for obs88:

| Compared layer | Pearson | Spearman | MAE | RMSE | Max abs diff |
|---|---:|---:|---:|---:|---:|
| `v1_rep0` | 0.999999991 | 1.000000000 | 0.000146 | 0.000190 | 0.000679 |
| `preds` | 0.983486309 | 0.977179963 | 0.255854 | 0.333684 | 0.793161 |

Interpretation:

- `rep0.safetensors` is validated as the local official teacher checkpoint.
- The `preds` layer is not identical to `v1_rep0`; it appears to be an aggregate or transformed prediction layer.
- For rep0 teacher distillation, compare against or derive labels from `v1_rep0` semantics.

## 9. Hook Discovery And Feature Stages

Hook discovery script:

```text
pocketreg-decima-mvp/scripts/discover_decima_feature_stages.py
```

Hook status files:

```text
pocketreg-decima-mvp/outputs/official_decima/hook_discovery/hook_discovery_status.json
pocketreg-decima-mvp/outputs/official_decima/hook_discovery/hook_discovery_status.md
pocketreg-decima-mvp/outputs/official_decima/hook_discovery/named_modules.txt
pocketreg-decima-mvp/outputs/official_decima/hook_discovery/selected_hook_stages.json
```

The feature-extraction script uses its `DEFAULT_STAGE_MODULES` mapping, and the completed 5k cache records the active hook modules in `feature_summary.json`. Treat the completed feature summary as the source of truth for extracted features.

Active semantic feature stages in the completed 5k cache:

| Stage | Meaning | Selected module |
|---|---|---|
| `E_conv6` | late convolutional tower | `model.embedding.conv_tower.blocks.6` |
| `M_transformer7` | late transformer tower | `model.embedding.transformer_tower.blocks.7` |
| `L_unet1` | U-Net tower | `model.embedding.unet_tower.blocks.1` |
| `P_prehead` | pre-head representation | `model.embedding.crop` |
| `T_task_prepool` | task-specific pre-pool representation | `model.head.channel_transform` |
| `O_final_pool` | final pooled head | `model.head.pool` |

The hook-discovery JSON is still useful for auditing model internals and candidates. Its exact low-level selected entries include:

```text
model.embedding.crop.layer
model.embedding.transformer_tower.blocks.7.ffn.dense2.act.layer
model.embedding.unet_tower.blocks.1.sconv.pointwise
model.head.channel_transform.act.layer
model.head.channel_transform.dropout.layer
model.head.pool.layer
```

If official Decima internals change, rerun hook discovery first, then update the extraction script's default mapping or pass through a new mapping before generating a new cache.

## 10. Feature Cache And Middle Targets

Feature extraction script:

```text
pocketreg-decima-mvp/scripts/extract_decima_hierarchical_features.py
```

Analysis script:

```text
pocketreg-decima-mvp/scripts/analyze_decima_feature_layers.py
```

Middle-target build script:

```text
pocketreg-decima-mvp/scripts/build_hierarchical_middle_targets.py
```

5k feature cache:

```text
pocketreg-decima-mvp/outputs/teacher_cache/phase3_decima_hierarchy/feature_cache_5k/compressed_features.parquet
pocketreg-decima-mvp/outputs/teacher_cache/phase3_decima_hierarchy/feature_cache_5k/feature_summary.json
```

5k cache shape:

- rows: `5000`
- total columns: `18232`
- feature columns: `18228`
- stage dimensions:
  - `E_conv6`: `3072`
  - `M_transformer7`: `8192`
  - `L_unet1`: `3072`
  - `P_prehead`: `3840`
  - `T_task_prepool`: `26`
  - `O_final_pool`: `26`

Layer analysis:

```text
pocketreg-decima-mvp/outputs/teacher_cache/phase3_decima_hierarchy/layer_analysis/layer_analysis_status.json
```

Most useful probe results from the 5k layer analysis:

| Stage | Test Pearson | Test RMSE | Interpretation |
|---|---:|---:|---|
| `P_prehead` | 0.959266 | 0.366934 | strongest compact teacher signal |
| `L_unet1` | 0.941760 | 0.460789 | strong late representation |
| `T_task_prepool` | 0.896247 | 0.603318 | strong task-head signal |
| `M_transformer7` | 0.707289 | 0.954848 | useful but weaker alone |
| `E_conv6` | 0.524027 | 1.144066 | weak alone |

Middle targets:

```text
pocketreg-decima-mvp/outputs/teacher_cache/phase3_decima_hierarchy/middle_targets_5k/manifest_with_middle_targets_5k.parquet
pocketreg-decima-mvp/outputs/teacher_cache/phase3_decima_hierarchy/middle_targets_5k/middle_targets.parquet
pocketreg-decima-mvp/outputs/teacher_cache/phase3_decima_hierarchy/middle_targets_5k/middle_projection_train_only.npz
pocketreg-decima-mvp/outputs/teacher_cache/phase3_decima_hierarchy/middle_targets_5k/summary.json
```

Middle-target build facts:

- rows: `5000`
- split counts:
  - train: `2164`
  - validation: `1170`
  - test: `1666`
- input feature columns: `18228`
- output middle components: `64`
- projection fit rule: train split only
- explained variance ratio sum: `0.6206168532371521`

Do not fit scalers/PCA on validation or test genes. Any new full-data middle-target build must preserve the train-only projection rule.

## 11. Student Training Interface

Student training uses:

```text
pocketreg-decima-mvp/scripts/train_decima_student_v2.py
```

Export target-only checkpoint:

```text
pocketreg-decima-mvp/scripts/export_v2_target_only.py
```

Default student Python:

```text
/home/dongbos/miniconda3/envs/doge/bin/python
```

Phase-3 student interface smoke output:

```text
pocketreg-decima-mvp/outputs/teacher_cache/phase3_decima_hierarchy/student_interface/j7i_smoke.json
```

Smoke-test facts:

- manifest rows: `5000`
- constructed label width: `116`
- label blocks:
  - final: `1`
  - replicate: `4`
  - auxiliary: `8`
  - residual: `1`
  - middle: `64`
- `tcn_small` model with middle head: `515,214` trainable parameters
- exported target-only checkpoint: `126,337` parameters

The reported `126,337` parameter count refers to the exported target-only student used for compression comparisons. Training-time models can have additional auxiliary/middle heads.

## 12. J7J Middle-Distillation Screening

Submission script:

```text
pocketreg-decima-mvp/scripts/submit_phase3_j7j_middle_screening.py
```

Default command:

```bash
cd /home/dongbos/Pocket-borzoi-ICLR/pocketreg-decima-mvp
module load slurm/22.05.3
/home/dongbos/miniconda3/envs/doge/bin/python scripts/submit_phase3_j7j_middle_screening.py
```

Useful dry-run command:

```bash
/home/dongbos/miniconda3/envs/doge/bin/python scripts/submit_phase3_j7j_middle_screening.py --dry-run
```

Default inputs:

```text
base config:
pocketreg-decima-mvp/configs/next_jobs/context_scaling_obs88_128kb_seed1.yaml

middle-target manifest:
pocketreg-decima-mvp/outputs/teacher_cache/phase3_decima_hierarchy/middle_targets_5k/manifest_with_middle_targets_5k.parquet

run root:
pocketreg-decima-mvp/outputs/runs/phase3_middle_screening_5k

report root:
pocketreg-decima-mvp/outputs/reports/phase3_biological_grounding_update/j7j_middle_screening_5k
```

Submitted SLURM ids:

| Variant | SLURM id |
|---|---:|
| `H0_output_only_seed1` | 238983 |
| `H1_mid005_seed1` | 238984 |
| `H2_mid010_seed1` | 238985 |
| `H3_mid020_seed1` | 238986 |
| `H4_rep_mid010_seed1` | 238987 |
| `H5_a6_mid005_seed1` | 238988 |
| `H6_a6_mid010_seed1` | 238989 |
| `H7_a6_mid020_seed1` | 238990 |

All 8 runs completed and exported `checkpoints/best_target_only.pt`.

J7J results:

| Run | Loss recipe | Test Pearson | Test Spearman | Test R2 | Test RMSE | Test MAE | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| H0 | final only | 0.617538 | 0.584207 | 0.345448 | 1.031606 | 0.820101 | 8 |
| H1 | final + mid 0.05 | 0.643932 | 0.610099 | 0.410777 | 0.978773 | 0.794779 | 9 |
| H2 | final + mid 0.10 | 0.625141 | 0.591382 | 0.374312 | 1.008604 | n/a | n/a |
| H3 | final + mid 0.20 | 0.634415 | 0.603084 | 0.360449 | 1.019717 | n/a | n/a |
| H4 | final + rep 0.25 + mid 0.10 | 0.581837 | 0.549283 | 0.330088 | 1.043640 | n/a | n/a |
| H5 | A6 + mid 0.05 | 0.631071 | 0.599613 | 0.282690 | 1.079930 | n/a | n/a |
| H6 | A6 + mid 0.10 | 0.651357 | 0.623215 | 0.176409 | 1.157171 | n/a | n/a |
| H7 | A6 + mid 0.20 | 0.600544 | 0.573379 | 0.304886 | 1.063090 | n/a | n/a |

Interpretation:

- H1 is the best balanced J7J setting.
- H1 improved over H0 in Pearson, Spearman, R2, RMSE, and MAE.
- H6 had the highest Pearson, but calibration was poor: low R2 and high RMSE.
- Adding replicate/aux/residual losses to middle loss did not clearly help in the 5k screen.
- This is a 5k screening result, not the final full-data result.

Recommended scaled recipe from J7J:

```text
final_loss_weight = 1.0
rep_loss_weight = 0.0
aux_loss_weight = 0.0
residual_loss_weight = 0.0
mid_loss_weight = 0.05
middle targets = 64 train-only PCA components
teacher = official Decima rep0
```

## 13. Previous P0-P5 Results To Keep In Mind

The best Decima student before Phase-3 middle-target work remains the full-data 262 kb A6 run:

| Run | Context | Params | Test Pearson | Test Spearman | Test R2 | Test RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `262kb_A6_seed1` | 262 kb | 126,337 | 0.720245 | 0.689016 | 0.507734 | 0.890853 |
| 128 kb baseline seed1 | 128 kb | 126,337 | 0.694707 | 0.665485 | 0.458860 | 0.934030 |

P0-P5 conclusions:

- 262 kb context plus A6 is the current best full-data student result.
- A2/A6 behavior is context-dependent.
- Seed variability matters; future headline comparisons need multi-seed runs.
- Cross-target generalization is weak for several non-astrocyte targets.
- The earlier P5 capacity-scaling attempt was inconclusive because the exported target-only architecture stayed at `126,337` parameters.

## 14. Recommended Next Experiments

Run these in this order.

### 14.1 Full-Data 128 kb H1-Style Middle Distillation

Goal: test whether the J7J H1 middle-loss improvement survives beyond the 5k subset.

Recipe:

```text
context = 128 kb
target = obs88
teacher = rep0.safetensors
middle targets = train-only PCA, 64 components
losses = final + mid 0.05
seeds = 1, 2, 3
```

Compare against:

- existing 128 kb seed1 baseline
- existing 128 kb seed2/seed3 P1 results
- H0 output-only screen result only as a small-subset sanity reference

Success criterion:

- mean Pearson improves over matching 128 kb final-only/A6 baseline
- R2/RMSE do not degrade
- exported target-only checkpoint remains `126,337` params unless intentionally changing architecture

### 14.2 Full-Data 262 kb H1-Style Middle Distillation

Goal: combine the strongest previous context result with the best balanced middle-loss setting.

Recipe:

```text
context = 262 kb
target = obs88
teacher = rep0.safetensors
middle targets = train-only PCA, 64 components
losses = final + mid 0.05
seeds = 1, 2, 3
```

Compare against:

```text
pocketreg-decima-mvp/outputs/runs/p0_context262_obs88/262kb_A6_seed1/checkpoints/best_target_only.pt
```

This is the most important next experiment if GPU time is limited.

### 14.3 H6 Calibration Diagnostic

Goal: understand why H6 reached the highest Pearson but poor R2/RMSE.

Diagnostic checks:

- prediction scatter slope/intercept
- target scale drift
- per-split residual distributions
- whether auxiliary/residual losses create a scale mismatch after exporting target-only
- whether a simple validation-set calibration layer would recover R2/RMSE

Do not promote H6 as the default recipe until calibration is understood.

### 14.4 Middle Targets For Additional Representative Targets

Goal: address P3 cross-target weakness.

Targets from prior P3:

```text
obs1088: CNS-like, moderate performance
obs6107: macrophage/heart, weak
obs4248: capillary endothelial/heart, weak
obs4489: classical monocyte/blood/dengue, very weak
```

Recommended strategy:

- start with H1-style middle targets on `obs1088`
- then test one weak heart/blood target
- use small screening before scaling all targets

## 15. How To Read Completion Status

For J7J runs, a run is complete when both files exist:

```text
<run_dir>/metrics.json
<run_dir>/checkpoints/best_target_only.pt
```

For official environment:

```text
pocketreg-decima-mvp/outputs/official_decima/environment/environment_status.json
```

Expected status:

```json
{"status": "completed"}
```

For official sanity:

```text
pocketreg-decima-mvp/outputs/official_decima/sanity/sanity_status.json
```

Expected status:

```json
{"status": "completed"}
```

For middle target build:

```text
pocketreg-decima-mvp/outputs/teacher_cache/phase3_decima_hierarchy/middle_targets_5k/summary.json
```

Expected status:

```json
{"status": "completed"}
```

## 16. Common Failure Modes

`sbatch` or `squeue` not found:

- load the SLURM module: `module load slurm/22.05.3`
- this can happen in desktop/container shells even when the cluster supports SLURM

Official Decima import missing:

- do not install into `doge`
- use `${OFFICIAL_PY}`
- inspect `outputs/official_decima/environment/environment_status.json`

Hugging Face or genome cache writes into `$HOME`:

- set `HF_HOME`, `XDG_CACHE_HOME`, and `GENOMEPY_CACHE_DIR` to external storage before running official jobs

Matplotlib writes into `$HOME`:

- set `MPLCONFIGDIR=/tmp/mpl_phase3_${SLURM_JOB_ID:-manual}`

Middle target leakage risk:

- fit all normalizers/projections on train split only
- never fit PCA/scalers on val/test rows

Confusing official Decima layers:

- `v1_rep0` is the near-exact match for local `rep0.safetensors`
- `preds` is close but not identical and should not be treated as raw `rep0`

Result overclaiming:

- current J7J is a 5k screen
- the current best full-data run is still `262kb_A6_seed1`
- all metrics described here are Decima-teacher distillation metrics unless explicitly stated otherwise

## 17. Minimum Preflight Before Any New Decima Experiment

Run these checks:

```bash
cd /home/dongbos/Pocket-borzoi-ICLR/pocketreg-decima-mvp
test -x /home/dongbos/miniconda3/envs/doge/bin/python
test -x "${PROJECT_ROOT}/outputs/official_decima/environment/decima_official_072/bin/python"
test -f "${PROJECT_ROOT}/outputs/official_decima/environment/hf_snapshot/rep0.safetensors"
test -f "${PROJECT_ROOT}/outputs/official_decima/sanity/sanity_status.json"
test -f "${PROJECT_ROOT}/data/raw/decima/metadata.h5ad"
test -f "${STORAGE_ROOT}/dataset/external/reference/hg38/hg38.fa"
```

If using middle targets, also check:

```bash
test -f "${PROJECT_ROOT}/outputs/teacher_cache/phase3_decima_hierarchy/middle_targets_5k/manifest_with_middle_targets_5k.parquet"
test -f "${PROJECT_ROOT}/outputs/teacher_cache/phase3_decima_hierarchy/middle_targets_5k/middle_projection_train_only.npz"
```

For clean Git hygiene:

```bash
cd /home/dongbos/Pocket-borzoi-ICLR
git status --short
```

Only stage source/docs/configs/tests/scripts that are intentionally part of the experiment handoff.

## 18. Handoff Summary

Current verified state:

- `doge` is the student/dev environment.
- official Decima is isolated in external storage at `decima_official_072`.
- official `rep0.safetensors` is validated against `v1_rep0` with near-perfect agreement.
- official hidden-state extraction is unblocked and has produced a real 5k hierarchical feature cache.
- 64-dimensional train-only PCA middle targets are built and usable by the student.
- J7J 5k middle-distillation screening completed for H0-H7.
- H1, `final + mid 0.05`, is the best balanced middle-loss setting from the screen.
- Full-data 262 kb A6 remains the best overall completed student result until the H1 middle recipe is scaled.

Most likely next best use of GPU time:

```text
Run full-data 262 kb obs88 with official rep0 middle targets and H1 loss:
final_loss_weight=1.0, mid_loss_weight=0.05, no rep/aux/residual losses, seeds 1/2/3.
```

Keep all new heavy artifacts under:

```text
/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR
```

and keep this repo for the code, configs, docs, scripts, and small metadata that make the experiments reproducible.
