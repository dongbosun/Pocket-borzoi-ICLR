# Project Storage Rules

This repository lives in `$HOME` and must stay code-first and lightweight. Keep source code, configs, jobs, docs, tests, and small metadata in the repo. Do not store generated datasets, teacher caches, model checkpoints, plots, benchmark outputs, SLURM logs, downloaded references, or cloned external model repos in `$HOME`.

Use this external storage root for all large or generated project artifacts:

```text
/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR
```

Canonical subdirectories:

```text
dataset      -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset
results      -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results
checkpoints  -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/checkpoints
interim      -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim
logs         -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/logs
plots_tables -> /extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/plots
```

Path conventions:

- Use `POCKET_BORZOI_STORAGE_ROOT` to override the external root when needed.
- Use `src/pocketreg/paths.py` helpers in Python scripts instead of hard-coding `data/`, `outputs/`, `external/`, `checkpoints/`, `logs/`, or `plots/`.
- Treat `*.pt`, `*.pth`, `*.ckpt`, `*.h5`, `*.hdf5`, `*.npz`, large `*.parquet`, downloaded FASTA/GTF files, teacher labels, CUDA caches, and run folders as external artifacts.
- SLURM scripts should write logs under the external `logs/slurm` tree and use external CUDA/cache paths.
- Before adding files to Git, check `git status --short` and avoid staging generated artifact directories.
- Never run broad `git add -A` or `git add .` unless `.gitignore` has just been checked and the status output shows only lightweight source files.
- If generated artifacts appear in the repo, move them to the external storage root and leave only code/config/docs in `$HOME`.

Heavy compute still follows the existing cluster rule: teacher inference, training, and large evaluation must run through SLURM, not on the login/head node.
