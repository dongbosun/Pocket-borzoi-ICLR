# SLURM Jobs

These scripts are templates. Edit account/partition/conda paths for the local cluster.

All heavy jobs write logs under `/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/slurm` and call scripts that refuse
login/head-node execution unless running under SLURM, `--toy`, `--dry-run`, or
an explicit `--allow-local` debug flag.

Submit through the helper when possible:

```bash
python scripts/submit_sbatch.py \
  --job jobs/teacher_ref_k562_array.sbatch \
  --array 0-31 \
  --export ALL,REPO_ROOT=$PWD,CONFIG=configs/borzoi_k562_track_100k.yaml,SHARD_DIR=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/processed/shards/ref_k562,CACHE_DIR=/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/ref_k562,FASTA=/path/to/hg38.fa,ASSETS_CONFIG=configs/borzoi_assets.local.yaml,BATCH_SIZE=4 \
  --dry-run
```
