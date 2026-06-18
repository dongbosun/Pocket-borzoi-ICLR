"""Build and consume gene-centered Decima distillation manifests."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pocketreg.data.anndata_utils import (
    get_teacher_labels,
    make_json_safe,
    row_to_metadata,
    summarize_labels,
)
from pocketreg.data.fasta import FastaReader
from pocketreg.data.sequence import one_hot_encode, reverse_complement
from pocketreg.data.splits import (
    assert_no_chromosome_overlap,
    assign_chromosome_splits,
    chrom_key,
    split_counts_by_chromosome,
)

LOGGER = logging.getLogger(__name__)
REQUIRED_TARGET_COLS = {
    "target_cell_type": "cell_type",
    "target_tissue": "tissue",
    "target_organ": "organ",
    "target_disease": "disease",
    "target_region": "region",
    "target_subregion": "subregion",
    "target_celltype_coarse": "celltype_coarse",
}


def _numeric_col(var: pd.DataFrame, col: str) -> pd.Series:
    if col not in var:
        return pd.Series(np.nan, index=var.index, dtype="float64")
    return pd.to_numeric(var[col], errors="coerce")


def _string_col(var: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in var:
        return pd.Series(default, index=var.index, dtype="object")
    return var[col].astype("object")


def _compute_tss(var: pd.DataFrame) -> pd.Series:
    start = _numeric_col(var, "start")
    end = _numeric_col(var, "end")
    gene_start = _numeric_col(var, "gene_start").combine_first(start)
    gene_end = _numeric_col(var, "gene_end").combine_first(end)
    canonical = _numeric_col(var, "ensembl_canonical_tss")
    strand = _string_col(var, "strand").astype(str)

    tss = canonical.copy()
    missing = ~np.isfinite(tss)
    plus = missing & (strand == "+")
    minus = missing & (strand == "-")
    tss.loc[plus] = gene_start.loc[plus]
    tss.loc[minus] = gene_end.loc[minus]
    remaining = ~np.isfinite(tss)
    midpoint = (gene_start + gene_end) / 2.0
    tss.loc[remaining] = midpoint.loc[remaining]
    return tss


def _target_name(adata: Any, pseudobulk_idx: int) -> str:
    row = adata.obs.iloc[pseudobulk_idx]
    for col in ("target_name", "name", "sample", "sample_id", "cell_type"):
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return str(adata.obs.index[pseudobulk_idx])


def _assign_decima_dataset_splits(var: pd.DataFrame) -> pd.Series:
    if "dataset" not in var:
        raise ValueError(
            "split-mode decima_dataset requires adata.var['dataset']; use --split-mode chromosome."
        )
    labels = var["dataset"].astype(str).str.casefold()
    out = pd.Series(index=var.index, dtype="object")
    out[labels.str.contains("train", na=False)] = "train"
    out[labels.str.contains("val|valid", na=False, regex=True)] = "val"
    out[labels.str.contains("test", na=False)] = "test"
    if not {"train", "val", "test"}.issubset(set(out.dropna())):
        raise ValueError(
            "adata.var['dataset'] did not contain usable train/val/test labels; "
            "use --split-mode chromosome instead."
        )
    return out


def _assign_fold_splits(var: pd.DataFrame, val_fold: int = 0, test_fold: int = 1) -> pd.Series:
    if "fold" not in var:
        raise ValueError("split-mode fold requires adata.var['fold']; use --split-mode chromosome.")
    fold = pd.to_numeric(var["fold"], errors="coerce")
    out = pd.Series("train", index=var.index, dtype="object")
    out[fold == val_fold] = "val"
    out[fold == test_fold] = "test"
    return out


def _stratified_sample(df: pd.DataFrame, max_genes: int, seed: int) -> pd.DataFrame:
    if max_genes is None or max_genes <= 0 or len(df) <= max_genes:
        return df
    parts = []
    remaining = max_genes
    rng = np.random.default_rng(seed)
    split_sizes = df["split"].value_counts().to_dict()
    for split, group in df.groupby("split", sort=False):
        if remaining <= 0:
            break
        target = max(1, int(round(max_genes * split_sizes[split] / len(df))))
        target = min(target, len(group), remaining)
        indices = rng.choice(group.index.to_numpy(), size=target, replace=False)
        parts.append(group.loc[indices])
        remaining -= target
    sampled = pd.concat(parts, axis=0)
    if len(sampled) > max_genes:
        sampled = sampled.sample(n=max_genes, random_state=seed)
    return sampled.sort_values("gene_idx").reset_index(drop=True)


def validate_manifest_fasta(
    manifest: pd.DataFrame, fasta_path: str | Path, context_len: int, n_examples: int = 10
) -> pd.DataFrame:
    """Add fasta_chrom and verify that sample windows return the requested length."""
    reader = FastaReader(fasta_path)
    manifest = manifest.copy()
    manifest["fasta_chrom"] = manifest["chrom"].map(reader.normalize_chrom)
    for _, row in manifest.head(n_examples).iterrows():
        seq = reader.fetch(row["fasta_chrom"], int(row["seq_start"]), int(row["seq_end"]), pad=True)
        if len(seq) != context_len:
            raise ValueError(
                f"FASTA validation failed for {row['gene_id']}: got length {len(seq)}, "
                f"expected {context_len}."
            )
    return manifest


def build_manifest_from_anndata(
    adata: Any,
    pseudobulk_idx: int,
    label_layer: str,
    context_len: int,
    *,
    split_mode: str = "chromosome",
    include_sex_chromosomes: bool = False,
    all_gene_types: bool = False,
    max_frac_n: float = 0.05,
    max_genes: int | None = None,
    seed: int = 13,
    coordinate_convention: str = "zero_based",
    fasta_path: str | Path | None = None,
    skip_fasta_check: bool = True,
    val_fold: int = 0,
    test_fold: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Build a one-row-per-gene manifest from Decima AnnData metadata."""
    if coordinate_convention not in {"zero_based", "one_based"}:
        raise ValueError("coordinate_convention must be 'zero_based' or 'one_based'.")
    if context_len <= 0:
        raise ValueError("context_len must be positive.")

    labels = get_teacher_labels(adata, pseudobulk_idx, label_layer)
    var = adata.var.copy()
    has_gene_type = "gene_type" in var
    gene_id = _string_col(var, "gene_id", default="").astype(str)
    gene_id = gene_id.where(gene_id.ne(""), pd.Series(var.index.astype(str), index=var.index))
    if "gene_name" in var:
        gene_name = var["gene_name"].astype(str)
    elif "symbol" in var:
        gene_name = var["symbol"].astype(str)
    else:
        gene_name = gene_id

    tss = _compute_tss(var)
    start = _numeric_col(var, "start")
    end = _numeric_col(var, "end")
    gene_start = _numeric_col(var, "gene_start").combine_first(start)
    gene_end = _numeric_col(var, "gene_end").combine_first(end)
    if coordinate_convention == "one_based":
        for series in (tss, start, end, gene_start, gene_end):
            series -= 1

    manifest = pd.DataFrame(
        {
            "gene_idx": np.arange(adata.n_vars, dtype=np.int64),
            "gene_id": gene_id.to_numpy(),
            "gene_name": gene_name.to_numpy(),
            "chrom": _string_col(var, "chrom").astype(str).to_numpy(),
            "start": start.to_numpy(),
            "end": end.to_numpy(),
            "gene_start": gene_start.to_numpy(),
            "gene_end": gene_end.to_numpy(),
            "strand": _string_col(var, "strand").astype(str).to_numpy(),
            "tss": tss.to_numpy(),
            "context_len": int(context_len),
            "label_layer": label_layer,
            "y_teacher": labels,
            "target_obs_idx": int(pseudobulk_idx),
            "target_name": _target_name(adata, pseudobulk_idx),
            "gene_type": _string_col(var, "gene_type").astype(str).to_numpy(),
            "gene_length": _numeric_col(var, "gene_length").to_numpy(),
            "frac_N": _numeric_col(var, "frac_N").to_numpy(),
            "frac_nan": _numeric_col(var, "frac_nan").to_numpy(),
            "mean_counts": _numeric_col(var, "mean_counts").to_numpy(),
            "n_tracks": _numeric_col(var, "n_tracks").to_numpy(),
            "decima_var_dataset": _string_col(var, "dataset").to_numpy(),
            "decima_var_fold": _numeric_col(var, "fold").to_numpy(),
            "pearson": _numeric_col(var, "pearson").to_numpy(),
            "size_factor_pearson": _numeric_col(var, "size_factor_pearson").to_numpy(),
        }
    )
    obs_row = adata.obs.iloc[pseudobulk_idx]
    for target_col, obs_col in REQUIRED_TARGET_COLS.items():
        manifest[target_col] = make_json_safe(obs_row[obs_col]) if obs_col in obs_row else None

    manifest["chrom_key"] = manifest["chrom"].map(chrom_key)
    allowed = {str(i) for i in range(1, 23)}
    if include_sex_chromosomes:
        allowed |= {"X", "Y"}
    keep = manifest["chrom_key"].isin(allowed)
    keep &= np.isfinite(manifest["tss"])
    keep &= np.isfinite(manifest["y_teacher"])
    if not all_gene_types and has_gene_type:
        keep &= manifest["gene_type"].astype(str).str.casefold().eq("protein_coding")
    if "frac_N" in manifest:
        keep &= manifest["frac_N"].isna() | (manifest["frac_N"] <= max_frac_n)
    manifest = manifest.loc[keep].copy()

    manifest["tss"] = np.floor(manifest["tss"]).astype(np.int64)
    manifest["seq_start"] = manifest["tss"] - int(context_len // 2)
    manifest["seq_end"] = manifest["seq_start"] + int(context_len)

    if split_mode == "chromosome":
        manifest["split"] = assign_chromosome_splits(
            manifest["chrom"], include_sex_chromosomes=include_sex_chromosomes
        )
        manifest = manifest.dropna(subset=["split"]).copy()
        assert_no_chromosome_overlap(manifest)
    elif split_mode == "decima_dataset":
        split_series = _assign_decima_dataset_splits(var)
        manifest["split"] = split_series.iloc[manifest["gene_idx"].to_numpy()].to_numpy()
        manifest = manifest.dropna(subset=["split"]).copy()
    elif split_mode == "fold":
        split_series = _assign_fold_splits(var, val_fold=val_fold, test_fold=test_fold)
        manifest["split"] = split_series.iloc[manifest["gene_idx"].to_numpy()].to_numpy()
    else:
        raise ValueError("split_mode must be chromosome, decima_dataset, or fold.")

    if manifest.empty:
        raise ValueError(
            "Manifest is empty after filtering. Relax gene filters, check chromosome names, "
            "or inspect missing teacher labels."
        )

    manifest = _stratified_sample(manifest.reset_index(drop=True), max_genes, seed)
    if fasta_path is not None and not skip_fasta_check:
        manifest = validate_manifest_fasta(manifest, fasta_path, context_len)
    elif "fasta_chrom" not in manifest:
        manifest["fasta_chrom"] = manifest["chrom"]

    target_metadata = row_to_metadata(adata.obs.iloc[pseudobulk_idx], pseudobulk_idx)
    summary = summarize_manifest(manifest, labels, target_metadata)
    return manifest.reset_index(drop=True), target_metadata, summary


def summarize_manifest(
    manifest: pd.DataFrame, labels: np.ndarray, target_metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return a JSON-safe summary for a manifest."""
    split_counts = manifest["split"].value_counts().sort_index().to_dict()
    chrom_counts = split_counts_by_chromosome(manifest).to_dict(orient="records")
    return {
        "n_genes": int(len(manifest)),
        "context_len": int(manifest["context_len"].iloc[0]) if len(manifest) else None,
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "split_counts_by_chromosome": [
            {str(k): make_json_safe(v) for k, v in row.items()} for row in chrom_counts
        ],
        "target_metadata": target_metadata or {},
        "teacher_label_stats_all_vars": summarize_labels(labels),
        "teacher_label_stats_manifest": summarize_labels(manifest["y_teacher"].to_numpy()),
    }


def save_manifest_outputs(
    manifest: pd.DataFrame,
    out_path: str | Path,
    target_metadata: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Save manifest parquet/csv and sidecar JSON summaries."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out_path, index=False)
    manifest.to_csv(out_path.with_suffix(".csv.gz"), index=False)
    with (out_path.parent / "target_metadata.json").open("w") as f:
        json.dump({k: make_json_safe(v) for k, v in target_metadata.items()}, f, indent=2)
    with (out_path.parent / "manifest_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


class DecimaGeneSequenceDataset(Dataset):
    """PyTorch dataset that extracts gene-centered DNA windows on demand."""

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        fasta_path: str | Path,
        y_mean: float,
        y_std: float,
        *,
        augment_rc: bool = False,
        cache_size: int = 512,
    ):
        if isinstance(manifest, (str, Path)):
            self.manifest = pd.read_parquet(manifest).reset_index(drop=True)
        else:
            self.manifest = manifest.reset_index(drop=True).copy()
        self.fasta_path = Path(fasta_path)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std) if float(y_std) > 0 else 1.0
        self.augment_rc = bool(augment_rc)
        self.cache_size = int(cache_size)
        self._reader: FastaReader | None = None
        self._cache: OrderedDict[tuple[str, int, int], str] = OrderedDict()

    @property
    def reader(self) -> FastaReader:
        if self._reader is None:
            self._reader = FastaReader(self.fasta_path)
        return self._reader

    def __len__(self) -> int:
        return len(self.manifest)

    def _fetch_sequence(self, chrom: str, start: int, end: int) -> str:
        key = (str(chrom), int(start), int(end))
        if self.cache_size > 0 and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        seq = self.reader.fetch(chrom, start, end, pad=True)
        if self.cache_size > 0:
            self._cache[key] = seq
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return seq

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.manifest.iloc[int(idx)]
        chrom = row.get("fasta_chrom", row["chrom"])
        seq = self._fetch_sequence(chrom, int(row["seq_start"]), int(row["seq_end"]))
        if self.augment_rc and np.random.random() < 0.5:
            seq = reverse_complement(seq)
        x = one_hot_encode(seq)
        y_raw = float(row["y_teacher"])
        y = (y_raw - self.y_mean) / self.y_std
        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.tensor(y, dtype=torch.float32),
            "y_raw": torch.tensor(y_raw, dtype=torch.float32),
            "gene_id": str(row["gene_id"]),
            "chrom": str(row["chrom"]),
            "split": str(row["split"]),
        }
