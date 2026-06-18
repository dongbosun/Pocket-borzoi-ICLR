"""Pocket-Decima v2 multi-signal data utilities."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from pocketreg.data.anndata_utils import make_json_safe, row_to_metadata, summarize_labels
from pocketreg.data.fasta import FastaReader
from pocketreg.data.manifest import build_manifest_from_anndata, save_manifest_outputs
from pocketreg.data.sequence import one_hot_encode, reverse_complement

LOGGER = logging.getLogger(__name__)
REP_LAYERS = tuple(f"v1_rep{i}" for i in range(4))
STRICT_METADATA_COLS = ("gene_length", "frac_N")
ORACLE_METADATA_COLS = (
    "gene_length",
    "frac_N",
    "frac_nan",
    "mean_counts",
    "n_tracks",
    "pearson",
    "size_factor_pearson",
)


def _layer_row(layer: Any, row_idx: int) -> np.ndarray:
    row = layer[row_idx, :]
    if sparse.issparse(row):
        row = row.toarray()
    return np.asarray(row).reshape(-1).astype(np.float32)


def _layer_rows(layer: Any, row_indices: list[int]) -> np.ndarray:
    rows = layer[row_indices, :]
    if sparse.issparse(rows):
        rows = rows.toarray()
    return np.asarray(rows, dtype=np.float32)


def parse_target_indices(value: str | None, primary: int) -> list[int]:
    """Parse a comma-separated target-index list and keep primary first."""
    indices = [int(primary)]
    if value:
        for item in value.split(","):
            item = item.strip()
            if item:
                indices.append(int(item))
    out: list[int] = []
    for idx in indices:
        if idx not in out:
            out.append(idx)
    if not 1 <= len(out) <= 3:
        raise ValueError("v2 supports 1-3 target celltype outputs.")
    return out


def select_auxiliary_obs(
    adata: ad.AnnData,
    primary_idx: int,
    *,
    same_organ: bool = True,
    same_disease: bool = True,
    same_region: bool = False,
    cell_type_contains: str | None = None,
    max_obs: int = 64,
) -> list[int]:
    """Select biologically nearby pseudobulks for auxiliary PCA labels."""
    obs = adata.obs.copy()
    row = obs.iloc[int(primary_idx)]
    mask = pd.Series(True, index=obs.index)
    for enabled, col in ((same_organ, "organ"), (same_disease, "disease"), (same_region, "region")):
        if enabled and col in obs and col in row and pd.notna(row[col]):
            mask &= obs[col].astype(str).str.casefold().eq(str(row[col]).casefold())
    if cell_type_contains and "cell_type" in obs:
        mask &= obs["cell_type"].astype(str).str.contains(
            cell_type_contains, case=False, na=False, regex=False
        )
    candidates = obs.loc[mask].copy()
    if candidates.empty:
        LOGGER.warning("No aux-neighborhood obs matched; using primary target only.")
        return [int(primary_idx)]
    candidates["_obs_pos"] = [obs.index.get_loc(idx) for idx in candidates.index]
    sort_cols = [c for c in ("test_pearson", "val_pearson", "n_cells") if c in candidates]
    if sort_cols:
        candidates = candidates.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    positions = candidates["_obs_pos"].astype(int).tolist()
    if primary_idx not in positions:
        positions.insert(0, int(primary_idx))
    positions = positions[: int(max_obs)]
    if primary_idx not in positions:
        positions[-1] = int(primary_idx)
    return positions


def add_target_and_replicate_columns(
    manifest: pd.DataFrame,
    adata: ad.AnnData,
    target_indices: list[int],
    *,
    final_layer: str = "preds",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add final and replicate Decima target columns to a manifest."""
    if final_layer not in adata.layers:
        raise KeyError(f"Layer {final_layer!r} not found. Available layers: {list(adata.layers)}")
    manifest = manifest.copy()
    gene_idx = manifest["gene_idx"].to_numpy(dtype=int)
    summary: dict[str, Any] = {
        "target_indices": [int(i) for i in target_indices],
        "target_metadata": [row_to_metadata(adata.obs.iloc[i], i) for i in target_indices],
        "final_layer": final_layer,
        "replicate_layers": [],
    }
    final_matrix = _layer_rows(adata.layers[final_layer], target_indices).T
    for target_pos in range(len(target_indices)):
        col = f"y_final_t{target_pos}"
        manifest[col] = final_matrix[gene_idx, target_pos].astype(np.float32)
    manifest["y_teacher"] = manifest["y_final_t0"].astype(np.float32)

    for rep_layer in REP_LAYERS:
        if rep_layer not in adata.layers:
            continue
        rep_matrix = _layer_rows(adata.layers[rep_layer], target_indices).T
        for target_pos in range(len(target_indices)):
            manifest[f"y_rep_{rep_layer}_t{target_pos}"] = rep_matrix[
                gene_idx, target_pos
            ].astype(np.float32)
        summary["replicate_layers"].append(rep_layer)
    return manifest, summary


def add_aux_pca_columns(
    manifest: pd.DataFrame,
    adata: ad.AnnData,
    aux_obs_indices: list[int],
    *,
    layer: str = "preds",
    n_components: int = 8,
    seed: int = 13,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add train-fitted biological-neighborhood PCA labels."""
    if n_components <= 0:
        return manifest, {"enabled": False, "reason": "n_components <= 0"}
    if layer not in adata.layers:
        raise KeyError(f"Aux layer {layer!r} not found. Available layers: {list(adata.layers)}")
    manifest = manifest.copy()
    gene_idx = manifest["gene_idx"].to_numpy(dtype=int)
    aux_matrix = _layer_rows(adata.layers[layer], aux_obs_indices).T
    values = aux_matrix[gene_idx, :]
    n_components = min(int(n_components), values.shape[1], int((manifest["split"] == "train").sum()))
    if n_components <= 0:
        return manifest, {"enabled": False, "reason": "not enough aux observations or train genes"}
    train_mask = manifest["split"].to_numpy() == "train"
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=seed)
    train_scaled = scaler.fit_transform(values[train_mask])
    pca.fit(train_scaled)
    comps = pca.transform(scaler.transform(values)).astype(np.float32)
    for i in range(n_components):
        manifest[f"aux_pca_{i}"] = comps[:, i]
    return manifest, {
        "enabled": True,
        "layer": layer,
        "aux_obs_indices": [int(i) for i in aux_obs_indices],
        "n_aux_obs": int(len(aux_obs_indices)),
        "n_components": int(n_components),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
    }


def add_residual_columns(
    manifest: pd.DataFrame,
    *,
    target_cols: list[str],
    strict_cols: tuple[str, ...] = STRICT_METADATA_COLS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add strict-metadata baseline predictions and residual target columns."""
    manifest = manifest.copy()
    cols = [c for c in strict_cols if c in manifest]
    if not cols:
        return manifest, {"enabled": False, "reason": "no strict metadata columns"}
    train_mask = manifest["split"].eq("train")
    summary: dict[str, Any] = {"enabled": True, "features": cols, "targets": {}}
    for target_col in target_cols:
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        model.fit(manifest.loc[train_mask, cols], manifest.loc[train_mask, target_col])
        pred = model.predict(manifest[cols]).astype(np.float32)
        pred_col = f"{target_col}_strict_metadata_pred"
        resid_col = f"y_resid_{target_col.removeprefix('y_')}"
        manifest[pred_col] = pred
        manifest[resid_col] = manifest[target_col].astype(np.float32) - pred
        summary["targets"][target_col] = {
            "pred_col": pred_col,
            "residual_col": resid_col,
            "residual_stats": summarize_labels(manifest[resid_col].to_numpy()),
        }
    return manifest, summary


def add_mid_feature_columns(
    manifest: pd.DataFrame, mid_feature_path: str | Path | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge optional precomputed Decima pre-head/middle feature projection columns."""
    if not mid_feature_path:
        return manifest, {"enabled": False, "reason": "no precomputed mid-feature file provided"}
    path = Path(mid_feature_path)
    if not path.exists():
        raise FileNotFoundError(f"Mid-feature file not found: {path}")
    if path.suffix == ".parquet":
        mid = pd.read_parquet(path)
    elif path.suffix in {".csv", ".tsv"}:
        mid = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    else:
        raise ValueError("Mid-feature file must be parquet/csv/tsv with gene_id or gene_idx columns.")
    key = "gene_id" if "gene_id" in mid.columns else "gene_idx" if "gene_idx" in mid.columns else None
    if key is None:
        raise ValueError("Mid-feature file must contain gene_id or gene_idx.")
    feature_cols = [c for c in mid.columns if c.startswith("mid_")]
    if not feature_cols:
        numeric = [c for c in mid.columns if c != key and pd.api.types.is_numeric_dtype(mid[c])]
        feature_cols = numeric
        mid = mid.rename(columns={c: f"mid_{i}" for i, c in enumerate(numeric)})
        feature_cols = [f"mid_{i}" for i in range(len(numeric))]
    merged = manifest.merge(mid[[key] + feature_cols], on=key, how="left")
    return merged, {
        "enabled": True,
        "path": str(path),
        "key": key,
        "columns": feature_cols,
        "missing_rows": int(merged[feature_cols].isna().any(axis=1).sum()),
    }


def sanity_check_decima_layers(
    manifest: pd.DataFrame,
    adata: ad.AnnData,
    target_indices: list[int],
    *,
    final_layer: str = "preds",
    n: int = 256,
    seed: int = 13,
) -> dict[str, Any]:
    """Verify manifest labels match h5ad final and replicate layers."""
    rng = np.random.default_rng(seed)
    take = min(int(n), len(manifest))
    sample_idx = rng.choice(np.arange(len(manifest)), size=take, replace=False)
    sample = manifest.iloc[sample_idx]
    gene_idx = sample["gene_idx"].to_numpy(dtype=int)
    out: dict[str, Any] = {}
    for layer_name in (final_layer, *REP_LAYERS):
        if layer_name not in adata.layers:
            continue
        mat = _layer_rows(adata.layers[layer_name], target_indices).T
        diffs = []
        for target_pos in range(len(target_indices)):
            if layer_name == final_layer:
                col = f"y_final_t{target_pos}"
            else:
                col = f"y_rep_{layer_name}_t{target_pos}"
            if col not in sample:
                continue
            diffs.append(np.nanmax(np.abs(sample[col].to_numpy(dtype=float) - mat[gene_idx, target_pos])))
        out[layer_name] = float(np.nanmax(diffs)) if diffs else None
    return {"n_checked": int(take), "max_abs_diff_by_layer": out}


def build_v2_manifest_from_anndata(
    adata: ad.AnnData,
    primary_idx: int,
    target_indices: list[int],
    context_len: int,
    *,
    fasta_path: str | Path | None = None,
    final_layer: str = "preds",
    split_mode: str = "chromosome",
    include_sex_chromosomes: bool = False,
    all_gene_types: bool = False,
    max_frac_n: float = 0.05,
    max_genes: int | None = None,
    seed: int = 13,
    skip_fasta_check: bool = True,
    aux_pca_components: int = 8,
    aux_same_region: bool = False,
    aux_cell_type_contains: str | None = None,
    aux_max_obs: int = 64,
    residual: bool = True,
    mid_feature_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Build a v2 manifest with final, replicate, aux PCA, residual, and optional mid labels."""
    base, target_metadata, _ = build_manifest_from_anndata(
        adata,
        primary_idx,
        final_layer,
        context_len,
        split_mode=split_mode,
        include_sex_chromosomes=include_sex_chromosomes,
        all_gene_types=all_gene_types,
        max_frac_n=max_frac_n,
        max_genes=max_genes,
        seed=seed,
        fasta_path=fasta_path,
        skip_fasta_check=skip_fasta_check,
    )
    manifest, target_summary = add_target_and_replicate_columns(
        base, adata, target_indices, final_layer=final_layer
    )
    aux_obs = select_auxiliary_obs(
        adata,
        primary_idx,
        same_region=aux_same_region,
        cell_type_contains=aux_cell_type_contains,
        max_obs=aux_max_obs,
    )
    manifest, aux_summary = add_aux_pca_columns(
        manifest,
        adata,
        aux_obs,
        layer=final_layer,
        n_components=aux_pca_components,
        seed=seed,
    )
    final_cols = [f"y_final_t{i}" for i in range(len(target_indices))]
    if residual:
        manifest, residual_summary = add_residual_columns(manifest, target_cols=final_cols)
    else:
        residual_summary = {"enabled": False, "reason": "disabled"}
    manifest, mid_summary = add_mid_feature_columns(manifest, mid_feature_path)
    sanity = sanity_check_decima_layers(manifest, adata, target_indices, final_layer=final_layer, seed=seed)
    summary = {
        "version": "pocket-decima-targeted-distillation-v2",
        "n_genes": int(len(manifest)),
        "context_len": int(context_len),
        "split_counts": {str(k): int(v) for k, v in manifest["split"].value_counts().to_dict().items()},
        "target_metadata": target_metadata,
        "target_signals": target_summary,
        "aux_pca": aux_summary,
        "residual": residual_summary,
        "mid_features": mid_summary,
        "sanity": sanity,
        "teacher_label_stats_manifest": {
            col: summarize_labels(manifest[col].to_numpy()) for col in final_cols
        },
    }
    return manifest.reset_index(drop=True), target_metadata, summary


def save_v2_manifest_outputs(
    manifest: pd.DataFrame,
    out_path: str | Path,
    target_metadata: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Save a v2 manifest and sidecar summaries."""
    save_manifest_outputs(manifest, out_path, target_metadata, summary)
    out_path = Path(out_path)
    with (out_path.parent / "v2_signal_summary.json").open("w") as f:
        json.dump({k: make_json_safe(v) for k, v in summary.items()}, f, indent=2)


def _columns_with_prefix(frame: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in frame.columns if c.startswith(prefix)]


class DecimaV2GeneSequenceDataset(Dataset):
    """On-the-fly 5-channel DNA+gene-mask dataset for v2 multi-head training."""

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        fasta_path: str | Path,
        *,
        label_columns: dict[str, list[str]],
        normalizers: dict[str, dict[str, list[float]]],
        augment_rc: bool = False,
        cache_size: int = 512,
        input_channels: int = 5,
    ):
        if isinstance(manifest, (str, Path)):
            self.manifest = pd.read_parquet(manifest).reset_index(drop=True)
        else:
            self.manifest = manifest.reset_index(drop=True).copy()
        self.fasta_path = Path(fasta_path)
        self.label_columns = label_columns
        self.normalizers = normalizers
        self.augment_rc = bool(augment_rc)
        self.cache_size = int(cache_size)
        if int(input_channels) not in (4, 5):
            raise ValueError("DecimaV2GeneSequenceDataset input_channels must be 4 or 5.")
        self.input_channels = int(input_channels)
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

    @staticmethod
    def _gene_mask(row: pd.Series) -> np.ndarray:
        start = int(row["seq_start"])
        end = int(row["seq_end"])
        length = end - start
        gene_start = int(row["gene_start"]) if np.isfinite(row["gene_start"]) else int(row["start"])
        gene_end = int(row["gene_end"]) if np.isfinite(row["gene_end"]) else int(row["end"])
        lo = max(start, min(gene_start, gene_end))
        hi = min(end, max(gene_start, gene_end))
        mask = np.zeros(length, dtype=np.float32)
        if hi > lo:
            mask[(lo - start) : (hi - start)] = 1.0
        return mask

    def _standardized(self, row: pd.Series, group: str) -> torch.Tensor:
        cols = self.label_columns.get(group, [])
        if not cols:
            return torch.zeros(0, dtype=torch.float32)
        values = row[cols].astype(float).to_numpy(dtype=np.float32)
        norm = self.normalizers[group]
        mean = np.asarray(norm["mean"], dtype=np.float32)
        std = np.asarray(norm["std"], dtype=np.float32)
        std = np.where(std > 0, std, 1.0)
        values = np.where(np.isfinite(values), values, mean)
        return torch.from_numpy((values - mean) / std).float()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.manifest.iloc[int(idx)]
        chrom = row.get("fasta_chrom", row["chrom"])
        seq = self._fetch_sequence(chrom, int(row["seq_start"]), int(row["seq_end"]))
        mask = self._gene_mask(row)
        if self.augment_rc and np.random.random() < 0.5:
            seq = reverse_complement(seq)
            mask = mask[::-1].copy()
        dna = one_hot_encode(seq)
        if self.input_channels == 5:
            x = np.concatenate([dna, mask[None, :]], axis=0).astype(np.float32)
        else:
            x = dna.astype(np.float32)
        out = {
            "x": torch.from_numpy(x).float(),
            "gene_id": str(row["gene_id"]),
            "chrom": str(row["chrom"]),
            "split": str(row["split"]),
        }
        for group in ("final", "rep", "aux", "residual", "mid"):
            out[group] = self._standardized(row, group)
            raw_cols = self.label_columns.get(group, [])
            if raw_cols:
                out[f"{group}_raw"] = torch.tensor(
                    row[raw_cols].astype(float).to_numpy(dtype=np.float32), dtype=torch.float32
                )
            else:
                out[f"{group}_raw"] = torch.zeros(0, dtype=torch.float32)
        return out


class DecimaV2CachedGeneSequenceDataset(Dataset):
    """Memmap-backed v2 dataset using precomputed uint8 DNA+mask channels."""

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        cache_dir: str | Path,
        *,
        label_columns: dict[str, list[str]],
        normalizers: dict[str, dict[str, list[float]]],
        augment_rc: bool = False,
        input_channels: int = 5,
    ):
        if isinstance(manifest, (str, Path)):
            self.manifest = pd.read_parquet(manifest).reset_index(drop=True)
        else:
            self.manifest = manifest.reset_index(drop=True).copy()
        self.cache_dir = Path(cache_dir)
        summary_path = self.cache_dir / "cache_summary.json"
        index_path = self.cache_dir / "cache_index.parquet"
        if not summary_path.exists():
            raise FileNotFoundError(f"Sequence cache summary not found: {summary_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Sequence cache index not found: {index_path}")
        with summary_path.open() as handle:
            self.cache_summary = json.load(handle)
        self.cache_index = pd.read_parquet(index_path)
        self.label_columns = label_columns
        self.normalizers = normalizers
        self.augment_rc = bool(augment_rc)
        if int(input_channels) not in (4, 5):
            raise ValueError("DecimaV2CachedGeneSequenceDataset input_channels must be 4 or 5.")
        self.input_channels = int(input_channels)
        self._memmap: np.memmap | None = None
        self._cache_idx_by_gene_idx = {
            int(row.gene_idx): int(row.cache_idx) for row in self.cache_index.itertuples(index=False)
        }
        missing = sorted(set(self.manifest["gene_idx"].astype(int)) - set(self._cache_idx_by_gene_idx))
        if missing:
            raise ValueError(
                f"Sequence cache is missing {len(missing)} manifest gene_idx values; "
                f"first missing: {missing[:5]}"
            )
        expected_context = int(self.cache_summary["context_len"])
        if "context_len" in self.manifest and not self.manifest["context_len"].astype(int).eq(expected_context).all():
            raise ValueError(
                f"Manifest context_len does not match sequence cache context_len={expected_context}."
            )

    @property
    def memmap(self) -> np.memmap:
        if self._memmap is None:
            shape = tuple(int(x) for x in self.cache_summary["shape"])
            dtype = np.dtype(self.cache_summary.get("dtype", "uint8"))
            path = self.cache_dir / self.cache_summary.get("cache_file", "sequence_cache_uint8.dat")
            if not path.exists():
                raise FileNotFoundError(f"Sequence cache data file not found: {path}")
            self._memmap = np.memmap(path, mode="r", dtype=dtype, shape=shape)
        return self._memmap

    def __len__(self) -> int:
        return len(self.manifest)

    def _standardized(self, row: pd.Series, group: str) -> torch.Tensor:
        cols = self.label_columns.get(group, [])
        if not cols:
            return torch.zeros(0, dtype=torch.float32)
        values = row[cols].astype(float).to_numpy(dtype=np.float32)
        norm = self.normalizers[group]
        mean = np.asarray(norm["mean"], dtype=np.float32)
        std = np.asarray(norm["std"], dtype=np.float32)
        std = np.where(std > 0, std, 1.0)
        values = np.where(np.isfinite(values), values, mean)
        return torch.from_numpy((values - mean) / std).float()

    @staticmethod
    def _reverse_complement_channels(x: np.ndarray) -> np.ndarray:
        dna = x[:4][[3, 2, 1, 0], ::-1]
        if x.shape[0] == 4:
            return dna.copy()
        mask = x[4:5, ::-1]
        return np.concatenate([dna, mask], axis=0).copy()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.manifest.iloc[int(idx)]
        cache_idx = self._cache_idx_by_gene_idx[int(row["gene_idx"])]
        x = np.asarray(self.memmap[cache_idx, : self.input_channels, :], dtype=np.float32)
        if self.augment_rc and np.random.random() < 0.5:
            x = self._reverse_complement_channels(x)
        out = {
            "x": torch.from_numpy(x).float(),
            "gene_id": str(row["gene_id"]),
            "chrom": str(row["chrom"]),
            "split": str(row["split"]),
        }
        for group in ("final", "rep", "aux", "residual", "mid"):
            out[group] = self._standardized(row, group)
            raw_cols = self.label_columns.get(group, [])
            if raw_cols:
                out[f"{group}_raw"] = torch.tensor(
                    row[raw_cols].astype(float).to_numpy(dtype=np.float32), dtype=torch.float32
                )
            else:
                out[f"{group}_raw"] = torch.zeros(0, dtype=torch.float32)
        return out
