#!/usr/bin/env python
"""Prepare Pocket-Decima v2 5k ablation configs and manifest variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketreg.data.v2 import _layer_rows  # noqa: E402


ABLATIONS = {
    "A0_final_only_dna4": {
        "purpose": "pure v2 trunk baseline without gene mask/aux",
        "manifest_variant": "base",
        "input_channels": 4,
        "weights": {"final": 1.0, "rep": 0.0, "aux": 0.0, "residual": 0.0, "mid": 0.0},
    },
    "A1_final_only_dna5_mask": {
        "purpose": "isolate gene mask contribution",
        "manifest_variant": "base",
        "input_channels": 5,
        "weights": {"final": 1.0, "rep": 0.0, "aux": 0.0, "residual": 0.0, "mid": 0.0},
    },
    "A2_mask_replicates": {
        "purpose": "isolate replicate soft-label contribution",
        "manifest_variant": "base",
        "input_channels": 5,
        "weights": {"final": 1.0, "rep": 0.25, "aux": 0.0, "residual": 0.0, "mid": 0.0},
    },
    "A3_mask_aux_raw_pca": {
        "purpose": "isolate raw aux PCA contribution",
        "manifest_variant": "base",
        "input_channels": 5,
        "weights": {"final": 1.0, "rep": 0.0, "aux": 0.1, "residual": 0.0, "mid": 0.0},
    },
    "A4_mask_residual": {
        "purpose": "isolate residual branch contribution",
        "manifest_variant": "base",
        "input_channels": 5,
        "weights": {"final": 1.0, "rep": 0.0, "aux": 0.0, "residual": 0.2, "mid": 0.0},
    },
    "A5_full_v2_reproduce": {
        "purpose": "reproduce current v2 under controlled ablation runner",
        "manifest_variant": "base",
        "input_channels": 5,
        "weights": {"final": 1.0, "rep": 0.25, "aux": 0.1, "residual": 0.2, "mid": 0.0},
    },
    "A6_full_v2_no_aux_pc1": {
        "purpose": "test whether raw aux PCA improvement is dominated by PC1",
        "manifest_variant": "aux_no_pc1",
        "input_channels": 5,
        "weights": {"final": 1.0, "rep": 0.25, "aux": 0.1, "residual": 0.2, "mid": 0.0},
    },
    "A7_full_v2_aux_residualized": {
        "purpose": "make aux signal more target/tissue-specific and less common-prior dominated",
        "manifest_variant": "aux_residualized",
        "input_channels": 5,
        "weights": {"final": 1.0, "rep": 0.25, "aux": 0.1, "residual": 0.2, "mid": 0.0},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/decima_v2_100k.yaml")
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--signal-summary", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs/ablations/v2_5k_obs88_64kb")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def remove_aux_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aux_cols = [c for c in frame.columns if c.startswith("aux_pca_")]
    return frame.drop(columns=aux_cols)


def strict_features(frame: pd.DataFrame) -> list[str]:
    return [c for c in ("gene_length", "frac_N") if c in frame.columns]


def aux_matrix_for_manifest(adata: ad.AnnData, frame: pd.DataFrame, aux_obs_indices: list[int]) -> np.ndarray:
    matrix = _layer_rows(adata.layers["preds"], aux_obs_indices).T
    return matrix[frame["gene_idx"].to_numpy(dtype=int), :].astype(np.float32)


def add_aux_no_pc1(frame: pd.DataFrame, adata: ad.AnnData, aux_obs_indices: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = remove_aux_columns(frame).copy()
    values = aux_matrix_for_manifest(adata, out, aux_obs_indices)
    train = out["split"].eq("train").to_numpy()
    scaler = StandardScaler()
    pca = PCA(n_components=9, random_state=1)
    pca.fit(scaler.fit_transform(values[train]))
    comps = pca.transform(scaler.transform(values)).astype(np.float32)[:, 1:9]
    for idx in range(comps.shape[1]):
        out[f"aux_pca_{idx}"] = comps[:, idx]
    return out, {
        "variant": "aux_no_pc1",
        "source": "raw aux preds PCA, PC1 dropped",
        "n_components_fit": 9,
        "n_components_used": int(comps.shape[1]),
        "explained_variance_ratio_fit": [float(v) for v in pca.explained_variance_ratio_],
        "used_original_pc_indices": list(range(1, 9)),
    }


def residualize_aux(values: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    features = strict_features(frame)
    if not features:
        raise ValueError("Cannot residualize aux PCA without strict metadata features.")
    train = frame["split"].eq("train").to_numpy()
    x = frame[features].apply(pd.to_numeric, errors="coerce")
    residuals = np.zeros_like(values, dtype=np.float32)
    for col_idx in range(values.shape[1]):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        y = values[:, col_idx]
        model.fit(x.loc[train], y[train])
        residuals[:, col_idx] = y - model.predict(x).astype(np.float32)
    return residuals


def add_aux_residualized(frame: pd.DataFrame, adata: ad.AnnData, aux_obs_indices: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = remove_aux_columns(frame).copy()
    values = aux_matrix_for_manifest(adata, out, aux_obs_indices)
    residuals = residualize_aux(values, out)
    train = out["split"].eq("train").to_numpy()
    scaler = StandardScaler()
    pca = PCA(n_components=8, random_state=1)
    pca.fit(scaler.fit_transform(residuals[train]))
    comps = pca.transform(scaler.transform(residuals)).astype(np.float32)
    for idx in range(comps.shape[1]):
        out[f"aux_pca_{idx}"] = comps[:, idx]
    return out, {
        "variant": "aux_residualized",
        "source": "aux preds residualized by strict metadata Ridge, then PCA",
        "strict_features": strict_features(out),
        "n_components_used": int(comps.shape[1]),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
    }


def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def ablation_config(base: dict[str, Any], name: str, spec: dict[str, Any], manifest_path: Path, out_root: Path, seed: int) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base))
    cfg["seed"] = int(seed)
    cfg["manifest_path"] = str(manifest_path)
    cfg["output_dir"] = str(out_root / "runs" / name)
    cfg.setdefault("model", {})
    cfg["model"]["input_channels"] = int(spec["input_channels"])
    cfg.setdefault("train", {})
    weights = spec["weights"]
    cfg["train"]["final_loss_weight"] = float(weights["final"])
    cfg["train"]["rep_loss_weight"] = float(weights["rep"])
    cfg["train"]["aux_loss_weight"] = float(weights["aux"])
    cfg["train"]["residual_loss_weight"] = float(weights["residual"])
    cfg["train"]["mid_loss_weight"] = float(weights["mid"])
    cfg.setdefault("logging", {})
    cfg["ablation"] = {"name": name, "purpose": spec["purpose"], "manifest_variant": spec["manifest_variant"]}
    return cfg


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifests_dir = args.out_root / "manifests"
    configs_dir = args.out_root / "configs"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    base_manifest = pd.read_parquet(args.base_manifest)
    with args.signal_summary.open() as handle:
        signal_summary = json.load(handle)
    aux_obs_indices = [int(x) for x in signal_summary["aux_pca"]["aux_obs_indices"]]
    adata = ad.read_h5ad(args.adata)
    variants: dict[str, Path] = {"base": args.base_manifest.resolve()}
    variant_summaries: dict[str, Any] = {
        "base": {"path": str(args.base_manifest.resolve()), "source": "existing v2 5k manifest"}
    }
    no_pc1, no_pc1_summary = add_aux_no_pc1(base_manifest, adata, aux_obs_indices)
    no_pc1_path = manifests_dir / "manifest_aux_no_pc1.parquet"
    no_pc1.to_parquet(no_pc1_path, index=False)
    variants["aux_no_pc1"] = no_pc1_path.resolve()
    variant_summaries["aux_no_pc1"] = {"path": str(no_pc1_path.resolve()), **no_pc1_summary}

    resid, resid_summary = add_aux_residualized(base_manifest, adata, aux_obs_indices)
    resid_path = manifests_dir / "manifest_aux_residualized.parquet"
    resid.to_parquet(resid_path, index=False)
    variants["aux_residualized"] = resid_path.resolve()
    variant_summaries["aux_residualized"] = {"path": str(resid_path.resolve()), **resid_summary}

    with args.base_config.open() as handle:
        base_cfg = yaml.safe_load(handle)
    configs: dict[str, str] = {}
    for name, spec in ABLATIONS.items():
        cfg = ablation_config(base_cfg, name, spec, variants[spec["manifest_variant"]], args.out_root, args.seed)
        cfg_path = configs_dir / f"{name}.yaml"
        write_yaml(cfg, cfg_path)
        (configs_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
        configs[name] = str(cfg_path.resolve())
    summary = {
        "job": "job2_v2_5k_ablation_prepare",
        "status": "completed",
        "seed": int(args.seed),
        "out_root": str(args.out_root.resolve()),
        "ablations": ABLATIONS,
        "configs": configs,
        "manifest_variants": variant_summaries,
    }
    (args.out_root / "ablation_prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "outputs/reports/job2_ablation_prepare_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
