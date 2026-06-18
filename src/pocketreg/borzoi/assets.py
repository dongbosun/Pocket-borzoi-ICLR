"""Borzoi asset config parsing and lightweight inspection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .targets import find_k562_rnaseq_candidates, parse_targets


@dataclass(frozen=True)
class BorzoiAssetConfig:
    borzoi_repo: Path | None = None
    baskerville_repo: Path | None = None
    westminster_repo: Path | None = None
    k562_weights_fold0: Path | None = None
    k562_weights_fold1: Path | None = None
    k562_targets: Path | None = None
    k562_params: Path | None = None
    hg38_fasta: Path | None = None
    gencode_gtf: Path | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "BorzoiAssetConfig":
        def maybe_path(key: str) -> Path | None:
            value = data.get(key)
            return Path(value) if value else None

        return cls(
            borzoi_repo=maybe_path("borzoi_repo"),
            baskerville_repo=maybe_path("baskerville_repo"),
            westminster_repo=maybe_path("westminster_repo"),
            k562_weights_fold0=maybe_path("k562_weights_fold0"),
            k562_weights_fold1=maybe_path("k562_weights_fold1"),
            k562_targets=maybe_path("k562_targets"),
            k562_params=maybe_path("k562_params"),
            hg38_fasta=maybe_path("hg38_fasta"),
            gencode_gtf=maybe_path("gencode_gtf"),
        )


def load_assets_config(path: str | Path) -> BorzoiAssetConfig:
    with Path(path).open() as handle:
        data = yaml.safe_load(handle) or {}
    return BorzoiAssetConfig.from_mapping(data)


def path_status(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def sha256_file(path: str | Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    remaining = max_bytes
    with Path(path).open("rb") as handle:
        while True:
            chunk_size = 1024 * 1024
            if remaining is not None:
                if remaining <= 0:
                    break
                chunk_size = min(chunk_size, remaining)
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def load_params(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Borzoi params are usually JSON; this fallback keeps inspection actionable.
        return {"_raw_text": text}


def deep_find_first(mapping: Any, keys: set[str]) -> Any | None:
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            if key in keys:
                return value
            found = deep_find_first(value, keys)
            if found is not None:
                return found
    elif isinstance(mapping, list):
        for value in mapping:
            found = deep_find_first(value, keys)
            if found is not None:
                return found
    return None


def _infer_borzoi_conv_geometry(params: dict[str, Any]) -> dict[str, Any]:
    model = params.get("model", {}) if isinstance(params, dict) else {}
    seq_length = model.get("seq_length")
    trunk = model.get("trunk", []) if isinstance(model, dict) else []
    downsample = 1
    upsample = 1
    crop_bins = 0
    if isinstance(trunk, list):
        for block in trunk:
            if not isinstance(block, dict):
                continue
            repeat = int(block.get("repeat", 1) or 1)
            pool_size = block.get("pool_size")
            if pool_size:
                downsample *= int(pool_size) ** repeat
            if block.get("upsample_conv"):
                upsample *= 2 ** repeat
            if block.get("name") == "Cropping1D" and block.get("cropping") is not None:
                cropping = block["cropping"]
                if isinstance(cropping, int):
                    crop_bins += 2 * cropping
                elif isinstance(cropping, (list, tuple)) and len(cropping) == 2:
                    crop_bins += int(cropping[0]) + int(cropping[1])
    if not seq_length or downsample == 0 or upsample == 0:
        return {}
    if downsample % upsample != 0:
        return {}
    bin_size = downsample // upsample
    bins_before_crop = int(seq_length) // bin_size
    output_bins = bins_before_crop - crop_bins
    return {
        "bin_size": bin_size if bin_size > 0 else None,
        "output_num_bins": output_bins if output_bins > 0 else None,
        "output_crop_bins": crop_bins // 2 if crop_bins else 0,
        "bins_before_crop": bins_before_crop,
        "geometry_source": "inferred_from_model_trunk",
    }


def detect_shape_metadata(params: dict[str, Any]) -> dict[str, Any]:
    inferred = _infer_borzoi_conv_geometry(params)
    detected = {
        "input_len": deep_find_first(params, {"seq_length", "seq_len", "input_len", "input_length"}),
        "output_num_bins": deep_find_first(
            params,
            {"target_length", "output_bins", "output_length", "num_bins"},
        ),
        "bin_size": deep_find_first(params, {"pool_width", "bin_size", "target_pool"}),
        "output_crop": deep_find_first(params, {"crop_bp", "crop", "target_crop"}),
        "output_num_tracks": deep_find_first(params, {"num_targets", "target_channels", "tracks"}),
    }
    if detected["output_num_bins"] is None:
        detected["output_num_bins"] = inferred.get("output_num_bins")
    if detected["bin_size"] is None:
        detected["bin_size"] = inferred.get("bin_size")
    if detected["output_crop"] is None:
        detected["output_crop"] = inferred.get("output_crop_bins")
    detected["geometry_source"] = inferred.get("geometry_source")
    detected["bins_before_crop"] = inferred.get("bins_before_crop")
    return detected


def inspect_assets(config: BorzoiAssetConfig) -> dict[str, Any]:
    report: dict[str, Any] = {
        "paths": {
            "borzoi_repo": path_status(config.borzoi_repo),
            "baskerville_repo": path_status(config.baskerville_repo),
            "westminster_repo": path_status(config.westminster_repo),
            "k562_weights_fold0": path_status(config.k562_weights_fold0),
            "k562_weights_fold1": path_status(config.k562_weights_fold1),
            "k562_targets": path_status(config.k562_targets),
            "k562_params": path_status(config.k562_params),
            "hg38_fasta": path_status(config.hg38_fasta),
            "gencode_gtf": path_status(config.gencode_gtf),
        },
        "targets": {},
        "params": {},
    }
    if config.k562_params and config.k562_params.exists():
        params = load_params(config.k562_params)
        report["params"] = {
            "detected": detect_shape_metadata(params),
            "sha256": sha256_file(config.k562_params),
        }
    if config.k562_targets and config.k562_targets.exists():
        targets = parse_targets(config.k562_targets)
        candidates = find_k562_rnaseq_candidates(targets)
        report["targets"] = {
            "num_targets": len(targets),
            "columns": list(targets[0].keys()) if targets else [],
            "num_k562_rnaseq_candidates": len(candidates),
            "candidate_rows": candidates,
            "sha256": sha256_file(config.k562_targets),
        }
    return report
