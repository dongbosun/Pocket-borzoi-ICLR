"""Mini-Borzoi K562 target selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

import numpy as np
import pandas as pd
import yaml

from pocketreg.borzoi.targets import find_k562_rnaseq_candidates, parse_targets, row_target_index


def _row_name(row: dict[str, Any]) -> str:
    identifier = row.get("identifier") or row.get("target") or row.get("index")
    desc = row.get("description") or row.get("desc") or ""
    return f"{identifier}:{desc}".strip(":")


def _target_record(row: dict[str, Any]) -> dict[str, Any]:
    idx = row_target_index(row)
    return {
        "index": int(idx) if idx is not None else None,
        "identifier": row.get("identifier"),
        "description": row.get("description") or row.get("desc"),
        "name": _row_name(row),
        "row": row,
    }


def _requested_primary(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    primary_cfg = config.get("primary_targets", {}) or {}
    requested_indices = {int(v) for v in primary_cfg.get("indices", []) or []}
    requested_names = [str(v).lower() for v in primary_cfg.get("names", []) or []]
    selected: list[dict[str, Any]] = []
    for row in rows:
        idx = row_target_index(row)
        haystack = " ".join(str(value) for value in row.values()).lower()
        if idx in requested_indices or any(name in haystack for name in requested_names):
            selected.append(row)
    return selected


def select_targets(
    targets_path: str | Path,
    config: dict[str, Any],
    ref_labels_path: str | Path | None = None,
) -> dict[str, Any]:
    """Select one primary target and metadata-only auxiliary K562 RNA-seq targets.

    When rich per-target cache is not available yet, aux target correlation cannot
    be computed. This function records that bootstrap mode explicitly.
    """

    rows = parse_targets(targets_path)
    candidates = find_k562_rnaseq_candidates(rows)
    if not candidates:
        raise ValueError(f"No K562 RNA/RNA-seq targets found in {targets_path}")

    requested = _requested_primary(rows, config)
    primary_cfg = config.get("primary_targets", {}) or {}
    max_primary = int(primary_cfg.get("max_primary", 1) or 1)
    primary_rows = (requested or candidates)[:max_primary]
    primary_indices = {row_target_index(row) for row in primary_rows}

    aux_cfg = config.get("aux_targets", {}) or {}
    k_aux = int(aux_cfg.get("k_aux", 16) or 16)
    exclude_primary = bool(aux_cfg.get("exclude_primary", True))
    aux_rows = []
    for row in candidates:
        idx = row_target_index(row)
        if exclude_primary and idx in primary_indices:
            continue
        aux_rows.append(row)
        if len(aux_rows) >= k_aux:
            break

    ref_label_stats: dict[str, Any] = {}
    if ref_labels_path and Path(ref_labels_path).exists():
        frame = pd.read_parquet(ref_labels_path)
        value_col = "q_teacher" if "q_teacher" in frame else None
        if value_col:
            values = pd.to_numeric(frame[value_col], errors="coerce")
            ref_label_stats = {
                "source": str(ref_labels_path),
                "value_col": value_col,
                "finite_fraction": float(np.isfinite(values).mean()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    selection_mode = "metadata_only_pending_rich_cache"
    return {
        "selection_mode": selection_mode,
        "targets_path": str(targets_path),
        "num_targets": len(rows),
        "num_k562_rnaseq_candidates": len(candidates),
        "primary_targets": [_target_record(row) for row in primary_rows],
        "aux_targets": [_target_record(row) for row in aux_rows],
        "requested_primary_found": bool(requested),
        "ref_label_stats": ref_label_stats,
        "notes": [
            "Aux targets are metadata-ranked until rich per-target train-gene cache is available.",
            "Re-run after cache_borzoi_ref_rich.py to rank aux targets by train-gene correlation.",
        ],
    }


def write_selection_outputs(selection: dict[str, Any], out_dir: Path, local_config: Path, manifest_json: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for group in ("primary_targets", "aux_targets"):
        for rank, target in enumerate(selection[group]):
            rows.append(
                {
                    "group": group.replace("_targets", ""),
                    "rank": rank,
                    "index": target.get("index"),
                    "identifier": target.get("identifier"),
                    "description": target.get("description"),
                    "name": target.get("name"),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "task_selection.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    config_data = {
        "selection_mode": selection["selection_mode"],
        "primary_targets": [
            {
                "index": target["index"],
                "identifier": target["identifier"],
                "description": target["description"],
                "name": target["name"],
            }
            for target in selection["primary_targets"]
        ],
        "aux_targets": [
            {
                "index": target["index"],
                "identifier": target["identifier"],
                "description": target["description"],
                "name": target["name"],
            }
            for target in selection["aux_targets"]
        ],
    }
    local_config.parent.mkdir(parents=True, exist_ok=True)
    local_config.write_text(yaml.safe_dump(config_data, sort_keys=False))
