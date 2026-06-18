"""Inspect official processed Borzoi/K562 data directories without loading large arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


METADATA_SUFFIXES = {".json", ".yaml", ".yml", ".txt", ".tsv", ".csv", ".bed"}
HDF5_SUFFIXES = {".h5", ".hdf5", ".w5"}
TFRECORD_SUFFIXES = {".tfr", ".tfrecord", ".tfrecords"}


def inspect_hdf5(path: Path) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
    except Exception as exc:
        return {"path": str(path), "error": f"h5py unavailable: {exc}"}
    info: dict[str, Any] = {"path": str(path), "datasets": {}, "attrs": {}}
    with h5py.File(path, "r") as handle:
        info["attrs"] = {key: str(value) for key, value in handle.attrs.items()}

        def visit(name: str, obj) -> None:
            if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                info["datasets"][name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }

        handle.visititems(visit)
    return info


def inspect_processed_data_dir(data_dir: str | Path, max_files: int = 5000) -> dict[str, Any]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    report: dict[str, Any] = {
        "data_dir": str(root),
        "metadata_files": [],
        "hdf5_files": [],
        "tfrecord_files": [],
        "bed_files": [],
        "detected": {},
    }
    files = [path for path in root.rglob("*") if path.is_file()]
    for path in files[:max_files]:
        rel = str(path.relative_to(root))
        suffix = path.suffix.lower()
        lower_name = path.name.lower()
        if suffix in METADATA_SUFFIXES:
            report["metadata_files"].append({"path": rel, "size_bytes": path.stat().st_size})
        if suffix in HDF5_SUFFIXES:
            report["hdf5_files"].append(inspect_hdf5(path))
        if suffix in TFRECORD_SUFFIXES:
            report["tfrecord_files"].append({"path": rel, "size_bytes": path.stat().st_size})
        if suffix == ".bed":
            report["bed_files"].append({"path": rel, "size_bytes": path.stat().st_size})
        if lower_name == "targets_human.txt":
            report["detected"]["targets_human"] = rel
        if "param" in lower_name and suffix in {".json", ".yaml", ".yml"}:
            report["detected"].setdefault("params", []).append(rel)
    report["num_files_scanned"] = min(len(files), max_files)
    report["num_files_total"] = len(files)
    report["looks_usable_as_labels"] = bool(report["hdf5_files"] or report["tfrecord_files"]) and bool(
        report["detected"].get("targets_human")
    )
    if report["looks_usable_as_labels"]:
        report["suggested_config"] = {
            "mode": "official_processed_k562_data",
            "data_dir": str(root),
            "targets": report["detected"].get("targets_human"),
        }
    else:
        report["note"] = (
            "Insufficient metadata to map labels to genome intervals. "
            "Use teacher-pseudolabel mode until targets, intervals, and arrays can be linked."
        )
    return report


def write_processed_report(report: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "processed_k562_inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"data_dir: {report['data_dir']}",
        f"files_scanned: {report['num_files_scanned']} / {report['num_files_total']}",
        f"metadata_files: {len(report['metadata_files'])}",
        f"hdf5_files: {len(report['hdf5_files'])}",
        f"tfrecord_files: {len(report['tfrecord_files'])}",
        f"looks_usable_as_labels: {report['looks_usable_as_labels']}",
        report.get("note", ""),
    ]
    (out / "processed_k562_inspection.txt").write_text("\n".join(lines) + "\n")
