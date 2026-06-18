"""Keras/Borzoi layer inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json


def _shape_to_list(shape: Any) -> list[int | None] | list[list[int | None]] | None:
    if shape is None:
        return None
    if isinstance(shape, (list, tuple)):
        if shape and all(isinstance(item, (list, tuple)) for item in shape):
            return [_shape_to_list(item) for item in shape]  # type: ignore[list-item]
        out: list[int | None] = []
        for value in shape:
            if value is None:
                out.append(None)
            else:
                try:
                    out.append(int(value))
                except Exception:
                    out.append(None)
        return out
    try:
        return [int(shape)]
    except Exception:
        return None


def _layer_output_shape(layer: Any) -> Any:
    for attr in ("output_shape", "batch_output_shape"):
        try:
            value = getattr(layer, attr)
            if value is not None:
                return value
        except Exception:
            pass
    try:
        return layer.output.shape
    except Exception:
        return None


def _dtype_name(layer: Any) -> str | None:
    for attr in ("dtype", "compute_dtype"):
        try:
            value = getattr(layer, attr)
            if value is not None:
                return str(value)
        except Exception:
            pass
    return None


def can_make_intermediate_model(model: Any, layer_name: str) -> bool:
    """Return whether a layer can be used as a Keras intermediate output."""

    try:
        import tensorflow as tf  # type: ignore

        layer = model.get_layer(layer_name)
        tf.keras.Model(inputs=model.inputs, outputs=layer.output)
        return True
    except Exception:
        return False


def inspect_keras_model(model: Any) -> dict[str, Any]:
    """Collect serializable model and layer metadata."""

    layers: list[dict[str, Any]] = []
    for idx, layer in enumerate(model.layers):
        raw_shape = _layer_output_shape(layer)
        shape = _shape_to_list(raw_shape)
        rank = len(shape) if isinstance(shape, list) and (not shape or isinstance(shape[0], (int, type(None)))) else None
        layers.append(
            {
                "index": idx,
                "name": str(layer.name),
                "class_name": layer.__class__.__name__,
                "output_shape": shape,
                "dtype": _dtype_name(layer),
                "trainable": bool(getattr(layer, "trainable", False)),
                "rank": rank,
                "can_intermediate_output": can_make_intermediate_model(model, str(layer.name)),
            }
        )
    return {
        "input_shape": _shape_to_list(getattr(model, "input_shape", None)),
        "output_shape": _shape_to_list(getattr(model, "output_shape", None)),
        "num_layers": len(layers),
        "layers": layers,
        "candidates": identify_candidate_layers(layers),
    }


def _last_layers_with_rank(layers: list[dict[str, Any]], rank: int) -> list[dict[str, Any]]:
    return [layer for layer in layers if layer.get("rank") == rank and layer.get("can_intermediate_output")]


def identify_candidate_layers(layers: list[dict[str, Any]]) -> dict[str, Any]:
    """Heuristically identify final, penultimate, head-input, and middle layers."""

    candidates: dict[str, Any] = {}
    usable = [layer for layer in layers if layer.get("can_intermediate_output")]
    if usable:
        candidates["final_output"] = usable[-1]
    rank3 = _last_layers_with_rank(layers, 3)
    if rank3:
        candidates["last_spatial"] = rank3[-1]
        if len(rank3) >= 2:
            candidates["penultimate_spatial"] = rank3[-2]
        if len(rank3) >= 3:
            candidates["head_input"] = rank3[-3]
        candidates["middle_spatial"] = rank3[len(rank3) // 2]
    rank2 = _last_layers_with_rank(layers, 2)
    if rank2:
        candidates["last_vector"] = rank2[-1]
    return candidates


def write_layer_report(report: dict[str, Any], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def write_summary_markdown(fold_reports: dict[str, dict[str, Any]], path: Path) -> None:
    lines = ["# Borzoi Teacher Layer Summary", ""]
    for fold_name, report in sorted(fold_reports.items()):
        lines.extend(
            [
                f"## {fold_name}",
                "",
                f"- input shape: `{report.get('input_shape')}`",
                f"- output shape: `{report.get('output_shape')}`",
                f"- number of layers: `{report.get('num_layers')}`",
                "",
                "| candidate | index | name | class | output shape |",
                "|---|---:|---|---|---|",
            ]
        )
        for key, layer in (report.get("candidates") or {}).items():
            lines.append(
                f"| {key} | {layer.get('index')} | `{layer.get('name')}` | "
                f"{layer.get('class_name')} | `{layer.get('output_shape')}` |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
