"""Inference benchmark helpers."""

from __future__ import annotations

import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from pocketreg.data.manifest import DecimaGeneSequenceDataset
from pocketreg.models.small_cnn import count_parameters, estimate_model_size_mb


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def _random_onehot(batch_size: int, context_len: int, device: torch.device) -> torch.Tensor:
    idx = torch.randint(0, 4, (batch_size, context_len), device=device)
    return torch.nn.functional.one_hot(idx, num_classes=4).permute(0, 2, 1).float()


def summarize_times(times: list[float], batch_size: int) -> dict[str, float]:
    """Summarize seconds-per-batch timings in milliseconds and throughput."""
    arr = np.asarray(times, dtype=np.float64)
    mean_ms = float(np.mean(arr) * 1000)
    median_ms = float(median(arr) * 1000)
    p95_ms = float(np.percentile(arr, 95) * 1000)
    return {
        "mean_ms_per_batch": mean_ms,
        "median_ms_per_batch": median_ms,
        "p95_ms_per_batch": p95_ms,
        "mean_ms_per_gene": mean_ms / batch_size,
        "genes_per_second": float(batch_size / np.mean(arr)),
    }


def benchmark_model_only(
    model: torch.nn.Module,
    *,
    device: torch.device,
    context_len: int,
    batch_size: int,
    num_warmup: int,
    num_steps: int,
) -> dict[str, Any]:
    """Benchmark model forward latency using generated one-hot input."""
    model.eval()
    x = _random_onehot(batch_size, context_len, device)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
        _sync(device)
        times = []
        for _ in range(num_steps):
            start = time.perf_counter()
            _ = model(x)
            _sync(device)
            times.append(time.perf_counter() - start)
    return summarize_times(times, batch_size)


def benchmark_end_to_end(
    model: torch.nn.Module,
    manifest_path: str | Path,
    fasta_path: str | Path,
    *,
    device: torch.device,
    y_mean: float,
    y_std: float,
    batch_size: int,
    num_warmup: int,
    num_steps: int,
) -> dict[str, Any]:
    """Benchmark DataLoader plus FASTA extraction plus model forward."""
    dataset = DecimaGeneSequenceDataset(manifest_path, fasta_path, y_mean, y_std, cache_size=0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    iterator = iter(loader)
    model.eval()

    def next_batch() -> dict[str, Any]:
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    with torch.no_grad():
        for _ in range(num_warmup):
            batch = next_batch()
            _ = model(batch["x"].to(device))
        _sync(device)
        times = []
        for _ in range(num_steps):
            start = time.perf_counter()
            batch = next_batch()
            _ = model(batch["x"].to(device))
            _sync(device)
            times.append(time.perf_counter() - start)
    return summarize_times(times, batch_size)


def memory_report(device: torch.device, before_rss_mb: float, after_rss_mb: float) -> dict[str, Any]:
    """Return device-aware memory fields."""
    report: dict[str, Any] = {
        "process_rss_mb_before": before_rss_mb,
        "process_rss_mb_after": after_rss_mb,
    }
    if device.type == "cuda":
        report["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    elif device.type == "mps":
        report["peak_memory_mb"] = None
        report["mps_peak_unavailable"] = True
    else:
        report["peak_memory_mb"] = after_rss_mb
    return report


def attach_common_fields(
    result: dict[str, Any],
    *,
    mode: str,
    device: torch.device,
    batch_size: int,
    context_len: int,
    num_warmup: int,
    num_steps: int,
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Attach common benchmark metadata to one result row."""
    result = dict(result)
    result.update(
        {
            "mode": mode,
            "device": str(device),
            "batch_size": int(batch_size),
            "context_len": int(context_len),
            "n_steps": int(num_steps),
            "warmup": int(num_warmup),
            "model_params": int(count_parameters(model)),
            "model_size_mb": float(estimate_model_size_mb(model)),
        }
    )
    return result
