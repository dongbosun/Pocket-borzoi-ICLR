"""Map genomic intervals to Borzoi output bins and aggregate selected tracks."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BinOverlap:
    bin_index: int
    overlap_bp: int


@dataclass(frozen=True)
class AggregationResult:
    q: float
    raw_mean: float
    raw_sum: float
    log1p_mean: float
    n_bins_used: int


class BorzoiOutputMapper:
    def __init__(
        self,
        input_seq_start: int,
        input_len: int,
        output_num_bins: int,
        bin_size: int,
        target_index: int,
        output_core_start: int | None = None,
    ):
        if input_len <= 0 or output_num_bins <= 0 or bin_size <= 0:
            raise ValueError("input_len, output_num_bins, and bin_size must be positive")
        self.input_seq_start = int(input_seq_start)
        self.input_len = int(input_len)
        self.output_num_bins = int(output_num_bins)
        self.bin_size = int(bin_size)
        self.target_index = int(target_index)
        output_core_bp = self.output_num_bins * self.bin_size
        if output_core_start is None:
            offset = floor((self.input_len - output_core_bp) / 2)
            output_core_start = self.input_seq_start + offset
        self.output_core_start = int(output_core_start)
        self.output_core_end = self.output_core_start + output_core_bp

    def bin_interval(self, bin_index: int) -> tuple[int, int]:
        start = self.output_core_start + bin_index * self.bin_size
        return start, start + self.bin_size

    def genomic_interval_to_bins(
        self,
        start: int,
        end: int,
        mode: str = "overlap",
    ) -> list[BinOverlap]:
        if end <= start:
            return []
        clipped_start = max(start, self.output_core_start)
        clipped_end = min(end, self.output_core_end)
        if clipped_end <= clipped_start:
            return []
        first = max(0, (clipped_start - self.output_core_start) // self.bin_size)
        last = min(
            self.output_num_bins - 1,
            (clipped_end - 1 - self.output_core_start) // self.bin_size,
        )
        overlaps: list[BinOverlap] = []
        for idx in range(first, last + 1):
            bin_start, bin_end = self.bin_interval(idx)
            overlap = max(0, min(end, bin_end) - max(start, bin_start))
            if overlap <= 0:
                continue
            if mode == "contained" and overlap != self.bin_size:
                continue
            overlaps.append(BinOverlap(idx, overlap))
        return overlaps

    def _extract_track(self, output: np.ndarray) -> np.ndarray:
        arr = np.asarray(output)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected Borzoi output with 2 or 3 dimensions, got {arr.shape}")
        if self.target_index >= arr.shape[-1]:
            raise ValueError(
                f"target_index {self.target_index} out of bounds for output shape {arr.shape}"
            )
        return arr[:, self.target_index]

    def aggregate_bins(
        self,
        output: np.ndarray,
        overlaps: list[BinOverlap],
        mode: str,
    ) -> AggregationResult | None:
        if not overlaps:
            return None
        track = self._extract_track(output)
        values = track[[item.bin_index for item in overlaps]]
        raw_mean = float(np.mean(values))
        raw_sum = float(np.sum(values))
        log1p_mean = float(np.log1p(raw_mean))
        if mode.endswith("sum"):
            q = raw_sum
        elif mode.endswith("log1p_mean"):
            q = log1p_mean
        else:
            q = raw_mean
        return AggregationResult(
            q=q,
            raw_mean=raw_mean,
            raw_sum=raw_sum,
            log1p_mean=log1p_mean,
            n_bins_used=len(overlaps),
        )

    def aggregate_gene_body(
        self,
        output: np.ndarray,
        gene_start: int,
        gene_end: int,
        mode: str = "gene_body_log1p_mean",
    ) -> AggregationResult | None:
        return self.aggregate_bins(
            output,
            self.genomic_interval_to_bins(gene_start, gene_end),
            mode=mode,
        )

    def aggregate_exons(
        self,
        output: np.ndarray,
        exon_intervals: Iterable[tuple[int, int]],
        mode: str = "exon_log1p_mean",
    ) -> AggregationResult | None:
        seen: dict[int, BinOverlap] = {}
        for start, end in exon_intervals:
            for overlap in self.genomic_interval_to_bins(start, end):
                prev = seen.get(overlap.bin_index)
                if prev is None or overlap.overlap_bp > prev.overlap_bp:
                    seen[overlap.bin_index] = overlap
        return self.aggregate_bins(output, list(seen.values()), mode=mode)

    def aggregate_tss_window(
        self,
        output: np.ndarray,
        tss: int,
        flank: int,
        mode: str = "tss_window_mean",
    ) -> AggregationResult | None:
        return self.aggregate_bins(
            output,
            self.genomic_interval_to_bins(tss - flank, tss + flank + 1),
            mode=mode,
        )
