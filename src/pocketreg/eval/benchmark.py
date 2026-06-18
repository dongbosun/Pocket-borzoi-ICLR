"""Benchmark helpers for later student evaluation scripts."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TimerResult:
    seconds: float
    steps_per_second: float


def time_callable(fn, steps: int) -> TimerResult:
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    seconds = time.perf_counter() - start
    return TimerResult(seconds=seconds, steps_per_second=steps / seconds if seconds else 0.0)
