"""Project storage paths.

The repository is intentionally kept lightweight in $HOME. Generated datasets,
teacher caches, logs, plots, checkpoints, and downloaded assets live on /extra.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_STORAGE_ROOT = Path("/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR")
STORAGE_ROOT_ENV = "POCKET_BORZOI_STORAGE_ROOT"


def storage_root() -> Path:
    return Path(os.environ.get(STORAGE_ROOT_ENV, DEFAULT_STORAGE_ROOT)).expanduser()


def dataset_dir(*parts: str) -> Path:
    return storage_root().joinpath("dataset", *parts)


def results_dir(*parts: str) -> Path:
    return storage_root().joinpath("results", *parts)


def checkpoints_dir(*parts: str) -> Path:
    return storage_root().joinpath("checkpoints", *parts)


def interim_dir(*parts: str) -> Path:
    return storage_root().joinpath("interim", *parts)


def logs_dir(*parts: str) -> Path:
    return storage_root().joinpath("logs", *parts)


def plots_dir(*parts: str) -> Path:
    return storage_root().joinpath("plots", *parts)


def external_dir(*parts: str) -> Path:
    return dataset_dir("external", *parts)


def teacher_cache_dir(*parts: str) -> Path:
    return interim_dir("teacher_cache", *parts)


def manifest_dir(*parts: str) -> Path:
    return dataset_dir("manifests", *parts)


def variants_dir(*parts: str) -> Path:
    return dataset_dir("variants", *parts)


def processed_dir(*parts: str) -> Path:
    return interim_dir("processed", *parts)
