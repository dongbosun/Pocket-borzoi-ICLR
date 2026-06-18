"""Shard helpers."""

from __future__ import annotations


def assign_even_shards(rows: list[dict], num_shards: int) -> list[list[dict]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    shards = [[] for _ in range(num_shards)]
    for i, row in enumerate(rows):
        shards[i % num_shards].append(row)
    return shards


def padded_shard_name(index: int) -> str:
    return f"shard_{index:03d}.parquet"
