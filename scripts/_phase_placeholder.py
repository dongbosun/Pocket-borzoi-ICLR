#!/usr/bin/env python
"""Shared placeholder for phases not implemented yet."""

from __future__ import annotations

import argparse

import _path  # noqa: F401

from pocketreg.cluster.safety import assert_compute_context, print_cluster_context


def main(task_name: str, heavy: bool) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("args", nargs="*")
    args, _ = parser.parse_known_args()
    print_cluster_context()
    if heavy:
        assert_compute_context(task_name, allow_local=args.allow_local, toy=args.toy or args.dry_run)
    raise SystemExit(f"{task_name} is reserved for a later phase and is not implemented in Phase 0/1.")
