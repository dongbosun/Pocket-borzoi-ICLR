#!/usr/bin/env python
"""Prepare local Borzoi asset paths and write an asset config."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import _path  # noqa: F401
import yaml


REPOS = {
    "baskerville": "https://github.com/calico/baskerville.git",
    "borzoi": "https://github.com/calico/borzoi.git",
    "westminster": "https://github.com/calico/westminster.git",
}


def run_or_print(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def copy_or_download(src: str, dst: Path, dry_run: bool) -> Path:
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
    src_path = Path(src)
    if src_path.exists():
        print(f"copy {src_path} -> {dst}")
        if not dry_run:
            shutil.copy2(src_path, dst)
        return dst
    if src.startswith("gs://"):
        if shutil.which("gsutil") is None:
            raise RuntimeError("gsutil is required for gs:// assets but was not found on PATH.")
        run_or_print(["gsutil", "cp", src, str(dst)], dry_run)
        return dst
    if src.startswith(("http://", "https://")):
        print(f"download {src} -> {dst}")
        if not dry_run:
            urllib.request.urlretrieve(src, dst)
        return dst
    raise ValueError(f"Asset source is neither a local path nor URL: {src}")


def clone_repo(name: str, work_dir: Path, force: bool, dry_run: bool) -> Path:
    dst = work_dir / name
    if dst.exists() and not force:
        print(f"exists, not overwriting: {dst}")
        return dst
    if dst.exists() and force and not dry_run:
        raise RuntimeError(f"Refusing to delete existing repo {dst}. Move it manually, then retry.")
    run_or_print(["git", "clone", REPOS[name], str(dst)], dry_run)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="external")
    parser.add_argument("--download-mini-k562", action="store_true")
    parser.add_argument("--download-official-repos", action="store_true")
    parser.add_argument("--include-westminster", action="store_true")
    parser.add_argument("--install-editable", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-config", default="configs/borzoi_assets.local.yaml")
    parser.add_argument("--k562-fold0-weights")
    parser.add_argument("--k562-fold1-weights")
    parser.add_argument("--k562-targets")
    parser.add_argument("--k562-params")
    parser.add_argument("--hg38-fasta")
    parser.add_argument("--gencode-gtf")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    assets_dir = work_dir / "assets" / "k562_mini"
    config: dict[str, str | None] = {
        "borzoi_repo": None,
        "baskerville_repo": None,
        "westminster_repo": None,
        "k562_weights_fold0": args.k562_fold0_weights,
        "k562_weights_fold1": args.k562_fold1_weights,
        "k562_targets": args.k562_targets,
        "k562_params": args.k562_params,
        "hg38_fasta": args.hg38_fasta,
        "gencode_gtf": args.gencode_gtf,
    }

    if args.download_official_repos:
        if not args.dry_run:
            work_dir.mkdir(parents=True, exist_ok=True)
        for name in ("baskerville", "borzoi"):
            config[f"{name}_repo"] = str(clone_repo(name, work_dir, args.force, args.dry_run))
        if args.include_westminster:
            config["westminster_repo"] = str(clone_repo("westminster", work_dir, args.force, args.dry_run))
        if args.install_editable:
            for key in ("baskerville_repo", "borzoi_repo", "westminster_repo"):
                if config.get(key):
                    run_or_print([sys.executable, "-m", "pip", "install", "-e", config[key]], args.dry_run)

    if args.download_mini_k562:
        explicit = {
            "k562_weights_fold0": args.k562_fold0_weights,
            "k562_weights_fold1": args.k562_fold1_weights,
            "k562_targets": args.k562_targets,
            "k562_params": args.k562_params,
        }
        missing = [key for key, value in explicit.items() if not value]
        if missing:
            raise SystemExit(
                "Mini-Borzoi K562 asset URLs/paths were not hard-coded because official locations "
                f"can change. Provide explicit sources for: {', '.join(missing)}"
            )
        names = {
            "k562_weights_fold0": "fold0_weights",
            "k562_weights_fold1": "fold1_weights",
            "k562_targets": "targets.txt",
            "k562_params": "params.json",
        }
        for key, src in explicit.items():
            assert src is not None
            dst = assets_dir / names[key]
            config[key] = str(copy_or_download(src, dst, args.dry_run))

    out = Path(args.out_config)
    print(f"write {out}")
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
