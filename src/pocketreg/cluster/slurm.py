"""Small helpers for SLURM submission wrappers."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class SubmissionRecord:
    command: list[str]
    dry_run: bool
    job_id: str | None
    submitted_at_utc: str
    stdout: str
    stderr: str


def build_sbatch_command(
    job: Path,
    array: str | None = None,
    export: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = ["sbatch"]
    if array:
        cmd.append(f"--array={array}")
    if export:
        cmd.append(f"--export={export}")
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(job))
    return cmd


def parse_sbatch_job_id(stdout: str) -> str | None:
    match = re.search(r"Submitted batch job\s+(\d+)", stdout)
    return match.group(1) if match else None


def submit_sbatch(
    command: list[str],
    dry_run: bool,
    submissions_dir: Path,
) -> SubmissionRecord:
    stdout = ""
    stderr = ""
    job_id = None
    if not dry_run:
        proc = subprocess.run(command, check=False, text=True, capture_output=True)
        stdout = proc.stdout
        stderr = proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(
                f"sbatch failed with exit code {proc.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        job_id = parse_sbatch_job_id(stdout)
    record = SubmissionRecord(
        command=command,
        dry_run=dry_run,
        job_id=job_id,
        submitted_at_utc=datetime.now(timezone.utc).isoformat(),
        stdout=stdout,
        stderr=stderr,
    )
    submissions_dir.mkdir(parents=True, exist_ok=True)
    suffix = job_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    out_path = submissions_dir / f"submission_{suffix}.json"
    out_path.write_text(json.dumps(asdict(record), indent=2) + "\n")
    return record
