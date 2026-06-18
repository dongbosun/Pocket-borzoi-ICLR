"""Safety guards for heavy cluster jobs."""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


REFUSAL_TEMPLATE = (
    "Refusing to run heavy task {task_name} on login/head node. "
    "Submit via sbatch or pass --allow-local only for debugging."
)


def is_slurm_job() -> bool:
    """Return True when the current process is running inside a SLURM job."""

    return "SLURM_JOB_ID" in os.environ


def assert_compute_context(
    task_name: str,
    allow_local: bool = False,
    toy: bool = False,
) -> None:
    """Refuse heavy execution on a login/head node unless explicitly allowed."""

    if toy:
        return
    if allow_local:
        LOGGER.warning(
            "Running heavy task %s outside SLURM because --allow-local was set.",
            task_name,
        )
        return
    if not is_slurm_job():
        raise RuntimeError(REFUSAL_TEMPLATE.format(task_name=task_name))


def assert_not_head_node(
    allow_local: bool,
    toy: bool,
    task_name: str,
) -> None:
    """Compatibility wrapper requested by the MVP spec."""

    assert_compute_context(task_name=task_name, allow_local=allow_local, toy=toy)


@dataclass(frozen=True)
class ClusterContext:
    hostname: str
    slurm_job_id: str | None
    slurm_array_task_id: str | None
    cuda_visible_devices: str | None


def get_cluster_context() -> ClusterContext:
    return ClusterContext(
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


def print_cluster_context() -> None:
    """Print concise context at script startup."""

    ctx = get_cluster_context()
    print(f"hostname={ctx.hostname}")
    print(f"SLURM_JOB_ID={ctx.slurm_job_id or ''}")
    print(f"SLURM_ARRAY_TASK_ID={ctx.slurm_array_task_id or ''}")
    print(f"CUDA_VISIBLE_DEVICES={ctx.cuda_visible_devices or ''}")
