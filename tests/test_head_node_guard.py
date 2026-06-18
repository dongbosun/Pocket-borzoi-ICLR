from __future__ import annotations

import os
import unittest

import _bootstrap  # noqa: F401
from pocketreg.cluster.safety import REFUSAL_TEMPLATE, assert_compute_context


class HeadNodeGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old = os.environ.pop("SLURM_JOB_ID", None)

    def tearDown(self) -> None:
        if self.old is not None:
            os.environ["SLURM_JOB_ID"] = self.old
        else:
            os.environ.pop("SLURM_JOB_ID", None)

    def test_raises_without_slurm(self) -> None:
        with self.assertRaisesRegex(RuntimeError, REFUSAL_TEMPLATE.format(task_name="x")):
            assert_compute_context("x")

    def test_passes_toy(self) -> None:
        assert_compute_context("x", toy=True)

    def test_passes_allow_local(self) -> None:
        assert_compute_context("x", allow_local=True)

    def test_passes_slurm(self) -> None:
        os.environ["SLURM_JOB_ID"] = "123"
        assert_compute_context("x")


if __name__ == "__main__":
    unittest.main()
