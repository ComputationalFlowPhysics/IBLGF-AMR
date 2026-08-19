"""Tests for the headless multi-run vortex-identification driver."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_all import select_batch_cases


class SelectBatchCasesTests(unittest.TestCase):
    def test_selects_requested_cases_in_command_line_order(self) -> None:
        runs = [Path("/runs/tau_1p0"), Path("/runs/tau_3p0"), Path("/runs/tau_5p0")]

        selected = select_batch_cases(runs, ["tau_5p0", "tau_1p0"])

        self.assertEqual(selected, [Path("/runs/tau_5p0"), Path("/runs/tau_1p0")])

    def test_rejects_a_missing_case(self) -> None:
        with self.assertRaisesRegex(ValueError, "tau_7p0.*not found"):
            select_batch_cases([Path("/runs/tau_1p0")], ["tau_7p0"])

    def test_rejects_duplicate_case_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            select_batch_cases(
                [Path("/runs/tau_1p0")],
                ["tau_1p0", "tau_1p0"],
            )


if __name__ == "__main__":
    unittest.main()
