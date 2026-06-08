from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys

from iblgf import ns_amr_2d
from iblgf._config import mpi_rank
from iblgf._config import parse_override_items
from iblgf._config import stage_config


@contextmanager
def capture_rank0_output(run_dir: Path):
    if mpi_rank() not in (None, 0):
        yield
        return

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    stdout_path.touch()
    stderr_path.touch()

    saved_stdout = os.dup(sys.stdout.fileno())
    saved_stderr = os.dup(sys.stderr.fileno())

    with stdout_path.open("a") as stdout_file, stderr_path.open("a") as stderr_file:
        try:
            os.dup2(stdout_file.fileno(), sys.stdout.fileno())
            os.dup2(stderr_file.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(saved_stdout, sys.stdout.fileno())
            os.dup2(saved_stderr, sys.stderr.fileno())
            os.close(saved_stdout)
            os.close(saved_stderr)


def emit_result(run_dir: Path, *values: object) -> None:
    lines = [str(value) for value in values]
    if mpi_rank() in (None, 0):
        with (run_dir / "stdout.log").open("a") as stream:
            for line in lines:
                stream.write(f"{line}\n")
    for line in lines:
        print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the NS AMR 2D pybind example in a staged runs_pybind directory."
    )
    parser.add_argument("config_path", help="Path to the NS AMR 2D config file.")
    parser.add_argument(
        "--simulation-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a value in simulation_parameters. Repeat for multiple keys.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        simulation_overrides = parse_override_items(args.simulation_override)
    except ValueError as exc:
        raise SystemExit(str(exc))

    block_overrides = []
    if simulation_overrides:
        block_overrides.append(("simulation_parameters", 0, simulation_overrides))

    staged_config = stage_config(
        args.config_path,
        block_overrides=block_overrides,
    )
    run_dir = staged_config.parent
    with capture_rank0_output(run_dir):
        result = ns_amr_2d.run(staged_config)
    emit_result(
        run_dir,
        result.measured_linf_error,
        result.difference,
        result.fine_u2_linf_error,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
