from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

from ._bindings import load_bindings
from ._config import parse_override_items
from ._config import prepare_config as prepare_config_with_overrides
from ._config import run_from_template as run_from_template_with_overrides


def run(config_path: str | Path, cli_overrides: Sequence[str] | None = None):
    bindings = load_bindings()
    return bindings.ns_amr_2d.run(
        str(Path(config_path)), list(cli_overrides or [])
    )


def prepare_config(
    template_path: str | Path,
    output_path: str | Path | None = None,
    *,
    simulation_overrides: Mapping[str, object] | None = None,
) -> Path:
    block_overrides = []
    if simulation_overrides:
        block_overrides.append(("simulation_parameters", 0, simulation_overrides))

    return prepare_config_with_overrides(
        template_path,
        output_path,
        block_overrides=block_overrides,
    )


def run_from_template(
    template_path: str | Path,
    *,
    output_path: str | Path | None = None,
    simulation_overrides: Mapping[str, object] | None = None,
    cli_overrides: Sequence[str] | None = None,
):
    block_overrides = []
    if simulation_overrides:
        block_overrides.append(("simulation_parameters", 0, simulation_overrides))

    return run_from_template_with_overrides(
        run,
        template_path,
        output_path,
        block_overrides=block_overrides,
        cli_overrides=cli_overrides,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 2D NS AMR LGF pybind interface."
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

    if simulation_overrides:
        result = run_from_template(
            args.config_path,
            simulation_overrides=simulation_overrides,
        )
    else:
        result = run(args.config_path)

    print(result.measured_linf_error)
    print(result.difference)
    print(result.fine_u2_linf_error)
    return 0
