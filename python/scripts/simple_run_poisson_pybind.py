from __future__ import annotations

import argparse
import ast
from iblgf import poisson
import sys


def _parse_override_items(items: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid override `{item}`. Expected KEY=VALUE."
            )

        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ValueError(f"Invalid override `{item}`. Missing key.")

        lower_value = raw_value.lower()
        if lower_value == "true":
            value: object = True
        elif lower_value == "false":
            value = False
        else:
            try:
                value = ast.literal_eval(raw_value)
            except (SyntaxError, ValueError):
                value = raw_value

        overrides[key] = value
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal Poisson pybind example with optional config overrides."
    )
    parser.add_argument("config_path", help="Path to the Poisson config file.")
    parser.add_argument(
        "--vortex-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a value in the first vortex block. "
            "Repeat for multiple keys, for example --vortex-override R=0.2 "
            "--vortex-override 'center=(0.0, 0.0, 0.05)'"
        ),
    )
    parser.add_argument(
        "--simulation-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a value in simulation_parameters. "
            "Repeat for multiple keys."
        ),
    )
    args = parser.parse_args()

    try:
        vortex_overrides = _parse_override_items(args.vortex_override)
        simulation_overrides = _parse_override_items(args.simulation_override)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if vortex_overrides or simulation_overrides:
        result = poisson.run_from_template(
            args.config_path,
            simulation_overrides=simulation_overrides or None,
            vortex_overrides=[vortex_overrides] if vortex_overrides else None,
        )
    else:
        result = poisson.run(args.config_path)

    print(result.measured_linf_error)
    print(result.difference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
