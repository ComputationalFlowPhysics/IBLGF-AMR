#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re


TAU_VALUES = [
    "45.0",
]


def format_number(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.8g}"


def filename_number(value: float | str) -> str:
    return format_number(value).replace(".", "p").replace("-", "m")


def blank_comments(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return "".join("\n" if char == "\n" else " " for char in match.group(0))

    return re.sub(r"//[^\n]*|/\*.*?\*/", replace, text, flags=re.DOTALL)


def find_simulation_parameters_block(text: str) -> tuple[int, int]:
    clean = blank_comments(text)
    match = re.search(r"\bsimulation_parameters\s*\{", clean)
    if match is None:
        raise ValueError("Could not find simulation_parameters block.")

    brace_index = clean.find("{", match.start())
    depth = 0
    for index in range(brace_index, len(clean)):
        char = clean[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return match.start(), index + 1

    raise ValueError("Unbalanced simulation_parameters block.")


def replace_assignment(block_text: str, key: str, value: str) -> str:
    pattern = re.compile(
        rf"(^[ \t]*{re.escape(key)}[ \t]*=[ \t]*)([^;]*)(;)",
        flags=re.MULTILINE,
    )

    if pattern.search(block_text):
        return pattern.sub(
            lambda match: f"{match.group(1)}{value}{match.group(3)}",
            block_text,
            count=1,
        )

    raise KeyError(f"Required config key not found: {key}")


def apply_parameter_updates(config_text: str, updates: dict[str, str]) -> str:
    start, end = find_simulation_parameters_block(config_text)
    block_text = config_text[start:end]
    for key, value in updates.items():
        block_text = replace_assignment(block_text, key, format_number(value))
    return config_text[:start] + block_text + config_text[end:]


def build_parameter_sets() -> list[dict[str, str]]:
    return [{"b_f_tau": tau_value} for tau_value in TAU_VALUES]


def generate_configs(
    sample_config: Path,
    output_dir: Path,
    *,
    prefix: str,
) -> list[tuple[dict[str, str], Path]]:
    template = sample_config.read_text()
    parameter_sets = build_parameter_sets()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[dict[str, str], Path]] = []
    for updates in parameter_sets:
        tau_value = updates["b_f_tau"]
        config_path = (
            output_dir /
            f"{prefix}_tau{filename_number(tau_value)}.cfg"
        )
        config_path.write_text(apply_parameter_updates(template, updates))
        generated.append((updates, config_path))

    return generated


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ns_amr_lgf2D configs for a b_f_tau sweep."
        )
    )
    parser.add_argument(
        "sample_config",
        type=Path,
        help="Sample config file to copy from.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=Path(__file__).resolve().parent / "mass_configs",
        type=Path,
        help="Directory for generated configs.",
    )
    parser.add_argument(
        "--prefix",
        default="config",
        help="Generated config filename prefix.",
    )
    args = parser.parse_args()

    sample_config = args.sample_config.resolve()
    if not sample_config.is_file():
        raise SystemExit(f"Sample config not found: {sample_config}")

    generated = generate_configs(
        sample_config,
        args.output_dir.resolve(),
        prefix=args.prefix,
    )

    for updates, path in generated:
        print(
            f"{path.name} "
            f"b_f_tau={format_number(updates['b_f_tau'])}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
