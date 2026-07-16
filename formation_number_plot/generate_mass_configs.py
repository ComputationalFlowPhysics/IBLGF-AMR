#!/usr/bin/env python3

import argparse
from pathlib import Path
import re


TAU_VALUES = [
    "6.5",
]


def format_number(value):
    if isinstance(value, str):
        return value
    return "{:.8g}".format(value)


def filename_number(value):
    return format_number(value).replace(".", "p").replace("-", "m")


def blank_comments(text):
    def replace(match):
        return "".join("\n" if char == "\n" else " " for char in match.group(0))

    return re.sub(r"//[^\n]*|/\*.*?\*/", replace, text, flags=re.DOTALL)


def find_simulation_parameters_block(text):
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


def replace_assignment(block_text, key, value):
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

    raise KeyError("Required config key not found: {}".format(key))


def apply_parameter_updates(config_text, updates):
    start, end = find_simulation_parameters_block(config_text)
    block_text = config_text[start:end]
    for key, value in updates.items():
        block_text = replace_assignment(block_text, key, format_number(value))
    return config_text[:start] + block_text + config_text[end:]


def build_parameter_sets():
    return [{"b_f_tau": tau_value} for tau_value in TAU_VALUES]


def generate_configs(
    sample_config,
    output_dir,
    *,
    prefix
):
    template = sample_config.read_text()
    parameter_sets = build_parameter_sets()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for updates in parameter_sets:
        tau_value = updates["b_f_tau"]
        config_path = (
            output_dir /
            "{}_tau{}.cfg".format(prefix, filename_number(tau_value))
        )
        config_path.write_text(apply_parameter_updates(template, updates))
        generated.append((updates, config_path))

    return generated


def main():
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
        raise SystemExit("Sample config not found: {}".format(sample_config))

    generated = generate_configs(
        sample_config,
        args.output_dir.resolve(),
        prefix=args.prefix,
    )

    for updates, path in generated:
        print(
            "{} b_f_tau={}".format(
                path.name,
                format_number(updates["b_f_tau"]),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
