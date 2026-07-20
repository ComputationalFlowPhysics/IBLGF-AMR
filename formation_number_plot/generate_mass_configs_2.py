#!/usr/bin/env python3

import argparse
from pathlib import Path
import re


FREQ_VALUES = [
    "0.05", "0.0333", "0.025", "0.0167",
]
# 20, 30, 40, 60

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


def replace_or_insert_assignment(block_text, key, value):
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

    anchor = re.compile(
        r"(^[ \t]*b_f_tau[ \t]*=[ \t]*[^;]*;\n?)",
        flags=re.MULTILINE,
    )
    match = anchor.search(block_text)
    if match is not None:
        indent = re.match(r"[ \t]*", match.group(1)).group(0)
        insertion = "{}{}{}={};\n".format(match.group(1), indent, key, value)
        return block_text[:match.start()] + insertion + block_text[match.end():]

    raise KeyError("Required config key or insertion anchor not found: {}".format(key))


def apply_parameter_updates(config_text, updates):
    start, end = find_simulation_parameters_block(config_text)
    block_text = config_text[start:end]
    for key, value in updates.items():
        block_text = replace_or_insert_assignment(block_text, key, format_number(value))
    return config_text[:start] + block_text + config_text[end:]


def build_parameter_sets():
    return [{"b_f_freq": freq_value} for freq_value in FREQ_VALUES]


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
        freq_value = updates["b_f_freq"]
        config_path = (
            output_dir /
            "{}_freq{}.cfg".format(prefix, filename_number(freq_value))
        )
        config_path.write_text(apply_parameter_updates(template, updates))
        generated.append((updates, config_path))

    return generated


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate ns_amr_lgf2D configs for a b_f_freq sweep."
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
            "{} b_f_freq={}".format(
                path.name,
                format_number(updates["b_f_freq"]),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
