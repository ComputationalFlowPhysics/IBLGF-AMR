from __future__ import annotations

from functools import lru_cache
from importlib import import_module
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Mapping, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _binding_search_roots() -> list[Path]:
    repo_root = _repo_root()
    roots: list[Path] = []

    env_build_dir = os.environ.get("IBLGF_PYBUILD_DIR")
    if env_build_dir:
        roots.append(Path(env_build_dir).resolve())

    roots.extend(
        path.resolve()
        for path in sorted(repo_root.glob("build*"))
        if path.is_dir()
    )

    fallback_tmp = Path("/tmp/build-pybind")
    if fallback_tmp.is_dir():
        roots.append(fallback_tmp.resolve())

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique_roots.append(root)
    return unique_roots


def _binding_candidate_dirs() -> list[Path]:
    candidates: list[Path] = []
    for root in _binding_search_roots():
        candidates.extend(
            [
                root / "python" / "bindings",
                root / "bindings" / "python",
            ]
        )
    return candidates


@lru_cache(maxsize=1)
def _load_bindings():
    try:
        return import_module("iblgf_bindings")
    except ModuleNotFoundError:
        pass

    for candidate_dir in _binding_candidate_dirs():
        if not candidate_dir.exists():
            continue

        candidate_str = str(candidate_dir)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

        try:
            return import_module("iblgf_bindings")
        except ModuleNotFoundError:
            continue

    search_locations = "\n".join(
        f"- {candidate}" for candidate in _binding_candidate_dirs()
    )
    raise ModuleNotFoundError(
        "The `iblgf_bindings` extension is not available. "
        "Build the project with `-DIBLGF_BUILD_PYTHON=ON`. "
        "Searched these locations:\n"
        f"{search_locations}"
    )


def run(config_path: str | Path, cli_overrides: Sequence[str] | None = None):
    bindings = _load_bindings()
    return bindings.poisson.run(str(Path(config_path)), list(cli_overrides or []))


def prepare_config(
    template_path: str | Path,
    output_path: str | Path | None = None,
    *,
    simulation_overrides: Mapping[str, object] | None = None,
    vortex_overrides: Sequence[Mapping[str, object]] | None = None,
) -> Path:
    template = Path(template_path)
    text = template.read_text()

    if simulation_overrides:
        text = _update_named_block_assignments(
            text, "simulation_parameters", simulation_overrides, block_index=0
        )

    if vortex_overrides:
        for index, overrides in enumerate(vortex_overrides):
            text = _update_named_block_assignments(
                text, "vortex", overrides, block_index=index
            )

    if output_path is None:
        fd, temp_name = tempfile.mkstemp(
            prefix=f"{template.stem}_pybind_",
            suffix=template.suffix or ".cfg",
            dir=str(template.parent),
        )
        os.close(fd)
        Path(temp_name).write_text(text)
        return Path(temp_name)

    output = Path(output_path)
    output.write_text(text)
    return output


def run_from_template(
    template_path: str | Path,
    *,
    output_path: str | Path | None = None,
    simulation_overrides: Mapping[str, object] | None = None,
    vortex_overrides: Sequence[Mapping[str, object]] | None = None,
    cli_overrides: Sequence[str] | None = None,
):
    config_path = prepare_config(
        template_path,
        output_path,
        simulation_overrides=simulation_overrides,
        vortex_overrides=vortex_overrides,
    )
    return run(config_path, cli_overrides=cli_overrides)


def _blank_comments(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return "".join("\n" if char == "\n" else " " for char in match.group(0))

    return re.sub(r"//[^\n]*|/\*.*?\*/", replace, text, flags=re.DOTALL)


def _find_named_blocks(text: str, name: str) -> list[tuple[int, int]]:
    clean = _blank_comments(text)
    pattern = re.compile(rf"\b{re.escape(name)}\s*\{{")
    spans: list[tuple[int, int]] = []

    for match in pattern.finditer(clean):
        brace_index = clean.find("{", match.start())
        depth = 0
        end_index = None

        for index in range(brace_index, len(clean)):
            char = clean[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_index = index + 1
                    break

        if end_index is None:
            raise ValueError(f"Unbalanced block for `{name}` in config template.")

        spans.append((match.start(), end_index))

    return spans


def _update_named_block_assignments(
    text: str,
    block_name: str,
    overrides: Mapping[str, object],
    *,
    block_index: int,
) -> str:
    blocks = _find_named_blocks(text, block_name)
    if block_index >= len(blocks):
        raise IndexError(
            f"Config only has {len(blocks)} `{block_name}` block(s), "
            f"but block index {block_index} was requested."
        )

    start, end = blocks[block_index]
    block_text = text[start:end]

    for key, value in overrides.items():
        block_text = _replace_assignment(block_text, key, _format_value(value))

    return text[:start] + block_text + text[end:]


def _replace_assignment(block_text: str, key: str, value: str) -> str:
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

    insert_at = block_text.rfind("}")
    if insert_at == -1:
        raise ValueError("Invalid config block: missing closing brace.")

    insertion = f"    {key}={value};\n"
    return block_text[:insert_at] + insertion + block_text[insert_at:]


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, (tuple, list)):
        return "(" + ",".join(_format_value(item) for item in value) + ")"

    return str(value)
