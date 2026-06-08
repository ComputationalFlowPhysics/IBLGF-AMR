from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Callable, Mapping, Sequence

BlockOverride = tuple[str, int, Mapping[str, object]]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runs_pybind_root() -> Path:
    root = repo_root() / "runs_pybind"
    root.mkdir(parents=True, exist_ok=True)
    return root


def mpi_rank() -> int | None:
    for name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return None


def mpi_job_id() -> str | None:
    for name in (
        "PMIX_NAMESPACE",
        "OMPI_MCA_ess_base_jobid",
        "PMI_JOBID",
        "SLURM_JOB_ID",
    ):
        value = os.environ.get(name)
        if value:
            return value
    return None


def shared_generated_config_path(template: Path, text: str) -> Path:
    job_id = mpi_job_id() or "mpi"
    digest = hashlib.sha256(
        f"{template.resolve()}::{text}".encode("utf-8")
    ).hexdigest()[:12]
    run_dir = runs_pybind_root() / f"{template.stem}_pybind_{job_id}_{digest}"
    return run_dir / template.name


def wait_for_file(path: Path, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for generated config: {path}")


def cleanup_generated_config(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return

    try:
        parent = path.parent
        if parent.parent == runs_pybind_root():
            parent.rmdir()
    except OSError:
        pass


def create_ephemeral_generated_config(template: Path, text: str) -> Path:
    rank = mpi_rank()
    if rank is not None:
        generated_path = shared_generated_config_path(template, text)
        if rank == 0:
            generated_path.parent.mkdir(parents=True, exist_ok=True)
            generated_path.write_text(text)
        else:
            wait_for_file(generated_path)
        return generated_path

    run_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{template.stem}_pybind_",
            dir=str(runs_pybind_root()),
        )
    )
    generated_path = run_dir / template.name
    generated_path.write_text(text)
    return generated_path


def render_config_text(
    template_path: str | Path,
    *,
    block_overrides: Sequence[BlockOverride] = (),
) -> tuple[Path, str]:
    template = Path(template_path)
    text = template.read_text()

    for block_name, block_index, overrides in block_overrides:
        text = update_named_block_assignments(
            text,
            block_name,
            overrides,
            block_index=block_index,
        )

    return template, text


def stage_config(
    template_path: str | Path,
    *,
    block_overrides: Sequence[BlockOverride] = (),
) -> Path:
    template, text = render_config_text(
        template_path,
        block_overrides=block_overrides,
    )
    staged_path = shared_generated_config_path(template, text)
    rank = mpi_rank()

    if rank is not None:
        if rank == 0:
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_text(text)
        else:
            wait_for_file(staged_path)
        return staged_path

    staged_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path.write_text(text)
    return staged_path


def prepare_config(
    template_path: str | Path,
    output_path: str | Path | None = None,
    *,
    block_overrides: Sequence[BlockOverride] = (),
) -> Path:
    template, text = render_config_text(
        template_path,
        block_overrides=block_overrides,
    )

    if output_path is None:
        return create_ephemeral_generated_config(template, text)

    output = Path(output_path)
    output.write_text(text)
    return output


def run_from_template(
    run_config: Callable[[str | Path, Sequence[str] | None], object],
    template_path: str | Path,
    output_path: str | Path | None = None,
    *,
    block_overrides: Sequence[BlockOverride] = (),
    cli_overrides: Sequence[str] | None = None,
):
    generated_temp = output_path is None
    rank = mpi_rank()
    config_path = prepare_config(
        template_path,
        output_path,
        block_overrides=block_overrides,
    )
    try:
        return run_config(config_path, cli_overrides=cli_overrides)
    finally:
        if generated_temp and (rank is None or rank == 0):
            cleanup_generated_config(Path(config_path))


def blank_comments(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return "".join("\n" if char == "\n" else " " for char in match.group(0))

    return re.sub(r"//[^\n]*|/\*.*?\*/", replace, text, flags=re.DOTALL)


def find_named_blocks(text: str, name: str) -> list[tuple[int, int]]:
    clean = blank_comments(text)
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


def update_named_block_assignments(
    text: str,
    block_name: str,
    overrides: Mapping[str, object],
    *,
    block_index: int,
) -> str:
    blocks = find_named_blocks(text, block_name)
    if block_index >= len(blocks):
        raise IndexError(
            f"Config only has {len(blocks)} `{block_name}` block(s), "
            f"but block index {block_index} was requested."
        )

    start, end = blocks[block_index]
    block_text = text[start:end]

    for key, value in overrides.items():
        block_text = replace_assignment(block_text, key, format_value(value))

    return text[:start] + block_text + text[end:]


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

    insert_at = block_text.rfind("}")
    if insert_at == -1:
        raise ValueError("Invalid config block: missing closing brace.")

    insertion = f"    {key}={value};\n"
    return block_text[:insert_at] + insertion + block_text[insert_at:]


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, (tuple, list)):
        return "(" + ",".join(format_value(item) for item in value) + ")"

    return str(value)


def parse_override_items(items: Sequence[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override `{item}`. Expected KEY=VALUE.")

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
