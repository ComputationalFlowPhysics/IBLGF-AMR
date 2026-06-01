from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib import import_module
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Mapping, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runs_pybind_root() -> Path:
    root = _repo_root() / "runs_pybind"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _mpi_rank() -> int | None:
    for name in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return None


def _mpi_job_id() -> str | None:
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


def _shared_generated_config_path(template: Path, text: str) -> Path:
    job_id = _mpi_job_id() or "mpi"
    digest = hashlib.sha256(
        f"{template.resolve()}::{text}".encode("utf-8")
    ).hexdigest()[:12]
    run_dir = _runs_pybind_root() / f"{template.stem}_pybind_{job_id}_{digest}"
    return run_dir / template.name


def _wait_for_file(path: Path, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for generated config: {path}")


def _cleanup_generated_config(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return

    try:
        parent = path.parent
        if parent.parent == _runs_pybind_root():
            parent.rmdir()
    except OSError:
        pass


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
        rank = _mpi_rank()
        if rank is not None:
            generated_path = _shared_generated_config_path(template, text)
            if rank == 0:
                generated_path.parent.mkdir(parents=True, exist_ok=True)
                generated_path.write_text(text)
            else:
                _wait_for_file(generated_path)
            return generated_path

        run_dir = Path(
            tempfile.mkdtemp(
                prefix=f"{template.stem}_pybind_",
                dir=str(_runs_pybind_root()),
            )
        )
        generated_path = run_dir / template.name
        generated_path.write_text(text)
        return generated_path

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
    generated_temp = output_path is None
    rank = _mpi_rank()
    config_path = prepare_config(
        template_path,
        output_path,
        simulation_overrides=simulation_overrides,
        vortex_overrides=vortex_overrides,
    )
    try:
        return run(config_path, cli_overrides=cli_overrides)
    finally:
        if generated_temp and (rank is None or rank == 0):
            _cleanup_generated_config(Path(config_path))


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
