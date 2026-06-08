from __future__ import annotations

from functools import lru_cache
from importlib import import_module
import os
from pathlib import Path
import sys


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def binding_search_roots() -> list[Path]:
    roots: list[Path] = []

    env_build_dir = os.environ.get("IBLGF_PYBUILD_DIR")
    if env_build_dir:
        roots.append(Path(env_build_dir).resolve())

    roots.extend(
        path.resolve() for path in sorted(repo_root().glob("build*")) if path.is_dir()
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


def binding_candidate_dirs() -> list[Path]:
    candidates: list[Path] = []
    for root in binding_search_roots():
        candidates.extend(
            [
                root / "python" / "bindings",
                root / "bindings" / "python",
            ]
        )
    return candidates


@lru_cache(maxsize=None)
def load_bindings(module_name: str = "iblgf_bindings"):
    try:
        return import_module(f".{module_name}", package=__package__)
    except ModuleNotFoundError:
        pass

    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        pass

    for candidate_dir in binding_candidate_dirs():
        if not candidate_dir.exists():
            continue

        candidate_str = str(candidate_dir)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

        try:
            return import_module(module_name)
        except ModuleNotFoundError:
            continue

    search_locations = "\n".join(
        f"- {candidate}" for candidate in binding_candidate_dirs()
    )
    raise ModuleNotFoundError(
        f"The `{module_name}` extension is not available. "
        "Build the project with `-DIBLGF_BUILD_PYTHON=ON`. "
        "Searched these locations:\n"
        f"{search_locations}"
    )
