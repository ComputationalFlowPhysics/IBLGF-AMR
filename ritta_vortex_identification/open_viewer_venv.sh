#!/usr/bin/env bash

# Reuse the local viewer venv or the dedicated remote Conda environment.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "$script_dir/../.." && pwd)"

environment_candidates=()
if [[ -n "${RITTA_VORTEX_ENV:-}" ]]; then
    environment_candidates+=("$RITTA_VORTEX_ENV")
fi
environment_candidates+=(
    "$workspace_dir/iblgf-viewer/IBLGF-viewer/.venv"
    "$workspace_dir/conda-envs/ritta-vortex"
    "$script_dir/.venv"
)

environment_dir=""
for candidate in "${environment_candidates[@]}"; do
    if [[ -x "$candidate/bin/python" ]]; then
        environment_dir="$candidate"
        break
    fi
done

if [[ -z "$environment_dir" ]]; then
    echo "No Python environment was found." >&2
    echo "Set RITTA_VORTEX_ENV to its path, or create:" >&2
    echo "  $workspace_dir/conda-envs/ritta-vortex" >&2
    return 1 2>/dev/null || exit 1
fi

if [[ -f "$environment_dir/bin/activate" ]]; then
    source "$environment_dir/bin/activate"
else
    export PATH="$environment_dir/bin:$PATH"
    hash -r
fi

if ! python -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
    echo "Python 3.11 or newer is required. Found: $(python --version 2>&1)" >&2
    return 1 2>/dev/null || exit 1
fi

# Keep Matplotlib cache files out of the repository and home directory.
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/ritta-vortex-matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

# Install only when the reused environment is missing a required package.
if ! python -c 'import numpy, scipy, h5py, matplotlib' 2>/dev/null; then
    echo "Installing the standalone workflow dependencies into $environment_dir"
    if ! python -m pip install numpy scipy h5py matplotlib; then
        return 1 2>/dev/null || exit 1
    fi
fi

echo "Activated: $environment_dir"
python -c 'import numpy, scipy, h5py, matplotlib; print("numpy", numpy.__version__, "| scipy", scipy.__version__, "| h5py", h5py.__version__, "| matplotlib", matplotlib.__version__)'

# Executing opens a subshell; sourcing leaves the current shell activated.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Opening an interactive Bash shell. Run 'exit' to leave this environment."
    exec bash -i
fi
