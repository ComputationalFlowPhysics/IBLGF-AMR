#!/usr/bin/env bash

# Locate the existing viewer virtual environment relative to this script.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="$(cd "$script_dir/../.." && pwd)/iblgf-viewer/IBLGF-viewer/.venv"
activate_script="$venv_dir/bin/activate"

if [[ ! -f "$activate_script" ]]; then
    echo "The existing virtual environment was not found at: $venv_dir" >&2
    return 1 2>/dev/null || exit 1
fi

source "$activate_script"
# Keep Matplotlib cache files out of the repository and home directory.
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/ritta-vortex-matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

# Install only when the reused environment is missing a required package.
if ! python -c 'import numpy, scipy, h5py, matplotlib' 2>/dev/null; then
    echo "Installing the standalone workflow dependencies into $venv_dir"
    python -m pip install numpy scipy h5py matplotlib
fi

echo "Activated: $venv_dir"
python -c 'import numpy, scipy, h5py, matplotlib; print("numpy", numpy.__version__, "| scipy", scipy.__version__, "| h5py", h5py.__version__, "| matplotlib", matplotlib.__version__)'

# Executing opens a subshell; sourcing leaves the current shell activated.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Opening an interactive Bash shell. Run 'exit' to leave this environment."
    exec bash -i
fi
