#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/ritta-vortex-matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

usage() {
    echo "Usage: $(basename "$0") RUN_FOLDER CONFIG_FILE"
    echo
    echo "Runs vortex-identification stages 01 through 04 in order."
    echo "Stages 01-03 run without previews; Stage 04 opens the metrics preview prompt."
}

if [[ $# -ne 2 ]]; then
    usage >&2
    exit 1
fi

run_folder="$1"
config_file="$2"

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable not found: $python_bin" >&2
    echo "Source open_viewer_venv.sh first, or set PYTHON_BIN." >&2
    exit 1
fi

echo "Stage 01/04: finding h-maxima"
"$python_bin" "$script_dir/01_find_hmaxima.py" "$run_folder" "$config_file" --no-preview

echo
echo "Stage 02/04: making candidate regions"
"$python_bin" "$script_dir/02_make_regions.py" "$run_folder" "$config_file" --no-preview

echo
echo "Stage 03/04: fitting vortices"
"$python_bin" "$script_dir/03_fit_vortices.py" "$run_folder" "$config_file" --no-preview

echo
echo "Stage 04/04: calculating circulation and measured centers"
"$python_bin" "$script_dir/04_positive_vortex_metrics.py" "$run_folder" "$config_file"

echo
echo "Vortex fitting and positive-vortex measurements complete."
