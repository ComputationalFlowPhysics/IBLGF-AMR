#!/usr/bin/env bash
# Render GIFs from completed 2D largest-enclosed-threshold circulation shards.
# Usage: run_largest_threshold_gifs.sh ANALYSIS_OUTPUT [WORKERS] [STRIDE] [CASE...]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
analysis_output="${1:-}"
workers="${2:-64}"
stride="${3:-5}"

if [[ -z "$analysis_output" ]]; then
    echo "Usage: $(basename "$0") ANALYSIS_OUTPUT [WORKERS] [STRIDE] [CASE...]" >&2
    exit 2
fi
for value_name in workers stride; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: $value_name must be a positive integer; got $value" >&2
        exit 2
    fi
done

shift
[[ $# -gt 0 ]] && shift
[[ $# -gt 0 ]] && shift
cases=("$@")
case_args=()
if [[ ${#cases[@]} -gt 0 ]]; then
    case_args=(--cases "${cases[@]}")
fi

cd "$repo_root"
source ritta_vortex_identification/open_viewer_venv.sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python ritta_plotting_2D/plot_largest_threshold_gifs.py \
    "$analysis_output" \
    --workers "$workers" \
    --stride "$stride" \
    --resume \
    "${case_args[@]}"
