#!/usr/bin/env bash
# Analyze a 2D sweep and create upper-half circulation plots and GIFs.
# Usage: run_upper_half_circulation.sh SWEEP [WORKERS] [ANALYSIS_STRIDE] [GIF_STRIDE] [CASE...]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
sweep_input="${1:-}"
workers="${2:-64}"
analysis_stride="${3:-1}"
gif_stride="${4:-5}"

if [[ -z "$sweep_input" ]]; then
    echo "Usage: $(basename "$0") SWEEP [WORKERS] [ANALYSIS_STRIDE] [GIF_STRIDE] [CASE...]" >&2
    exit 2
fi
for value_name in workers analysis_stride gif_stride; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: $value_name must be a positive integer; got $value" >&2
        exit 2
    fi
done

shift
[[ $# -gt 0 ]] && shift
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

python ritta_plotting_2D/upper_half_circulation.py \
    "$sweep_input" \
    ritta_vortex_identification/configs/default.toml \
    --center-threshold-fraction 0.4 \
    --workers "$workers" \
    --analysis-stride "$analysis_stride" \
    --gif-stride "$gif_stride" \
    --y-limits -2.5 2.5 \
    --resume \
    "${case_args[@]}"
