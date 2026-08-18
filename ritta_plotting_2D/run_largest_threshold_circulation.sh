#!/usr/bin/env bash
# Analyze a 2D sweep with the independent largest-enclosed 2% threshold method.
# Usage: run_largest_threshold_circulation.sh SWEEP [WORKERS] [STRIDE] [CASE...]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
sweep_input="${1:-}"
workers="${2:-64}"
stride="${3:-1}"

if [[ -z "$sweep_input" ]]; then
    echo "Usage: $(basename "$0") SWEEP [WORKERS] [STRIDE] [CASE...]" >&2
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

python ritta_plotting_2D/largest_threshold_circulation.py \
    "$sweep_input" \
    ritta_vortex_identification/configs/default.toml \
    --threshold-fraction 0.02 \
    --workers "$workers" \
    --stride "$stride" \
    --resume \
    "${case_args[@]}"
