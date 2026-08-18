#!/bin/bash
# Apply the 2D vortex-identification method to 3D z slices and combine cases.
# Usage: sbatch run_slice_vortex_identification_rm.sbatch SWEEP [STRIDE] [WORKERS] [CASE...]

#SBATCH -A mch250004p
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH -t 48:00:00
#SBATCH -J slice-vortex3d
#SBATCH -o ritta_plotting_3D/slurm-slice-vortex3d-%j.out
#SBATCH -e ritta_plotting_3D/slurm-slice-vortex3d-%j.err

set -euo pipefail

repo_root="/ocean/projects/mch250004p/mchoi10/IBLGF-AMR"
submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
sweep_input="${1:-}"
stride="${2:-1}"
workers="${3:-32}"
minimum_peak_vorticity="${MINIMUM_PEAK_VORTICITY:-0.001}"

if [[ -z "$sweep_input" ]]; then
    echo "Usage: sbatch $0 SWEEP [STRIDE] [WORKERS] [CASE...]" >&2
    exit 2
fi
shift
[[ $# -gt 0 ]] && shift
[[ $# -gt 0 ]] && shift
cases=("$@")

for value_name in stride workers; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: $value_name must be a positive integer; got $value" >&2
        exit 2
    fi
done
if [[ "$workers" -gt 128 ]]; then
    echo "Error: WORKERS cannot exceed the 128 allocated RM cores." >&2
    exit 2
fi

if [[ "$sweep_input" = /* ]]; then
    sweep_dir="$sweep_input"
elif [[ -d "$PWD/$sweep_input" ]]; then
    # Direct execution inside an interactive allocation should follow the
    # current shell directory, not the allocation's older SLURM_SUBMIT_DIR.
    sweep_dir="$PWD/$sweep_input"
else
    sweep_dir="$submit_dir/$sweep_input"
fi
if [[ ! -d "$sweep_dir" ]]; then
    echo "Sweep folder not found: $sweep_dir" >&2
    exit 2
fi
sweep_dir="$(cd "$sweep_dir" && pwd)"
sweep_name="$(basename "$sweep_dir")"

cd "$repo_root"
source ritta_vortex_identification/open_viewer_venv.sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

config="ritta_vortex_identification/configs/default.toml"
output_dir="ritta_plotting_3D/outputs/${sweep_name}_slice_vortex_identification"
case_args=()
if [[ ${#cases[@]} -gt 0 ]]; then
    case_args=(--cases "${cases[@]}")
fi

echo "3D slice vortex-identification job"
echo "  Job:       $SLURM_JOB_ID"
echo "  Sweep:     $sweep_dir"
echo "  Stride:    $stride"
echo "  Workers:   $workers independent frame processes"
echo "  Slice:     z=1e-6"
echo "  Min peak:  omega >= $minimum_peak_vorticity"
echo "  Config:    $repo_root/$config"
echo "  Output:    $repo_root/$output_dir"
if [[ ${#cases[@]} -gt 0 ]]; then
    echo "  Cases:     ${cases[*]}"
fi
echo

python ritta_plotting_3D/slice_vortex_identification.py \
    "$sweep_dir" \
    "$config" \
    --stride "$stride" \
    --workers "$workers" \
    --slice-z 1e-6 \
    --minimum-peak-vorticity "$minimum_peak_vorticity" \
    --output-dir "$output_dir" \
    --resume \
    "${case_args[@]}"

python ritta_vortex_identification/plot_combined_time_series.py \
    "$output_dir/datasets.toml" \
    "$config" \
    --legend-by tau \
    --output-dir "$output_dir"

echo "Combined circulation plot: $repo_root/$output_dir/combined_circulation_vs_time.png"
