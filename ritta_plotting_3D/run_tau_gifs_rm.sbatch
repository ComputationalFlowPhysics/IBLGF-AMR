#!/bin/bash
# Render the circulation-slice and 2%-vorticity 3D GIFs for every tau run.
# Usage: sbatch ritta_plotting_3D/run_tau_gifs_rm.sbatch SWEEP_FOLDER [SLICE_STRIDE] [3D_STRIDE] [CASE_WORKERS] [CASE...]

#SBATCH -A mch250004p
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -J tau-gifs3d
#SBATCH -o ritta_plotting_3D/slurm-tau-gifs3d-%j.out
#SBATCH -e ritta_plotting_3D/slurm-tau-gifs3d-%j.err

set -euo pipefail

repo_root="/ocean/projects/mch250004p/mchoi10/IBLGF-AMR"
submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
sweep_input="${1:-}"
slice_stride="${2:-1}"
vorticity_stride="${3:-5}"
case_workers="${4:-${PARAVIEW_CASE_WORKERS:-6}}"
frame_workers="${PARAVIEW_FRAME_WORKERS:-1}"
gif_products="${GIF_PRODUCTS:-both}"
selected_cases=("${@:5}")

if [[ -z "$sweep_input" ]]; then
    echo "Usage: sbatch $0 SWEEP_FOLDER [SLICE_STRIDE] [3D_STRIDE] [CASE_WORKERS] [CASE...]" >&2
    exit 2
fi
for value_name in slice_stride vorticity_stride case_workers frame_workers; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: $value_name must be a positive integer; got $value" >&2
        exit 2
    fi
done
if [[ $((case_workers * frame_workers)) -gt 128 ]]; then
    echo "Error: CASE_WORKERS * PARAVIEW_FRAME_WORKERS cannot exceed 128." >&2
    echo "Got $case_workers * $frame_workers = $((case_workers * frame_workers))." >&2
    exit 2
fi
if [[ "$gif_products" != "both" && "$gif_products" != "vorticity" ]]; then
    echo "Error: GIF_PRODUCTS must be both or vorticity; got $gif_products" >&2
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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

log_dir="ritta_plotting_3D/outputs/${sweep_name}_gif_logs_${SLURM_JOB_ID}"
mkdir -p "$log_dir"

run_case() {
    local run_dir="$1"
    local run_name
    local circulation_output
    local vorticity_output
    local csv_path
    local log_path

    run_name="$(basename "$run_dir")"
    circulation_output="ritta_plotting_3D/outputs/${sweep_name}_${run_name}_circulation"
    vorticity_output="ritta_plotting_3D/outputs/${sweep_name}_${run_name}_vorticity_0p02"
    csv_path="$circulation_output/leading_vortex_circulation.csv"
    log_path="$log_dir/${run_name}.log"

    if [[ "$gif_products" == "both" && ! -s "$csv_path" ]]; then
        echo "[$run_name] Missing calculated circulation CSV: $csv_path" >&2
        return 1
    fi

    {
        if [[ "$gif_products" == "both" ]]; then
            echo "[$run_name] Rendering centered circulation-slice GIF from saved CSV values."
            ./ritta_plotting_3D/run_circulation_analysis.sh \
                "$run_dir" \
                "$slice_stride" \
                --render-from-csv \
                --resume \
                --output-dir "$circulation_output" \
                --vorticity-threshold-fraction 0.02 \
                --center-threshold-fraction 0.4
        fi

        echo "[$run_name] Rendering 3D normalized-vorticity GIF at threshold 0.02."
        ./ritta_plotting_3D/run_qcriterion_animation.sh \
            "$run_dir" \
            "$vorticity_stride" \
            --field vorticity \
            --vorticity-threshold-fraction 0.02 \
            --output-dir "$vorticity_output" \
            --resume \
            --workers "$frame_workers"
    } >"$log_path" 2>&1
}

run_dirs=()
found_cases=()
shopt -s nullglob
for candidate in "$sweep_dir"/*; do
    [[ -d "$candidate/output" ]] || continue
    snapshots=("$candidate"/output/flowTime_*.hdf5)
    [[ ${#snapshots[@]} -gt 0 ]] || continue
    if [[ ${#selected_cases[@]} -gt 0 ]]; then
        candidate_name="$(basename "$candidate")"
        selected=0
        for requested in "${selected_cases[@]}"; do
            if [[ "$candidate_name" == "$requested" ]]; then
                selected=1
                found_cases+=("$requested")
                break
            fi
        done
        [[ $selected -eq 1 ]] || continue
    fi
    run_dirs+=("$candidate")
done

if [[ ${#selected_cases[@]} -gt 0 ]]; then
    for requested in "${selected_cases[@]}"; do
        found=0
        for available in "${found_cases[@]}"; do
            [[ "$requested" == "$available" ]] && found=1
        done
        if [[ $found -eq 0 ]]; then
            echo "Requested case was not found with snapshots: $requested" >&2
            exit 2
        fi
    done
fi

if [[ ${#run_dirs[@]} -eq 0 ]]; then
    echo "No child tau runs with output/flowTime_*.hdf5 found in $sweep_dir" >&2
    exit 2
fi

echo "3D tau-sweep GIF job"
echo "  Job:                 $SLURM_JOB_ID"
echo "  Sweep:               $sweep_dir"
echo "  Tau runs:            ${#run_dirs[@]}"
echo "  Slice stride:        $slice_stride"
echo "  3D vorticity stride: $vorticity_stride"
echo "  Concurrent cases:    $case_workers"
echo "  Frame workers/case:  $frame_workers"
echo "  Max render workers:  $((case_workers * frame_workers))"
echo "  Products:            $gif_products"
echo "  3D threshold:        0.02 of max |vorticity|"
echo "  Logs:                 $repo_root/$log_dir"
echo

batch_pids=()
batch_names=()
batch_logs=()

wait_for_batch() {
    local index
    local failed=0
    for index in "${!batch_pids[@]}"; do
        if wait "${batch_pids[$index]}"; then
            echo "[${batch_names[$index]}] Requested GIF products completed."
        else
            echo "[${batch_names[$index]}] GIF generation failed." >&2
            echo "Log: ${batch_logs[$index]}" >&2
            tail -n 50 "${batch_logs[$index]}" >&2 || true
            failed=1
        fi
    done
    batch_pids=()
    batch_names=()
    batch_logs=()
    return "$failed"
}

failed=0
for run_dir in "${run_dirs[@]}"; do
    run_name="$(basename "$run_dir")"
    echo "[$run_name] Starting; log: $log_dir/${run_name}.log"
    run_case "$run_dir" &
    batch_pids+=("$!")
    batch_names+=("$run_name")
    batch_logs+=("$log_dir/${run_name}.log")
    if [[ ${#batch_pids[@]} -ge $case_workers ]]; then
        wait_for_batch || failed=1
    fi
done
if [[ ${#batch_pids[@]} -gt 0 ]]; then
    wait_for_batch || failed=1
fi

if [[ $failed -ne 0 ]]; then
    echo "One or more tau cases failed; see the case logs above." >&2
    exit 1
fi

echo "All requested tau-sweep GIF products completed."
