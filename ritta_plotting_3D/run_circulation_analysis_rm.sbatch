#!/bin/bash
# Resume one 3D circulation analysis on a full Bridges-2 RM node.
# Usage: sbatch ritta_plotting_3D/run_circulation_analysis_rm.sbatch RUN_FOLDER [STRIDE] [EXTRA_ARGS...]

#SBATCH -A mch250004p
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -J circulation3d
#SBATCH -o ritta_plotting_3D/slurm-circulation3d-%j.out
#SBATCH -e ritta_plotting_3D/slurm-circulation3d-%j.err

set -euo pipefail

repo_root="/ocean/projects/mch250004p/mchoi10/IBLGF-AMR"
submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
run_input="${1:-}"

if [[ -z "$run_input" ]]; then
    echo "Usage: sbatch $0 RUN_FOLDER [STRIDE] [EXTRA_ARGS...]" >&2
    exit 2
fi
shift

stride="${1:-1}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ "$run_input" = /* ]]; then
    run_dir="$run_input"
else
    run_dir="$submit_dir/$run_input"
fi
if [[ ! -d "$run_dir" ]]; then
    echo "Run folder not found: $run_dir" >&2
    exit 2
fi
run_dir="$(cd "$run_dir" && pwd)"

cd "$repo_root"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-128}"

echo "3D circulation analysis"
echo "  Job:         $SLURM_JOB_ID"
echo "  Host:        $(hostname)"
echo "  Run folder:  $run_dir"
echo "  Stride:      $stride"
echo "  CPU threads: $OMP_NUM_THREADS"
echo "  Resume:      yes"
echo

./ritta_plotting_3D/run_circulation_analysis.sh \
    "$run_dir" \
    "$stride" \
    --resume \
    "$@"
