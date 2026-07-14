#!/usr/bin/env bash
# Bridges-2 full-node rank-scaling benchmark for ns_amr_lgf2D.
# Submit from the IBLGF-AMR repository root with:
#   sbatch formation_number_plot/run_rank_benchmark_bridges2.sbatch

#SBATCH --job-name=iblgf-rank-bench
#SBATCH --account=mch250004p
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=06:00:00
#SBATCH --output=formation_number_plot/rank_scaling-%j.out
#SBATCH --error=formation_number_plot/rank_scaling-%j.err

set -euo pipefail

repo_root="/ocean/projects/mch250004p/mchoi10/IBLGF-AMR"
setup_script="/ocean/projects/mch250004p/mchoi10/iblgf-lib/setup_iblgf.sh"
config="${CONFIG_PATH:-$repo_root/config_scaling.cfg}"
runner="$repo_root/iblgf_remote.sh"

cd "$repo_root"

if [[ ! -f "$setup_script" ]]; then
  echo "Dependency setup script not found: $setup_script" >&2
  exit 1
fi
if [[ ! -f "$config" ]]; then
  echo "Benchmark config not found: $config" >&2
  exit 1
fi

# PSC requires the same MPI environment at runtime that was used to compile.
# setup_iblgf.sh provides that environment; mpiexec comes from that MPI stack.
source "$setup_script"
unset IBLGF_MPI_ARGS
export IBLGF_LAUNCHER=mpiexec
export IBLGF_BUILD_JOBS="${IBLGF_BUILD_JOBS:-8}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# When this file is submitted with sbatch, the special batch step does not
# consume the allocation and no overlap is needed. When it is instead run via
# `bash` from a normal srun attachment to an existing allocation, that parent
# step owns the node. Permit mpiexec's internal srun step to share the otherwise
# idle parent shell, and do not let it inherit the shell's one-task geometry.
case "${SLURM_STEP_ID:-}" in
  ""|batch|interactive|extern) ;;
  *)
    export SLURM_OVERLAP=1
    export SLURM_CPU_BIND=none
    unset SLURM_CPUS_PER_TASK
    unset SLURM_EXACT
    unset SLURM_NTASKS
    unset SLURM_NPROCS
    unset SLURM_NTASKS_PER_NODE
    ;;
esac

echo "Bridges-2 IBLGF MPI rank benchmark"
echo "  Job:      $SLURM_JOB_ID"
echo "  Step:     ${SLURM_STEP_ID:-allocation shell}"
echo "  Host:     $(hostname)"
echo "  Config:   $config"
echo "  Launcher: $(command -v mpiexec)"
mpiexec --version 2>&1 | head -n 2 || true

# Build once before timing. Subsequent run-test builds are no-op checks.
"$runner" build -j "$IBLGF_BUILD_JOBS"

# The largest run uses 108 of the RM node's 128 cores.
for ranks in 2 4 8 16 32 64 96 108; do
  echo
  echo "===== ${ranks} MPI ranks ====="
  "$runner" run-test ns_amr_lgf2D "$config" -n "$ranks" --bench
done

echo
echo "All rank benchmarks completed."
