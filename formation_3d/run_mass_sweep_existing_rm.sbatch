#!/bin/bash
# Run an existing 3D mass-config sweep sequentially on one full RM node.

#SBATCH -A mch250004p
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 48:00:00
#SBATCH -J mass3d
#SBATCH -o formation_3d/slurm-mass3d-rm-%j.out
#SBATCH -e formation_3d/slurm-mass3d-rm-%j.err

set -euo pipefail

repo_root="/ocean/projects/mch250004p/mchoi10/IBLGF-AMR"
configs_dir="${CONFIGS_DIR:-$repo_root/formation_3d/mass_configs}"
config_glob="${CONFIG_GLOB:-config_*.cfg}"
mpi_ranks="${MPI_RANKS:-32}"
progress_interval="${PROGRESS_INTERVAL:-20}"

cd "$repo_root"

if [[ ! -x ./formation_3d/run_mass_sweep_remote.sh ]]; then
  echo "Missing runner: $repo_root/formation_3d/run_mass_sweep_remote.sh" >&2
  exit 1
fi

if [[ ! "$mpi_ranks" =~ ^[1-9][0-9]*$ ]]; then
  echo "MPI_RANKS must be a positive integer: $mpi_ranks" >&2
  exit 1
fi
if [[ ! "$progress_interval" =~ ^[1-9][0-9]*$ ]]; then
  echo "PROGRESS_INTERVAL must be a positive integer: $progress_interval" >&2
  exit 1
fi

echo "Sequential RM 3D mass sweep"
echo "  Job:      ${SLURM_JOB_ID:-none}"
echo "  Host:     $(hostname)"
echo "  Configs:  $configs_dir/$config_glob"
echo "  MPI:      $mpi_ranks rank(s)"
echo "  Progress: every $progress_interval s"
echo

mkdir -p formation_3d/mass_logs_remote

env -u IBLGF_MPI_ARGS -u SLURM_MPI_TYPE \
  IBLGF_LAUNCHER=mpiexec \
  PROGRESS_INTERVAL="$progress_interval" \
  ./formation_3d/run_mass_sweep_remote.sh --existing "$mpi_ranks" "$config_glob"
