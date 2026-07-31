#!/bin/bash
# Run or resume an existing 3D mass-config sweep sequentially on one full RM node.
# Usage: sbatch formation_3d/run_mass_sweep_existing_rm.sbatch [CONFIG_FOLDER] [CONFIG_GLOB]

#SBATCH -A mch250004p
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH -t 48:00:00
#SBATCH -J mass3d
#SBATCH -o formation_3d/slurm-mass3d-rm-%j.out
#SBATCH -e formation_3d/slurm-mass3d-rm-%j.err

set -euo pipefail

repo_root="/ocean/projects/mch250004p/mchoi10/IBLGF-AMR"
submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
configs_input="${1:-${CONFIGS_DIR:-$repo_root/formation_3d/mass_configs}}"
config_glob="${2:-${CONFIG_GLOB:-*.cfg}}"
mpi_ranks="${MPI_RANKS:-32}"
progress_interval="${PROGRESS_INTERVAL:-20}"

if [[ "$configs_input" = /* ]]; then
  configs_dir="$configs_input"
else
  configs_dir="$submit_dir/$configs_input"
fi

if [[ ! -d "$configs_dir" ]]; then
  echo "Config directory not found: $configs_dir" >&2
  exit 2
fi
configs_dir="$(cd "$configs_dir" && pwd)"
sweep_name="${SWEEP_NAME:-$(basename "$configs_dir")}"
if [[ ! "$sweep_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "SWEEP_NAME may contain only letters, digits, dots, underscores, and hyphens: $sweep_name" >&2
  exit 2
fi
state_dir="${SWEEP_STATE_DIR:-$repo_root/formation_3d/mass_sweep_state/$sweep_name}"
logs_dir="${LOGS_DIR:-$state_dir/logs}"

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
echo "  State:    $state_dir"
echo "  Summary:  $state_dir/summary.tsv"
echo

mkdir -p "$logs_dir"

env -u IBLGF_MPI_ARGS -u SLURM_MPI_TYPE \
  CONFIGS_DIR="$configs_dir" \
  IBLGF_LAUNCHER=mpiexec \
  LOGS_DIR="$logs_dir" \
  PROGRESS_INTERVAL="$progress_interval" \
  SWEEP_STATE_DIR="$state_dir" \
  ./formation_3d/run_mass_sweep_remote.sh \
    --resumable "$mpi_ranks" "$config_glob"
