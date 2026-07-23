#!/bin/bash
# Run existing 2D mass-sweep configs sequentially on one RM node.

#SBATCH -A mch250004p
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 48:00:00
#SBATCH -J massfreq
#SBATCH -o formation_number_plot/slurm-massfreq-rm-%j.out
#SBATCH -e formation_number_plot/slurm-massfreq-rm-%j.err

set -euo pipefail

repo_root="/ocean/projects/mch250004p/mchoi10/IBLGF-AMR"
configs_dir="${CONFIGS_DIR:-$repo_root/formation_number_plot/mass_configs}"
config_glob="${CONFIG_GLOB:-config_freq*.cfg}"
mpi_ranks="${MPI_RANKS:-32}"
progress_interval="${PROGRESS_INTERVAL:-20}"

cd "$repo_root"

if [[ ! -x ./formation_number_plot/run_mass_sweep_remote.sh ]]; then
  echo "Missing runner: $repo_root/formation_number_plot/run_mass_sweep_remote.sh" >&2
  exit 1
fi

echo "Sequential RM mass sweep"
echo "  Job:      ${SLURM_JOB_ID:-none}"
echo "  Host:     $(hostname)"
echo "  Configs:  $configs_dir/$config_glob"
echo "  MPI:      $mpi_ranks rank(s)"
echo "  Progress: every $progress_interval s"
echo

env -u IBLGF_MPI_ARGS -u SLURM_MPI_TYPE \
  CONFIGS_DIR="$configs_dir" \
  IBLGF_LAUNCHER=mpiexec \
  PROGRESS_INTERVAL="$progress_interval" \
  ./formation_number_plot/run_mass_sweep_remote.sh \
    --existing "$mpi_ranks" "$config_glob"
