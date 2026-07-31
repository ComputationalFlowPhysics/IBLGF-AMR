#!/bin/bash
# Resume one existing 3D ns_amr_lgf run on an exclusive RM node.
# Usage: sbatch formation_3d/resume_ns_amr_lgf_rm.sbatch RUN_DIR [CONFIG_NAME]

#SBATCH -A mch250004p
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH -t 48:00:00
#SBATCH -J resume3d
#SBATCH -o formation_3d/slurm-resume3d-rm-%j.out
#SBATCH -e formation_3d/slurm-resume3d-rm-%j.err

set -euo pipefail

repo_root="/ocean/projects/mch250004p/mchoi10/IBLGF-AMR"
submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
run_input="${1:-}"
config_name="${2:-config_freq0p1.cfg}"

if [[ -z "$run_input" ]]; then
  echo "Usage: sbatch $0 RUN_DIR [CONFIG_NAME]" >&2
  exit 2
fi

if [[ "$run_input" = /* ]]; then
  run_dir="$run_input"
else
  run_dir="$submit_dir/$run_input"
fi
if [[ ! -d "$run_dir" ]]; then
  echo "Run directory not found: $run_dir" >&2
  exit 2
fi
run_dir="$(cd "$run_dir" && pwd)"
config_file="$run_dir/$config_name"

if [[ ! -f "$config_file" ]]; then
  echo "Run config not found: $config_file" >&2
  exit 2
fi
for restart_file in restart_field.hdf5 restart_info tree_info.bin; do
  if [[ ! -f "$run_dir/restart/$restart_file" ]]; then
    echo "Restart file not found: $run_dir/restart/$restart_file" >&2
    exit 2
  fi
done
if ! grep -Eq '^[[:space:]]*use_restart[[:space:]]*=[[:space:]]*true;' "$config_file"; then
  echo "Set use_restart=true in $config_file before submitting." >&2
  exit 2
fi
if ! grep -Eq '^[[:space:]]*write_restart[[:space:]]*=[[:space:]]*true;' "$config_file"; then
  echo "Set write_restart=true in $config_file so the resumed run keeps checkpointing." >&2
  exit 2
fi

mpi_ranks="${MPI_RANKS:-$(awk '$1 == "mpi_ranks:" {print $2}' "$run_dir/meta.txt")}"
if [[ ! "$mpi_ranks" =~ ^[1-9][0-9]*$ ]]; then
  echo "Could not determine a valid MPI rank count from $run_dir/meta.txt" >&2
  echo "Submit with MPI_RANKS=N sbatch ... to set it explicitly." >&2
  exit 2
fi

for saved_file in stdout.log stderr.log meta.txt; do
  if [[ -f "$run_dir/$saved_file" ]]; then
    cp -a \
      "$run_dir/$saved_file" \
      "$run_dir/${saved_file%.*}.before-resume-${SLURM_JOB_ID}.${saved_file##*.}"
  fi
done

cd "$repo_root"

echo "Resuming 3D IBLGF run"
echo "  Job:       $SLURM_JOB_ID"
echo "  Host:      $(hostname)"
echo "  Run dir:   $run_dir"
echo "  Config:    $config_file"
echo "  MPI ranks: $mpi_ranks"
echo

env -u IBLGF_MPI_ARGS -u SLURM_MPI_TYPE \
  IBLGF_LAUNCHER=mpiexec \
  ./iblgf_remote.sh run-test \
    ns_amr_lgf \
    "$config_file" \
    -n "$mpi_ranks" \
    --resume "$run_dir"
