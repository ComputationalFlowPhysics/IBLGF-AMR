# Resumable 3D Slurm sweeps

The RM submission script records each config's run directory, attempt count,
completion status, and per-attempt logs. After a wall-time cancellation, submit
the same command again. Completed configs are skipped, the interrupted config
is resumed from its restart files, and configs that never started run normally.

From the repository root, the initial submission and every later resubmission
use the same command:

```bash
sbatch --export=ALL,SWEEP_NAME=mass3d_v1,MPI_RANKS=32 \
  formation_3d/run_mass_sweep_existing_rm.sbatch \
  formation_3d/mass_configs 'config_*.cfg'
```

Choose a unique `SWEEP_NAME` for each distinct batch. Replace the config folder
and glob as needed. The default allocation is one exclusive RM node for 48
hours; the configs run sequentially with 32 MPI ranks unless `MPI_RANKS` is
changed.

Each source config must start with:

```text
write_restart=true;
use_restart=false;
```

The launcher enables `use_restart` only in the copy inside an interrupted run
directory. It does not modify the source config.

If the job is killed while a checkpoint is being replaced, the launcher also
checks the solver's `restart/backup` and `restart/backup_tmp` directories and
restores the newest complete fallback it can identify.

Progress can be inspected without opening every log:

```bash
column -t -s $'\t' formation_3d/mass_sweep_state/mass3d_v1/summary.tsv
tail formation_3d/mass_sweep_state/mass3d_v1/events.tsv
```

The summary reports `pending`, `incomplete`, `resume-ready`, or `completed`.
Per-attempt stdout and stderr are kept below the same state directory in
`logs/`. The state directory is ignored by Git.

The config names, order, paths, and checksums are saved in `manifest.tsv`.
Reusing a sweep name with a different config set is rejected to prevent an old
state file from skipping the wrong simulations. Use a new `SWEEP_NAME` when
the batch changes.
