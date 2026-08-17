# 3D plotting

## Q-criterion or vorticity GIF

```bash
./ritta_plotting_3D/run_qcriterion_animation.sh RUN_FOLDER STRIDE --field q-criterion
```

## Circulation and vortex-ring center

Run one analysis pass for the circulation, axial center, and radial center:

```bash
./ritta_plotting_3D/run_circulation_analysis.sh RUN_FOLDER STRIDE
```

For CSV data and time-history plots without the slice frames or GIF, use
`--data-only`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
./ritta_plotting_3D/run_circulation_analysis.sh RUN_FOLDER 1 --data-only
```

The leading-vortex circulation cutoff defaults to the paper's 2% of maximum
absolute vorticity. Set it explicitly with `--vorticity-threshold-fraction`.

The center threshold defaults to 40% of the maximum absolute vorticity in each
snapshot. Set a different fraction with:

```bash
./ritta_plotting_3D/run_circulation_analysis.sh RUN_FOLDER STRIDE \
    --center-threshold-fraction 0.3
```

The ring is assumed to travel along x with symmetry axis `y = z = 0`. The
script analyzes the positive-y half of the `z = 0` meridional slice. It applies
the selected vorticity threshold, keeps the largest connected core, and uses

```text
x_c   = integral(x r^2 omega) / integral(r^2 omega)
y_c^2 = integral(r^2 omega)   / integral(omega)
```

where `r = y` on this slice. Therefore the reported `center_y` is the positive
radial coordinate (the vortex-ring radius), not a signed vertical displacement.

Results are written to
`ritta_plotting_3D/outputs/RUN_NAME_circulation/`:

- `leading_vortex_circulation.csv` contains circulation, center coordinates,
  thresholds, and source snapshot metadata.
- `leading_vortex_circulation.png` plots circulation versus time.
- `leading_vortex_center_x.png` plots axial center versus time.
- `leading_vortex_center_y.png` plots radial center versus time.
- `leading_vortex_connectivity.gif` and `frames/` contain the slice rendering;
  a small bright-green circular marker shows the calculated center in every
  valid frame. The render-view background is transparent in this GIF and in
  the Q-criterion and vorticity GIFs made by `plot_3d.py`.

If an analysis is interrupted, resume from its flushed CSV rows and existing
PNG frames instead of recalculating completed snapshots:

```bash
./ritta_plotting_3D/run_circulation_analysis.sh RUN_FOLDER 1 --resume
```

On Bridges-2, submit the same resume operation to a full RM node with:

```bash
sbatch ritta_plotting_3D/run_circulation_analysis_rm.sbatch RUN_FOLDER 1
```

## Resolution-comparison plots

Combine circulation and axial-center histories for every immediate child run
in a resolution-sweep folder:

```bash
source ritta_vortex_identification/open_viewer_venv.sh
./ritta_plotting_3D/run_resolution_comparison.sh \
    runs/ns_amr_lgf/res_sweep 1 0.4
```

Existing per-run `leading_vortex_circulation.csv` files are reused. The wrapper
runs the circulation analysis only for cases whose CSV is missing, then saves
`combined_circulation_vs_time.png` and `combined_center_x_vs_time.png` under
`ritta_plotting_3D/outputs/res_sweep_comparison/`. Legends show each case's
`dx_base`, `nLevels`, and finest spacing `dx_base / 2^nLevels`.

For a formation-time sweep, label and numerically order the curves by `b_f_tau`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
./ritta_plotting_3D/run_resolution_comparison.sh \
    runs/ns_amr_lgf/formation 1 0.4 tau 0.02 6
```

The final argument is the number of independent cases to analyze concurrently.
This task-level parallelism preserves the serial connectivity calculation for
each snapshot; distributing one clipped slice over many ParaView MPI ranks can
leave empty partitions and make `vtkPConnectivityFilter` fail. The comparison
wrapper also uses data-only analysis and writes each case under a
campaign-qualified folder such as `outputs/formation_tau_5p0_circulation/`,
preventing identically named cases from different campaigns from overwriting
one another.

Formation-time comparisons also save
`combined_circulation_vs_time_over_tau.png`, which plots the same circulation
against `t / tau` over `0 <= t / tau <= 1`.

The circulation region always uses the paper's 2%-of-maximum-vorticity cutoff.
The fifth argument sets that cutoff explicitly. The third argument independently
sets the Lamb-center cutoff, and the sixth sets the concurrent case count.
