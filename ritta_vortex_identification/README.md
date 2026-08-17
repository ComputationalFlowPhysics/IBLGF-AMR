# vortex scripts

Quick notes for running my standalone 2D `edge_aux` analysis.

## before running

```bash
source open_viewer_venv.sh
```

This uses the local viewer venv when available, or the remote
`conda-envs/ritta-vortex` environment, and checks for NumPy, SciPy, h5py, and
Matplotlib. Set `RITTA_VORTEX_ENV` first if the environment is somewhere else.

`RUN_FOLDER` needs an `output/` folder containing `flowTime_<number>.hdf5`. Copy `configs/default.toml` and adjust it for the run; its inline comments explain the individual settings.

## run these in order

```bash
python 01_find_hmaxima.py RUN_FOLDER CONFIG_FILE
python 02_make_regions.py RUN_FOLDER CONFIG_FILE
python 03_fit_vortices.py RUN_FOLDER CONFIG_FILE
python 04_positive_vortex_metrics.py RUN_FOLDER CONFIG_FILE
python 05_plot_time_series.py RUN_FOLDER CONFIG_FILE
```

Do not skip stages. Each one reads the saved result from the previous stage.
Stage 2 reads `Re` from the run's copied simulation config and uses the saved
nondimensional simulation time as the vortex age. Its buffer widths grow diffusively as
`buffer_multiplier * sqrt(alpha_i^2 / (2 * alpha) + 2 * time / Re)`, so the
original forcing-based rectangle is recovered at time zero.

After Stage 3, export nine beginning/middle/end diagnostic PNGs with:

```bash
python export_fit_previews.py RUN_FOLDER CONFIG_FILE
```

The command chooses the earliest successful-fit frame, the successful frame
nearest the simulation midpoint, and the latest successful-fit frame. For each
one it selects the successful fit with the largest finite boundary radius and
saves only that candidate's local maximum, mirrored-point fitting rectangle,
and fitted vortex boundaries and centers under
`outputs/<run_name>/fit_previews/`.
It also stacks each diagnostic type's beginning, middle, and end PNGs
horizontally into three `*_combined.png` files, then removes the nine
individual panel PNGs. The preview axes default to `x in [-1, 5]` and
`y in [-1.5, 1.5]`; override them with `--x-axis-min`, `--x-axis-max`,
`--y-axis-min`, and `--y-axis-max`.

For a boundary-fraction sweep, Stage 3 can reuse existing maxima and regions
with `--input-results-dir`, override the radius fraction with
`--boundary-fraction`, and save a separate HDF5 file with `--output-file`.
The preview exporter accepts that file through `--fits-file`; use
`--fits-only` and `--filename-prefix` for one clearly labeled comparison PNG.

To make one labeled figure from several combined fit previews, repeat `--row`
with each row label and PNG:

```bash
python stack_combined_previews.py stacked.png \
  --row "Re = 200" re_200_fits_combined.png \
  --row "Re = 1000" re_1000_fits_combined.png \
  --row "Re = 1800" re_1800_fits_combined.png
```

To analyze every fifth sorted HDF5 frame, use `--stride 5` on Stage 1. Stages 2–4 automatically use that saved frame list:

```bash
python 01_find_hmaxima.py RUN_FOLDER CONFIG_FILE --stride 5
```

Optional check before stage 1: `python plot_vorticity.py RUN_FOLDER CONFIG_FILE`

Make a GIF of only the raw vorticity field, without running the vortex-identification stages:

```bash
python make_vorticity_gif.py RUN_FOLDER CONFIG_FILE --stride 5
```

This saves the selected PNG frames and `vorticity.gif` under
`outputs/<run_name>/vorticity_stride_<stride>/`. Use `--duration-ms 150` to
control the display time per frame, or `--output-dir FOLDER` to choose another
destination.

Optional vorticity-threshold masks:

```bash
python make_threshold_masks.py RUN_FOLDER CONFIG_FILE --stride 5
```

To parallelize the independent h-maxima and threshold-mask calculations across
the CPU cores of one compute node, add `--workers`. Keep native
numerical-library threads at one per worker to avoid oversubscribing the
allocation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python make_threshold_masks.py RUN_FOLDER CONFIG_FILE --stride 1 --workers 128 --no-preview
```

Both frame calculations use separate processes and temporary HDF5 shards. The
shards are merged into `hmaxima.h5` and `threshold_masks.h5` in the original
numerical frame order, so the reconstruction, extrema, threshold, and area
filtering calculations are unchanged. Tracking remains serial because it
depends on preceding frames.

Set `vorticity_threshold` and `minimum_region_area` under `[threshold_mask]`. The command runs Stage 1 first, removes mask regions below the physical-area cutoff, and marks only saved positive h-maxima inside the retained regions. It saves `hmaxima.h5`, `threshold_masks.h5`, and a terminal-selected `threshold_masks_preview.png`.

The same command also tracks the retained h-maxima across consecutive analyzed
frames. It uses `tracking.max_displacement` for existing tracks and
`tracking.new_track_max_displacement` for the required second-frame
confirmation of a new track. Each current detection can belong to at most one
track. A confirmed track remains active for `tracking.max_missed_frames`
frames without a detection. During a gap, its position is extrapolated from
the componentwise mean of its last `tracking.velocity_history_length`
observed velocities and physical simulation time. Tracks with less velocity
history use every available velocity. A reacquired detection must lie within
`tracking.max_displacement` of that predicted position. A successful match
keeps the same track ID and color, adds the velocity measured between the two
surrounding observations to the history, and connects those observed points
on the plot. After tracking, any track with fewer than
`tracking.minimum_track_points` observed detections is discarded. Results are
saved to `threshold_hmaxima_tracks.csv`, including blank track IDs for
unconfirmed or discarded extrema, and
`threshold_hmaxima_x_vs_time.png` and `threshold_hmaxima_y_vs_time.png`, where
each confirmed track has its own consistent color. These files are also
created with `--no-preview`.

Make PNG frames and a GIF from any saved vortex HDF5 file:

```bash
python make_h5_gif.py H5_FILE STRIDE
```

This supports `hmaxima.h5`, `regions.h5`, `fits.h5`, and `threshold_masks.h5`.
For `fits.h5`, each frame displays only the successful fit with the largest
finite boundary radius. Output goes beside the HDF5 file in
`<name>_stride_<stride>/`.
Use `--x-axis-min VALUE` and `--x-axis-max VALUE` to override the saved x-axis
limits for the rendered frames.
For `threshold_masks.h5`, use `--threshold-vorticity-background` to replace
the flat mask with the signed vorticity field while retaining the same extrema
overlay.

## fast headless runs

One run (either the run folder or its `output` folder works):

```bash
python run_all.py RUN_OR_OUTPUT CONFIG_FILE
```

A parent folder containing several runs:

```bash
python run_all.py PARENT_FOLDER CONFIG_FILE --batch
```

Add `--stride 5` to either `run_all.py` command to process every fifth frame.

These run stages 1–4 without previews. Each dataset gets a folder under
`outputs/pipeline_results/` containing `hmaxima.h5`, `regions.h5`, `fits.h5`,
and `positive_vortex_metrics.csv`. Use `--results-dir FOLDER` to put them
elsewhere.

The same folder gets `datasets.toml`. Each `name` identifies its dataset, while combined-plot legends use the corresponding forcing-end time. Then run:

```bash
python plot_combined_time_series.py outputs/pipeline_results/datasets.toml CONFIG_FILE
```

To make a second set of plots containing only selected forcing durations, pass
the desired values with `--tau-values` and use a separate output folder:

```bash
python plot_combined_time_series.py outputs/pipeline_results/datasets.toml CONFIG_FILE \
    --tau-values 1 3 5 7 9 11 20 45 \
    --normalized-zoom 0.6 1.0 0.5 0.9 \
    --output-dir outputs/pipeline_results/selected_tau
```

This reuses the existing metrics CSV files and leaves the original combined
plots unchanged. Every requested tau must exist in `datasets.toml`. The
optional normalized zoom is saved as
`combined_circulation_vs_time_over_tau_zoom.png`.

For a resolution sweep, label and order the curves by the copied simulation
config's `domain.dx_base` value instead of forcing-end time:

```bash
python plot_combined_time_series.py outputs/res_test_tau_20/datasets.toml CONFIG_FILE --legend-by dx-base
```

For a Reynolds-number sweep, read `Re` from each copied simulation config and
order the legend by increasing Reynolds number:

```bash
python plot_combined_time_series.py outputs/reynolds_sweep/datasets.toml CONFIG_FILE --legend-by reynolds
```

Add `--circulation-inset` to retain the full circulation history while showing
a magnified plateau inside the same figure. Its default zoom is
`x in [3, 25]` and `y in [0.78, 0.81]`; adjust it with `--inset-x-min`,
`--inset-x-max`, `--inset-y-min`, and `--inset-y-max`.

This makes `combined_circulation_vs_time.png`,
`combined_circulation_vs_time_over_tau.png`, and
`combined_x_displacement_vs_time.png`. The normalized circulation plot uses
\(t/\tau\) over \(0\leq t/\tau\leq1\), without forcing-end markers. Each
dataset line uses the confirmed fit with the largest boundary radius in every frame. On the
simulation-time figures, a matching-color `X` marker on each curve marks when
that dataset's forcing ends. New manifests save this as `forcing_end_time`,
read from `b_f_tau` in the run's simulation config; older manifests derive it
from `run_folder`. Combined-plot legends are ordered by increasing
`forcing_end_time` (increasing \(\tau\)). The shared plotting palette assigns
every dataset a different color and preserves the same tau-to-color mapping
across all combined figures. With `--legend-by dx-base`, legends use
\(\Delta x_{\mathrm{base}}\) and are ordered from largest to smallest grid
spacing. With `--legend-by reynolds`, legends use \(\mathrm{Re}\) and are
ordered from smallest to largest Reynolds number. The default remains the
\(\tau\) legend.

To fit the circulation slope-change time for every dataset and put all results
on one figure:

```bash
python plot_circulation_breakpoints.py outputs/pipeline_results/datasets.toml CONFIG_FILE
```

This starts each fit at that dataset's forcing-end time and iterates the
piecewise-linear model
`Gamma(t) = Gamma_0 + m_1 t + delta_m max(0, t - t_b)`. The output
`combined_circulation_with_breakpoints.png` retains the matching-color `X`
forcing markers and adds matching-color diamonds at the fitted
`t_b` times. Each diamond's circulation coordinate is linearly interpolated
from the actual plotted samples, so the marker lies directly on its curve
rather than on the piecewise-linear fit. The fitted times and slopes are also
printed. Use `--tolerance` and `--max-iterations` to change the stopping
criteria.

Set `time_axis_min` and `time_axis_max` in `[time_series]` to choose the horizontal plot range. Use `nan` for automatic limits.

## config reminders

- `input`: files, simulation config, time, and origin; keep the field as `edge_aux`
- `hmaxima`: raise `h` to keep only more prominent extrema; keep connectivity at 8;
  `merge_distance` replaces connected groups of nearby maxima by their mean location
- `threshold_mask`: threshold and preview colors for the optional binary masks
- `region`: rectangle scales and the shared `buffer_multiplier`
- `fit`: bounds and optimizer settings
- `tracking`: consecutive-frame displacement limits `L` and `L_new`
- `plot`: preview appearance
- `time_series`: reference-line slope and anchor

Frames are sorted by the number in `flowTime_<number>.hdf5`. Time is:

```text
t = cfl * dx_base * frame_number / 2^nLevels
```

## preview prompt

Stages 1–4 calculate every frame first, then save a preview PNG. At the prompt:

- Enter or `n`: next
- `p`: previous
- a frame index: jump there
- `q`: quit

## outputs

Everything goes to `outputs/<run_name>/`:

```text
hmaxima.h5                    hmaxima_preview.png
threshold_masks.h5            threshold_masks_preview.png
threshold_hmaxima_tracks.csv   threshold_hmaxima_x_vs_time.png
threshold_hmaxima_y_vs_time.png
regions.h5                    regions_preview.png
fits.h5                       fits_preview.png
positive_vortex_metrics.csv   positive_vortex_metrics_preview.png
circulation_vs_time.png       x_displacement_vs_time.png
```

The measured circulation uses positive original-cell vorticity inside the fitted positive circle, not the fitted `Gamma`. Existing tracks use one-to-one nearest-center matches in consecutive analyzed frames, limited by `tracking.max_displacement`. An unmatched candidate starts a track only when a second detection confirms it in the next analyzed frame within `tracking.new_track_max_displacement`; otherwise it is discarded as a one-frame detection. The time-series plots show the confirmed fit with the largest boundary radius in each frame.
