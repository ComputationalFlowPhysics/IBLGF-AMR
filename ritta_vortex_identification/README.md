# vortex scripts

Quick notes for running my standalone 2D `edge_aux` analysis.

## before running

```bash
source open_viewer_venv.sh
```

This uses the viewer venv and checks for NumPy, SciPy, h5py, and Matplotlib.

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

Optional check before stage 1: `python plot_vorticity.py RUN_FOLDER CONFIG_FILE`

## fast CSV-only runs

One run (either the run folder or its `output` folder works):

```bash
python run_all.py RUN_OR_OUTPUT CONFIG_FILE
```

A parent folder containing several runs:

```bash
python run_all.py PARENT_FOLDER CONFIG_FILE --batch
```

These run stages 1–4 without previews. Temporary HDF5 stage files are deleted; final CSV files go to `outputs/pipeline_results/`. Use `--results-dir FOLDER` to put them elsewhere.

The same folder gets `datasets.toml`. Edit each `name` in that file to choose the plot legend text, then run:

```bash
python plot_combined_time_series.py outputs/pipeline_results/datasets.toml CONFIG_FILE
```

This makes only `combined_circulation_vs_time.png` and `combined_x_displacement_vs_time.png`. Each dataset line uses the rightmost valid vortex in every frame.

Set `time_axis_min` and `time_axis_max` in `[time_series]` to choose the horizontal plot range. Use `nan` for automatic limits.

## config reminders

- `input`: files, simulation config, time, and origin; keep the field as `edge_aux`
- `hmaxima`: h value and reconstruction tolerances; keep connectivity at 8
- `region`: rectangle constants
- `fit`: bounds and optimizer settings
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
regions.h5                    regions_preview.png
fits.h5                       fits_preview.png
positive_vortex_metrics.csv   positive_vortex_metrics_preview.png
circulation_vs_time.png       x_displacement_vs_time.png
```

The measured circulation uses positive original-cell vorticity inside the fitted positive circle, not the fitted `Gamma`. All positive candidates are tracked. A missed detection leaves a gap in that vortex's time series.
