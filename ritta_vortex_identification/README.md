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

Set `vorticity_threshold` and `minimum_region_area` under `[threshold_mask]`. The command runs Stage 1 first, removes mask regions below the physical-area cutoff, and marks only saved positive h-maxima inside the retained regions. It saves `hmaxima.h5`, `threshold_masks.h5`, and a terminal-selected `threshold_masks_preview.png`.

Make PNG frames and a GIF from any saved vortex HDF5 file:

```bash
python make_h5_gif.py H5_FILE STRIDE
```

This supports `hmaxima.h5`, `regions.h5`, `fits.h5`, and `threshold_masks.h5`. Output goes beside the HDF5 file in `<name>_stride_<stride>/`.

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

This makes only `combined_circulation_vs_time.png` and `combined_x_displacement_vs_time.png`. Each dataset line uses the rightmost valid vortex in every frame. A matching-color `X` marker on each curve marks when that dataset's forcing ends. New manifests save this as `forcing_end_time`, read from `b_f_tau` in the run's simulation config; older manifests derive it from `run_folder`.

Set `time_axis_min` and `time_axis_max` in `[time_series]` to choose the horizontal plot range. Use `nan` for automatic limits.

## config reminders

- `input`: files, simulation config, time, and origin; keep the field as `edge_aux`
- `hmaxima`: raise `h` to keep only more prominent extrema; keep connectivity at 8
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
regions.h5                    regions_preview.png
fits.h5                       fits_preview.png
positive_vortex_metrics.csv   positive_vortex_metrics_preview.png
circulation_vs_time.png       x_displacement_vs_time.png
```

The measured circulation uses positive original-cell vorticity inside the fitted positive circle, not the fitted `Gamma`. Existing tracks use one-to-one nearest-center matches in consecutive analyzed frames, limited by `tracking.max_displacement`. An unmatched candidate starts a track only when a second detection confirms it in the next analyzed frame within `tracking.new_track_max_displacement`; otherwise it is discarded as a one-frame detection. The time-series plots show only the rightmost confirmed vortex in each frame.
