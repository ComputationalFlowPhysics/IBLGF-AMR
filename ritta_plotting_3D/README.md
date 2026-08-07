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
- `leading_vortex_connectivity.gif` and `frames/` contain the slice rendering.
