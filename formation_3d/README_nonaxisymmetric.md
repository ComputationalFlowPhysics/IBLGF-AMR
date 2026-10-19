# Non-axisymmetric 3D ring forcing

Branch: `rc-vrt-nonaxisymmetric-forcing`, based on `rc_vrt`.

The 3D solver now calls `body_force_profile_3d_nonaxisymmetric`, which uses
unchanged `body_force_profile_3d` as f0 and multiplies it by
`1 + sum(epsilon_m * cos(m * theta + phi_m))`. The code's axial direction
is x (the image calls it z), with `theta = atan2(z-z0, y-y0)`.
The force remains axial; the existing time envelope, pulse parameters,
spatial smoothing, and solver sign convention are unchanged. 2D is unchanged.
At exactly r=0 the angular mean f0 is used because theta is undefined.
The specified angular modulation has no unique continuous limit there;
no radial taper has been added to the requested equation.

## Config

Copy `formation_3d/config3D_nonaxisymmetric.cfg`, or add inside
`simulation_parameters` in an existing 3D config:

```
b_f_m=(2);
b_f_epsilon_m=(0.1);
b_f_phi_m=(0.0);
```

These are lists even for one mode. Each list entry describes an azimuthal
mode, not a pulse. Modes must be positive integers; amplitudes and phases
must be finite. Phases are in radians and default to zero when omitted.
The mode and amplitude lists must have equal lengths (and phases too if
specified). For example, a combination of m=2 and m=3 is:

```
b_f_m=(2,3);
b_f_epsilon_m=(0.1,0.05);
b_f_phi_m=(0.0,1.5707963267948966);
```

Omitting all three settings, or setting amplitudes to zero, recovers the
original axisymmetric forcing. The factor is not clipped: choosing
`sum(abs(epsilon_m)) <= 1` is sufficient to keep it nonnegative.
The sample is copied from `config3D_test` with the Fourier settings added and checkpoint writing enabled;
it retains the original grid and 4000 steps, so it is not a small smoke test.
The resumable batch runner requires `write_restart=true` and
`use_restart=false` for a fresh run; both sample configs use these settings.

## ACCESS / Bridges-2 submission

First transfer these branch changes to the cluster checkout (including the
new sample config). From that checkout on the login/access node, use the
existing batch script with an exact filename glob:

```bash
cd /ocean/projects/mch250004p/mchoi10/IBLGF-AMR
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
SWEEP_NAME=nonaxisymmetric_m2_eps0p1 MPI_RANKS=640 \
sbatch formation_3d/run_mass_sweep_existing_rm.sbatch \
  formation_3d config3D_nonaxisymmetric.cfg
```

This retains the script's current five RM nodes, 128 tasks per node,
48-hour limit, and allocation `mch250004p`. The runner invokes
`iblgf_remote.sh run-test ns_amr_lgf`, which rebuilds before running.
No batch-script changes are needed. Use a different `SWEEP_NAME` for a fresh
experiment: this script otherwise resumes/skips runs using saved sweep state.
Changing the local branch alone does not update cluster files.
