````instructions
# IBLGF-AMR Copilot Instructions

## Big Picture
- IBLGF-AMR solves incompressible Navier-Stokes on unbounded domains using LGF + AMR on octree meshes; the workflow is config-driven and MPI-parallel.
- Core data flow: dictionary config → domain/octree setup → LGF-based solves + time integration → refinement + redistribution → HDF5 output and restart files.

## Architecture Map (start here)
- Domain + AMR: `include/iblgf/domain/domain.hpp` and `include/iblgf/domain/decomposition/decomposition.hpp` (server-client MPI decomposition, octree ownership).
- Simulation orchestration: `include/iblgf/simulation.hpp` (reads dictionary, drives time loop and adaptation).
- Solvers: `include/iblgf/solver/` with LGF Poisson, IFHERK time integration (`include/iblgf/solver/time_integration/ifherk.hpp`), and PETSc/MKL variants.
- LGF kernels + tables: `include/iblgf/lgf/*.hpp` and lookup tables in `include/iblgf/lgf/lgf_gl_lookup.hpp`.
- Immersed boundary: `include/iblgf/domain/ib.hpp` and `include/iblgf/domain/ib_communicator.hpp` (IB forces communicated separately from halo exchanges).

## Configuration Pattern (project-specific)
- Tests and runs are driven by `.libconfig` files (not JSON/TOML); see `tests/ns_amr_lgf/configs/configFile_0` for a full example.
- Common keys: `simulation_parameters`, `domain`, `output`, `restart` (e.g., `base_level_threshold`, `bd_base`, `dx_base`, `Re`, `cfl`).
- Test harness loads every `config*` under a test’s `configs/` directory (see `tests/ns_amr_lgf/ns_amr_lgf_Test.cpp`).

## Build & Run Workflow (preferred)
- Use wrapper scripts: `./iblgf.sh build`, `./iblgf.sh test`, `./iblgf.sh run-test <test> <config>`.
- `run-test` writes a timestamped folder under `runs/<test>/` containing stdout/stderr and run metadata.
- Restart support: set `write_restart` / `use_restart` in config, then `./iblgf.sh run-test <test> <config> --resume`.
- Docker workflow: `./docker_iblgf.sh` launches the lab image and mounts repo at `/workspace2`.

## Conventions & Patterns
- Template-heavy design: `Domain<Dim, DataBlock, helmholtz, N_modes>` and setup traits in `setups/setup_base.hpp`.
- MPI layout: server-client decomposition with halo updates in `include/iblgf/mpi/haloCommunicator.hpp`.
- Output and restart files are HDF5-based; check `runs/` artifacts before changing I/O.

## External Dependencies (CMake expects these)
- C++17 + MPI, Boost (filesystem/serialization/mpi), FFTW, HDF5 (MPI), BLAS, xtensor + xtensor-blas + xtl + xsimd (see `README.md`).

## Environment Variables (used by `iblgf.sh`)
- `IBLGF_BUILD_JOBS`, `IBLGF_TEST_JOBS`, `IBLGF_MPI_RANKS`, `IBLGF_MPI_NP`.
````
- Config keys: `simulation_parameters`, `domain`, `output`, `restart` sections

- Simulation object reads dictionary at init time
