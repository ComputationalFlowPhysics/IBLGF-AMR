# IBLGF-AMR Copilot Instructions

## Project Overview
IBLGF-AMR solves the incompressible Navier-Stokes equations on unbounded domains using a fast lattice Green's function (LGF) method with adaptive mesh refinement (AMR). It employs a mimetic finite volume approach on octree-structured meshes refined based on flow features.

**Key papers:**
- Liska & Colonius (2016): Fast LGF method on unbounded domains
- Dorschner et al. (2020): Multi-resolution LGF for elliptic equations

## Architecture Overview

### Core Components

**Domain (`include/iblgf/domain/`)**
- `domain.hpp`: Central abstraction managing octree mesh, AMR logic, and domain decomposition
- Template `Domain<Dim, DataBlock, helmholtz, N_modes>` parameterizes dimensions and physics
- Octree structure provides spatial indexing; blocks store data at each node
- Decomposition handles MPI parallelization with server-client architecture

**Solvers (`include/iblgf/solver/`)**
- LGF-based Poisson solvers (`poisson/`) for pressure projection
- Time integration (`time_integration/ifherk.hpp`) for time stepping
- Stability analysis (`Stability_solver/`, `modal_analysis/`)
- Direct solvers (MKL PARDISO, PETSc) for specific problem types

**Lattice Green's Functions (`include/iblgf/lgf/`)**
- `lgf.hpp` (Stokes), `lgf_ge.hpp` (Green's expansion), `helmholtz.hpp` (modified operator)
- Lookup tables (`lgf_gl_lookup.hpp`) for fast evaluation avoiding direct computation
- Table generation in `lgf_table_gen/` populates reference data

**Immersed Boundary (`include/iblgf/domain/ib.hpp`, `ib_communicator.hpp`)**
- Enforces boundary conditions for complex geometries
- Distributed via MPI; communicator exchanges IB force information

**Configuration System (`include/iblgf/dictionary/`)**
- Dictionary class loads config files (`.libconfig` format, see `tests/ns_amr_lgf/configs/configFile_*`)
- Config keys: `simulation_parameters`, `domain`, `output`, `restart` sections
- Simulation object reads dictionary at init time

### Data Flow
1. **Config Load** → Dictionary parsed with domain geometry, physics parameters
2. **Domain Setup** → Octree initialized; base blocks created; ghost layers allocated
3. **Simulation Loop** → LGF-based Poisson solve → incompressibility enforcement → time step
4. **Refinement** → Error indicators checked; octree adapted; data re-distributed via decomposition
5. **I/O** → HDF5 output via chombo format; checkpoints for restart

## Build & Test Workflow

### Build Process
```bash
./iblgf.sh build            # Default: parallel jobs = nproc
./iblgf.sh build -j 4       # Explicit parallelism
```

**Build output:** `build/bin/*.x` executables, `build/lib/` libraries

**Key CMake configuration:**
- C++17 required (set globally in root `CMakeLists.txt`)
- External deps: Boost MPI, FFTW, HDF5, xtensor, xsimd
- Optional GPU support via `-DUSE_GPU=ON` (requires CUDA)
- Tests auto-discovered via `tests/CMakeLists.txt` subdirectories

### Testing
```bash
./iblgf.sh test             # Sequential test execution (safe default)
./iblgf.sh test -j 4        # Parallel ctest (4 concurrent tests)
./iblgf.sh run-test ns_amr_lgf configFile_0       # Single test + logging
./iblgf.sh run-test ns_amr_lgf configFile_0 -n 8  # Override MPI ranks
```

**Test structure:**
- Each test directory under `tests/` has executable + config files in `configs/`
- Test code loads all configs matching `config*` pattern and runs simulation for each
- Results logged to timestamped `runs/` directory with config, stdout, stderr saved
- Supports `write_restart` / `use_restart` for checkpoint-based debugging

### Docker Workflow
```bash
./docker_iblgf.sh -c 4      # Launch Docker with 4 CPU cores; repo at /workspace2
./iblgf.sh build            # Build inside container (preferred for consistency)
```

## Key Patterns & Conventions

### Template-Heavy Design
- Domain parameterized by dimension (2D/3D), DataBlock type, Helmholtz flag, N modes
- Enables compile-time specialization without runtime branching
- See `setups/setup_base.hpp` for example of traits-based setup abstraction
- CRTP (`include/iblgf/utilities/crtp.hpp`) used for static polymorphism in solvers

### Test-Config Pattern
- Tests read YAML-like `.libconfig` format configs (not JSON/TOML)
- Config defines domain bounds (`bd_base`, `bd_extent`, `dx_base`), refinement criteria (`base_level_threshold`), physics params (`Re`, `cfl`)
- Multiple configs per test enable parametric studies
- Example: `tests/ns_amr_lgf/configs/configFile_0` sets up 3D vortex ring problem

### MPI Decomposition
- Domain partitioned via `Decomposition` (server-client model)
- Server manages global octree; clients hold local block data
- Halo communication via `mpi/haloCommunicator.hpp` for ghost layer updates
- IB forces communicated separately if immersed boundary active

### Refinement & Adaptation
- Octree refined based on user-defined criteria (e.g., `base_level_threshold` on velocity magnitude)
- Refinement condition registered as callback; checked each timestep
- Adapted cells re-partitioned across MPI ranks; data interpolated to new mesh

## Critical File References

| Path | Purpose |
|------|---------|
| `CMakeLists.txt` | Root build config; sets C++17, finds external libs |
| `include/iblgf/global.hpp` | Global type aliases (e.g., `float_type = double`) |
| `include/iblgf/types.hpp` | `vector_type`, `coordinate_type` definitions |
| `include/iblgf/simulation.hpp` | Main Simulation class; orchestrates solve |
| `setups/setup_*.hpp` | Problem-specific setups (linear, helmholtz, Newton, etc.) |
| `tests/ns_amr_lgf/ns_amr_lgf_Test.cpp` | Test harness; loads configs and runs simulations |
| `iblgf.sh` | Build/test orchestration wrapper (bash) |
| `include/iblgf/domain/decomposition/decomposition.hpp` | Domain partition logic |

## Common Development Tasks

**Adding a new solver:**
- Create `include/iblgf/solver/my_solver/my_solver.hpp`
- Inherit from base solver (e.g., LGF_Base for LGF solvers)
- Implement required methods; leverage MPI communicator if distributed
- Reference in `setups/setup_*.hpp` or test directly

**Modifying refinement criteria:**
- Edit config file's `base_level_threshold` or add new parameter
- Register refinement function as callback in Simulation or Domain
- Rebuild config loading path if new params added to dictionary

**Debugging a test:**
- Run: `./iblgf.sh run-test test_name configFile_0`
- Check logs in `runs/` timestamped directory for output/error
- Adjust config (smaller domain, coarser mesh) for faster iteration
- Use `-n 1` to reduce MPI complexity for initial debugging

**Profiling:**
- Build with optimizations (default Release mode): `./iblgf.sh build`
- Run with profiler: `perf record ./build/bin/ns_amr_lgf.x config`
- LGF lookups and Poisson solve typically dominate; check solver timing

## Environment Variables
- `IBLGF_BUILD_JOBS`: Parallel build jobs (default: nproc)
- `IBLGF_TEST_JOBS`: Parallel ctest jobs (default: 1)
- `IBLGF_MPI_RANKS`: Default MPI ranks for run/run-test (default: 2)
- `IBLGF_MPI_NP`: Number of MPI processes for ctest (overrides auto-detection; useful in CI/CD with resource constraints)

