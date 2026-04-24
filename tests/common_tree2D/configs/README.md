# common_tree2D config templates

These configs are templates for validating:
- `common_tree.cpp` / `common_tree_Test.cpp`
- `symgrid.cpp` / `symgrid_Test.cpp`
- `test_POD.cpp` / `test_POD_Test.cpp`

## Required input files
Place restart snapshots under `tests/common_tree2D/output/` (or adjust paths in each config):
- `tree_info_100.bin`
- `flowTime_100.hdf5`
- and for `common_tree`: additional time steps from `nStart + k*nskip` for `k=1..nTotal-1`

Example for the default `config_common_tree2D` values:
- `tree_info_100.bin`, `flowTime_100.hdf5`
- `tree_info_200.bin`, `flowTime_200.hdf5`
- `tree_info_300.bin`, `flowTime_300.hdf5`

## Run tests with explicit config
From your build directory:

```bash
IBLGF_COMMON_TREE2D_CONFIG_COMMON_TREE=/home/cc/Research/iblgf-local/IBLGF-AMR/tests/common_tree2D/configs/config_common_tree2D ctest -R common_tree_Test --output-on-failure

IBLGF_COMMON_TREE2D_CONFIG_SYMGRID=/home/cc/Research/iblgf-local/IBLGF-AMR/tests/common_tree2D/configs/config_symgrid2D ctest -R symgrid_Test --output-on-failure

IBLGF_COMMON_TREE2D_CONFIG_POD=/home/cc/Research/iblgf-local/IBLGF-AMR/tests/common_tree2D/configs/config_pod2D ctest -R test_POD_Test --output-on-failure
```

## Correctness checks
A run is considered correct when:
- The tests pass without exceptions/crashes.
- No `GTEST_SKIP` appears due to missing files.
- The expected output files are produced in `tests/common_tree2D/` (for example `adapted_to_ref*.hdf5`, `init.hdf5`, `final.hdf5`) depending on the pipeline.
