This directory contains the Python package that wraps the C++ pybind extension.

Role:
- provide a clean Python API for users
- hide direct interaction with the C++ executables and internals
- ship installable CLI entrypoints for common solver runs
- sit on top of the compiled pybind extension

Layout:
- `python/iblgf/`: importable Python package
- `python/iblgf/cli/`: installable console entrypoints
- `python/scripts/`: example scripts that call the public package API
- `python/bindings/`: `pybind11` C++ glue source and CMake files
- `python/tests/`: pytest suite for the Python package layer

Minimal package usage:

```python
from iblgf import poisson

result = poisson.run("/path/to/configFile")
print(result.measured_linf_error)
print(result.difference)
```

The public Python modules are:
- `iblgf.poisson`
- `iblgf.ns_amr_2d`

Recommended development install inside Docker:

```bash
cd /workspace2/IBLGF-AMR
python3 -m pip install -e .[test]
```

After that, users can either write Python scripts:

```bash
mpiexec -n 2 python3 my_experiment.py
```

or use the installed console entrypoints:

```bash
iblgf-poisson /path/to/configFile
iblgf-ns-amr-2d /path/to/configFile
```

For a direct source-tree example without installing console scripts:

```bash
cd /workspace2/IBLGF-AMR
export PYTHONPATH="/workspace2/IBLGF-AMR/python:$PYTHONPATH"
mpiexec -n 2 python3 ./python/scripts/simple_run_poisson_pybind.py ./tests/poisson/configFile_1ring
```
