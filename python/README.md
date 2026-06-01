This directory is reserved for the future Python package that wraps the C++ bindings.

Planned role:
- provide a clean Python API for users
- hide direct interaction with the C++ executables and internals
- sit on top of the compiled pybind extension

Layout:
- `python/iblgf/`: importable Python package
- `python/scripts/`: user-facing runner scripts
- `python/bindings/`: `pybind11` C++ glue source and CMake files

Minimal usage goal:

```python
from iblgf import poisson

result = poisson.run("/path/to/configFile")
print(result.measured_linf_error)
print(result.difference)
```

Typical launch from Docker after building with `-DIBLGF_BUILD_PYTHON=ON`:

```bash
export PYTHONPATH="/workspace2/IBLGF-AMR/python:$PYTHONPATH"
mpiexec -n 2 python3 /workspace2/IBLGF-AMR/python/test_poisson_pybind.py /path/to/configFile
```
