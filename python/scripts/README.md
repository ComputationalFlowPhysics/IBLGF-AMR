This directory holds user-facing Python runner scripts.

Current script:
- `run_poisson.py`: launches the Poisson binding with a config file

Design split:
- `python/iblgf/`: importable Python package
- `python/scripts/`: executable Python entrypoints for users
- `python/bindings/`: C++ `pybind11` glue
