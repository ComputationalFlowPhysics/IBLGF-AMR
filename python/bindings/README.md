This directory holds the `pybind11` C++ extension glue for the Python interface.

Planned role:
- expose selected C++ entrypoints such as `poisson` to Python
- keep binding-specific C++ glue separate from solver/source code
- build a Python extension module without changing user-facing solver logic
