#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "poisson_runner.hpp"

namespace py = pybind11;

PYBIND11_MODULE(iblgf_bindings, m)
{
    m.doc() = "Python bindings for selected IBLGF entrypoints";

    auto poisson = m.def_submodule("poisson", "Poisson solver bindings");

    py::class_<iblgf::poisson::RunResult>(poisson, "RunResult")
        .def_readonly("measured_linf_error",
            &iblgf::poisson::RunResult::measured_linf_error)
        .def_readonly("expected_linf_error",
            &iblgf::poisson::RunResult::expected_linf_error)
        .def_readonly("difference", &iblgf::poisson::RunResult::difference)
        .def_readonly("config_path", &iblgf::poisson::RunResult::config_path)
        .def_readonly(
            "working_directory", &iblgf::poisson::RunResult::working_directory);

    poisson.def("run", &iblgf::poisson::run_from_config,
        py::arg("config_path"),
        py::arg("cli_overrides") = std::vector<std::string>{},
        py::call_guard<py::gil_scoped_release>(),
        "Run the Poisson setup using a config file path.");
}
