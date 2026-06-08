#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ns_amr_lgf2d_runner.hpp"
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

    auto ns_amr_2d =
        m.def_submodule("ns_amr_2d", "2D NS AMR LGF solver bindings");

    py::class_<iblgf::ns_amr_lgf2d::RunResult>(ns_amr_2d, "RunResult")
        .def_readonly("measured_linf_error",
            &iblgf::ns_amr_lgf2d::RunResult::measured_linf_error)
        .def_readonly("expected_linf_error",
            &iblgf::ns_amr_lgf2d::RunResult::expected_linf_error)
        .def_readonly("difference", &iblgf::ns_amr_lgf2d::RunResult::difference)
        .def_readonly("overall_u1_linf_error",
            &iblgf::ns_amr_lgf2d::RunResult::overall_u1_linf_error)
        .def_readonly("fine_u1_linf_error",
            &iblgf::ns_amr_lgf2d::RunResult::fine_u1_linf_error)
        .def_readonly("fine_u2_linf_error",
            &iblgf::ns_amr_lgf2d::RunResult::fine_u2_linf_error)
        .def_readonly("config_path", &iblgf::ns_amr_lgf2d::RunResult::config_path)
        .def_readonly("working_directory",
            &iblgf::ns_amr_lgf2d::RunResult::working_directory);

    ns_amr_2d.def("run", &iblgf::ns_amr_lgf2d::run_from_config,
        py::arg("config_path"),
        py::arg("cli_overrides") = std::vector<std::string>{},
        py::call_guard<py::gil_scoped_release>(),
        "Run the 2D NS AMR LGF setup using a config file path.");
}
