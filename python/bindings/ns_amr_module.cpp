#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ns_amr_lgf_runner.hpp"

namespace py = pybind11;

PYBIND11_MODULE(iblgf_bindings_ns_amr, m)
{
    m.doc() = "Python bindings for the 3D NS AMR LGF entrypoint";

    auto ns_amr = m.def_submodule("ns_amr", "3D NS AMR LGF solver bindings");

    py::class_<iblgf::ns_amr_lgf::RunResult>(ns_amr, "RunResult")
        .def_readonly("measured_linf_error",
            &iblgf::ns_amr_lgf::RunResult::measured_linf_error)
        .def_readonly("expected_linf_error",
            &iblgf::ns_amr_lgf::RunResult::expected_linf_error)
        .def_readonly("difference", &iblgf::ns_amr_lgf::RunResult::difference)
        .def_readonly("overall_u1_linf_error",
            &iblgf::ns_amr_lgf::RunResult::overall_u1_linf_error)
        .def_readonly("fine_u1_linf_error",
            &iblgf::ns_amr_lgf::RunResult::fine_u1_linf_error)
        .def_readonly("fine_u2_linf_error",
            &iblgf::ns_amr_lgf::RunResult::fine_u2_linf_error)
        .def_readonly("fine_u3_linf_error",
            &iblgf::ns_amr_lgf::RunResult::fine_u3_linf_error)
        .def_readonly("config_path", &iblgf::ns_amr_lgf::RunResult::config_path)
        .def_readonly("working_directory",
            &iblgf::ns_amr_lgf::RunResult::working_directory);

    ns_amr.def("run", &iblgf::ns_amr_lgf::run_from_config,
        py::arg("config_path"),
        py::arg("cli_overrides") = std::vector<std::string>{},
        py::call_guard<py::gil_scoped_release>(),
        "Run the 3D NS AMR LGF setup using a config file path.");
}
