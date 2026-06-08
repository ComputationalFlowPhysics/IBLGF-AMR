if(EXISTS "${IBLGF_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
    add_subdirectory(
        "${IBLGF_SOURCE_DIR}/extern/pybind11"
        "${CMAKE_CURRENT_BINARY_DIR}/pybind11"
        EXCLUDE_FROM_ALL
    )
else()
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG QUIET)
endif()

if(NOT TARGET pybind11::module)
    message(FATAL_ERROR
        "pybind11 was not found. Install pybind11 or vendor it under "
        "${IBLGF_SOURCE_DIR}/extern/pybind11 before enabling "
        "IBLGF_BUILD_PYTHON.")
endif()

pybind11_add_module(iblgf_bindings
    module.cpp
    "${IBLGF_SOURCE_DIR}/tests/poisson/poisson_runner.cpp"
    "${IBLGF_SOURCE_DIR}/tests/ns_amr_lgf2D/ns_amr_lgf2d_runner.cpp"
    ${SOURCES}
)

target_include_directories(iblgf_bindings PRIVATE
    "${IBLGF_SOURCE_DIR}/tests/poisson"
    "${IBLGF_SOURCE_DIR}/tests/ns_amr_lgf2D"
)

target_link_libraries(iblgf_bindings PRIVATE iblgf)
target_compile_features(iblgf_bindings PRIVATE cxx_std_17)

install(TARGETS iblgf_bindings
    LIBRARY DESTINATION iblgf
    RUNTIME DESTINATION iblgf
)
