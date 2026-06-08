if(NOT Python3_Interpreter_FOUND)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pytest --version
    RESULT_VARIABLE IBLGF_PYTEST_AVAILABLE
    OUTPUT_QUIET
    ERROR_QUIET
)

if(NOT IBLGF_PYTEST_AVAILABLE EQUAL 0)
    message(WARNING "pytest was not found; skipping Python binding tests.")
    return()
endif()

add_test(
    NAME python_bindings_unit
    COMMAND ${Python3_EXECUTABLE} -m pytest
            ${IBLGF_SOURCE_DIR}/python/tests
            -q
)
set_tests_properties(
    python_bindings_unit
    PROPERTIES
    ENVIRONMENT "PYTHONPATH=${IBLGF_SOURCE_DIR}/python"
)
