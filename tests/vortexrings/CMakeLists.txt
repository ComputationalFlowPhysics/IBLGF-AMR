#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

set(NAME vortexrings)

add_library(lib_${NAME} ${NAME}.cpp ${SOURCES})
target_link_libraries(lib_${NAME}
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)

add_executable(${NAME}.x ${NAME}_main.cpp ${SOURCES})

target_link_libraries(${NAME}.x
                      lib_${NAME})

# Testing --------------------------------------------------------------------

# copy config files
add_custom_target(copy_files ALL
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      ${CMAKE_CURRENT_SOURCE_DIR}/configs
                      ${CTEST_WORKING_DIRECTORY}/${NAME}/configs
                      )

# add excutables
add_executable(${NAME}Test ${NAME}_test.cpp)

add_dependencies(${NAME}Test copy_files)

target_link_libraries(${NAME}Test
    lib_${NAME}
    gtest_main
    )


# serial test

set(N_mpi_proc 2)

set(test_parameters ${MPIEXEC_NUMPROC_FLAG}
    ${N_mpi_proc} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}Test")

add_test(NAME ${NAME}_${N_mpi_proc}proc COMMAND ${MPIEXEC} ${test_parameters}
    WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME})

# parallel test

set(N_mpi_proc 13)

set(test_parameters ${MPIEXEC_NUMPROC_FLAG}
    ${N_mpi_proc} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}Test")

add_test(NAME ${NAME}_${N_mpi_proc}proc COMMAND ${MPIEXEC} ${test_parameters}
    WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME})



