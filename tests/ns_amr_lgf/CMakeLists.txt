#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

set(NAME ns_amr_lgf)
add_custom_target(copy_ifherk_files ALL
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      ${CMAKE_CURRENT_SOURCE_DIR}/configs
                      ${CTEST_WORKING_DIRECTORY}/${NAME}
                      )

set(Ptest_list ns_amr_lgf_Test)

# add the executable
add_executable(ns_amr_lgf.x ns_amr_lgf.cpp ${SOURCES})
target_link_libraries(ns_amr_lgf.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)

foreach (item ${Ptest_list})
    add_executable(${item} ${item}.cpp ${SOURCES})
    add_dependencies(${item} copy_ifherk_files)
    target_link_libraries(${item} iblgf_test_main)
    add_test(
        NAME ${item}_parallel_np13
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 10 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
    )
endforeach()
