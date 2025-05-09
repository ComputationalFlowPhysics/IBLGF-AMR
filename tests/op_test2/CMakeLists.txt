#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

set(NAME op_test2)
# add_custom_target(copy_ifherk_files2D ALL
#                       COMMAND ${CMAKE_COMMAND} -E copy_directory
#                       ${CMAKE_CURRENT_SOURCE_DIR}/configs
#                       ${CTEST_WORKING_DIRECTORY}/${NAME}
#                       )

# set(Ptest_list ns_amr_lgf_Test2D)

# add the executable
add_executable(op_test2.x op_test2.cpp ${SOURCES})
target_link_libraries(op_test2.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)


