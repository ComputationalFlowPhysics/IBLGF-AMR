include_directories(${IBLGF_SOURCE_DIR}/src )

# add the executable
add_executable(vortexrings.x vortexrings.cpp ${SOURCES})
target_link_libraries(vortexrings.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES}
                      ${BLAS_LIBRARIES}
                      )
