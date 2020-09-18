#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

# add the executable
add_executable(postProcessIntrp.x postProcessIntrp.cpp ${SOURCES})
target_link_libraries(postProcessIntrp.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)
