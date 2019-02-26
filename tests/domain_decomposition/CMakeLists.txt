#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

# add the executable
add_executable(decompTest.x domainTest.cpp ${SOURCES})
target_link_libraries(decompTest.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3 blas
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor)
