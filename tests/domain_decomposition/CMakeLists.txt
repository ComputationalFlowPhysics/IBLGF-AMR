#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

# add the executable
add_executable(decompTest.x domainTest.cpp ${SOURCES})
target_link_libraries(decompTest.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES} #boost_mpi boost_serialization boost_filesystem
                      xtensor)
