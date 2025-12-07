#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )
add_executable(convTest.x convTest.cpp ${SOURCES})
target_link_libraries(convTest.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)