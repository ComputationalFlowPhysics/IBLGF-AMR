#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src ${IBLGF_SOURCE_DIR}/include) 
 
# add the executable
add_executable(poisson.x main.cpp ${SOURCES})
target_link_libraries(poisson.x 
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${BLAS_LIBRARIES}
                      ${FFTW_LIBRARIES}  fftw3_omp blas
                      xtensor)
