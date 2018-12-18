#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src ) 
 
# add the executable
add_executable(decompTest.x domainTest.cpp)
target_link_libraries(decompTest.x 
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES} boost_mpi 
                      ${FFTW_LIBRARIES}  
                      xtensor)
