#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src ) 
 
# add the executable
add_executable(server_client.x main.cpp)
target_link_libraries(server_client.x 
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES} boost_mpi 
                      ${FFTW_LIBRARIES}  
                      xtensor)
