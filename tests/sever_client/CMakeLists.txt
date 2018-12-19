#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src ${IBLGF_SOURCE_DIR}/include/domain/mpi) 
 
# add the executable
add_executable(server_client.x main.cpp)

target_link_libraries(server_client.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      )
