#Bring the headers, such as Student.h into the project
#include_directories(${IBLGF_SOURCE_DIR}/src )

#find_package(PETSc)
#find_package(SLEPc)

#INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
#INCLUDE_DIRECTORIES(${SLEPC_INC})
#ADD_DEFINITIONS(${PETSC_DEFINITIONS})

set(NAME OMP_MPI)



# add the executable
add_executable(OMP_MPI.x OMP_MPI.cpp)
target_link_libraries(OMP_MPI.x
                      PUBLIC ${MPI_LIBRARIES}
                      PUBLIC ${Boost_LIBRARIES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(OMP_MPI.x PUBLIC OpenMP::OpenMP_CXX)
endif()
