cmake_minimum_required (VERSION 2.9)
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt.user")
    include(${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt.user)
else()
endif()
project (IBLGF)

# Define path where to search for customised "Find.cmake" files
# ------------------------------------------------------------------------------
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# ------------------------------------------------------------------------------
 
#Bring the headers, such as Student.h into the project
include_directories(include)
 
set(SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/lgf/lgf_lookup.cpp")

#Output compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

#include(${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt.user)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fconcepts -Wno-narrowing -fmax-errors=2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
#LIBS

#MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
link_directories(${MPI_LIBRARIES} )

#BOOST
find_package(Boost COMPONENTS serialization mpi system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})


#CBLAS
find_package(BLAS REQUIRED)
include_directories(SYSTEM ${BLAS_INCLUDE_DIRS})
link_directories(${BLAS_LIBRARIES} )


#FFTW
set(FFTW_DIR "cmake/")
find_package(FFTW REQUIRED)
include_directories(SYSTEM ${FFTW_INCLUDES})
link_directories(${FFTW_LIBRARIES} )

#XTensor 
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

# add the executable
#add_executable(iblgf "src/main.cpp")
#target_link_libraries(iblgf
#                      ${MPI_LIBRARIES}
#                      ${Boost_LIBRARIES} boost_mpi 
#                      ${FFTW_LIBRARIES} fftw3_omp blas
#                      xtensor)

add_subdirectory (tests/sever_client)
add_subdirectory (tests/domain_decomposition)
add_subdirectory (setups/poissonProblem)
enable_testing ()

#Sample test
set (test_parameters -np "2"  )
add_test (NAME DomainTest 
          COMMAND "/usr/bin/mpirun.openmpi" ${test_parameters} 
          bin/decompTest.x 
          "${CMAKE_CURRENT_LIST_DIR}/tests/domain_decomposition/configFile")



