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

include_directories(include)

set(SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/lgf/lgf_gl_lookup.cpp")

#Output compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

#include(${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt.user)
set (CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -O3  -ggdb -fmax-errors=2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -O3 -ggdb -pg -Wall -fmax-errors=2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
#LIBS

#MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
link_directories(${MPI_LIBRARIES} )

#BOOST
#if(NOT DEFINED Boost_INCLUDE_DIR)
find_package(Boost COMPONENTS system filesystem serialization mpi REQUIRED)
#endif(NOT DEFINED Boost_INCLUDE_DIR)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})



#CBLAS
find_package(BLAS REQUIRED)
include_directories(SYSTEM ${BLAS_INCLUDE_DIRS})
link_directories(${BLAS_LIBRARIES} )
link_directories(${BLAS_LIBRARY_DIR} )



#FFTW

#if(NOT DEFINED FFTW_INCLUDES)
#set(FFTW_DIR "cmake/")
#find_package(FFTW REQUIRED)
#endif(NOT DEFINED FFTW_INCLUDES)
include_directories(SYSTEM ${FFTW_INCLUDES})
link_directories(${FFTW_LIBRARIES} )


#XTensor
find_package(xtensor REQUIRED)
find_package(xtl REQUIRED)
find_package(xtensor-blas REQUIRED)
include_directories(SYSTEM ${xtensor_blas_INCLUDE_DIRS})

#HDF5
#if(NOT DEFINED HDF5_INCLUDE_DIRS)
#find_package(HDF5)
#endif(NOT DEFINED HDF5_INCLUDE_DIRS)
include_directories(${HDF5_INCLUDE_DIRS})
link_directories(${HDF5_LIBRARY_DIRS} )
#message( ${HDF5_INCLUDE_DIRS} )
#message( ${HDF5_LIBRARY_DIRS} )

#message( ${HDF5_LIBRARIES} )

# add the executable
#add_executable(iblgf "src/main.cpp")
#target_link_libraries(iblgf
#                      ${MPI_LIBRARIES}
#                      ${Boost_LIBRARIES} boost_mpi
#                      ${FFTW_LIBRARIES} fftw3_omp blas
#                      xtensor)

add_subdirectory (tests/sever_client)
add_subdirectory (tests/vortexring)
add_subdirectory (tests/operators)
add_subdirectory (setups/poissonProblem)
enable_testing ()

#Sample test
set (test_parameters -np "2"  )
add_test (NAME DomainTest
          COMMAND "/usr/bin/mpirun.openmpi" ${test_parameters}
          bin/decompTest.x
          "${CMAKE_CURRENT_LIST_DIR}/tests/domain_decomposition/configFile")



