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

set(SOURCES 
    "${CMAKE_CURRENT_LIST_DIR}/src/utilities/convolution.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/lgf/lgf_gl_lookup.cpp"
   )

#Output compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

#include(${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt.user)
set (CMAKE_CXX_STANDARD 17)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo -O3 -static-intel -Wall -fmax-errors=4")
else()
    message("Using compiler: " ${CMAKE_CXX_COMPILER_ID})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ftree-vectorize -funroll-loops  -ftree-vectorizer-verbose=3  -Wall -O3 -Wall ")
endif()    

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
message(${BLAS_LIBRARIES})


#FFTW
#if(NOT DEFINED FFTW_INCLUDES)
#set(FFTW_DIR "cmake/")
#find_package(FFTW REUIRED)
#endif(NOT DEFINED FFTW_INCLUDES)
include_directories(SYSTEM ${FFTW_INCLUDES})
link_directories(${FFTW_LIBRARIES} )


#XTensor
find_package(xsimd REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtl REQUIRED)
find_package(xtensor-blas REQUIRED)
include_directories(SYSTEM ${xtensor_INCLUDE_DIRS} 
                           ${xtensor_blas_INCLUDE_DIRS} 
                           ${xsimd_INCLUDE_DIRS})

#HDF5
if(NOT DEFINED HDF5_INCLUDE_DIRS)
find_package(HDF5)
endif(NOT DEFINED HDF5_INCLUDE_DIRS)
include_directories(${HDF5_INCLUDE_DIRS})
link_directories(${HDF5_LIBRARY_DIRS} )

# GTEST
# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()

# Now simply link against gtest or gtest_main as needed. Eg
#add_executable(example example.cpp)
#target_link_libraries(example gtest_main)
#add_test(NAME example_test COMMAND example)

# Define these in CMakeLists.txt.user (point to locally compiled Gtests)
include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIBRARY_DIR})

# add the executable
#add_executable(iblgf "src/main.cpp")
#target_link_libraries(iblgf
#                      ${MPI_LIBRARIES}
#                      ${Boost_LIBRARIES} boost_mpi
#                      ${FFTW_LIBRARIES} fftw3_omp blas
#                      xtensor)

add_subdirectory (tests/sever_client)
add_subdirectory (tests/vortexring)
add_subdirectory (tests/vortexrings)
#add_subdirectory (tests/operators)
add_subdirectory (tests/ifherk_heat)
add_subdirectory (setups/poissonProblem)
enable_testing ()

#Sample test
set (test_parameters -np "2"  )
add_test (NAME DomainTest
          COMMAND "/usr/bin/mpirun.openmpi" ${test_parameters}
          bin/decompTest.x
          "${CMAKE_CURRENT_LIST_DIR}/tests/domain_decomposition/configFile")



