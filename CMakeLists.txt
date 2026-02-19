#      ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
#     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
#      ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀
#          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌
#          ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄
#          ▐░▌     ▐░░░░░░░░░░▌ ▐░▌          ▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌
#          ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀
#          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌
#      ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌
#     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌
#      ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀

cmake_minimum_required(VERSION 3.12.0)
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt.user")
    include(${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt.user)
else()
endif()
project(IBLGF VERSION 0.0.0 LANGUAGES CXX C)


option(USE_GPU "Build with GPU support" OFF)
if(USE_GPU)
    enable_language(CUDA)
    message(STATUS "Building IBLGF with GPU support")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    message(STATUS "Building IBLGF without GPU support")
endif()

cmake_policy(SET CMP0048 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CXX_FLAGS "-march=native -Wno-error=deprecated-declarations")
set(CMAKE_CXX_FLAGS "-O3 -Wno-error=deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
#     --std=c++17 \
#     --compiler-options=-std=c++17")

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3  -fmax-errors=4 ")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo -O3 -static-intel -Wall -fmax-errors=4")
else()
    message("Using compiler of family: " ${CMAKE_CXX_COMPILER_ID})
endif()


set(SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/src/utilities/convolution.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/utilities/misc_math_functions.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/lgf/lgf_gl_lookup.cpp"
   )
set(GPU_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/src/utilities/convolution_GPU.cu"
   )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(IBLGF_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${IBLGF_MODULE_PATH}")

find_package(MPI REQUIRED)
find_package(Boost 1.68 REQUIRED COMPONENTS system filesystem serialization mpi)
find_package(xsimd REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(xtl REQUIRED)
find_package(FFTW REQUIRED)
find_package(BLAS REQUIRED)

#################################################################################################
#FIXME:
if(NOT DEFINED HDF5_INCLUDE_DIRS)
find_package(HDF5)
endif(NOT DEFINED HDF5_INCLUDE_DIRS)
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${xtensor_blas_INCLUDE_DIRS})
link_directories(${HDF5_LIBRARY_DIRS} )
#################################################################################################

# Boost CMake target compatibility:
# - older setups expose Boost::boost
# - newer setups expose Boost::headers
if(TARGET Boost::headers)
    set(IBLGF_BOOST_HEADERS_TARGET Boost::headers)
elseif(TARGET Boost::boost)
    set(IBLGF_BOOST_HEADERS_TARGET Boost::boost)
else()
    message(FATAL_ERROR "Neither Boost::headers nor Boost::boost target is available.")
endif()

set(IBLGF_BOOST_TARGETS
    ${IBLGF_BOOST_HEADERS_TARGET}
    Boost::system
    Boost::filesystem
    Boost::serialization
    Boost::mpi
)

# Keep legacy variable for test CMake files that still consume Boost_LIBRARIES directly.
set(Boost_LIBRARIES ${IBLGF_BOOST_TARGETS})

set(IBLGF_SKIP_MPICXX ON CACHE BOOL "True if your compiler wrapper includes MPI already (as CRAY PE for instance)")
if (IBLGF_SKIP_MPICXX)
    set(MPI_CXX_SKIP_MPICXX ON)
else()
    set(MPI_CXX_SKIP_MPICXX OFF)
endif()

set(IBLGF_TESTING ON CACHE BOOL "True if tests shall be built")

# Allow MPI oversubscription for environments with fewer cores than requested processes
# This is necessary for CI/CD environments (e.g., GitHub Actions) where tests may request
# more MPI ranks than available CPU cores

# Default MPI ranks for tests
if(NOT DEFINED IBLGF_MPI_NP)
  include(ProcessorCount)
  ProcessorCount(_host_cpus)
  if(NOT _host_cpus OR _host_cpus EQUAL 0)
    set(_host_cpus 2)  # safe fallback
  endif()
  if(_host_cpus GREATER 12)
    set(_host_cpus 12)  # ensure at least 4 ranks for testing
  endif()

  set(IBLGF_MPI_NP "${_host_cpus}" CACHE STRING "MPI ranks for tests")
endif()

message(STATUS "IBLGF_MPI_NP = ${IBLGF_MPI_NP}")

add_library(iblgf INTERFACE)
add_library(iblgf::iblgf ALIAS iblgf)
target_include_directories(iblgf INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )
include_directories(include)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_include_directories(iblgf INTERFACE ${MPI_INCLUDE_PATH})
target_link_libraries(iblgf INTERFACE ${IBLGF_BOOST_TARGETS})
target_link_libraries(iblgf INTERFACE MPI::MPI_CXX)
target_link_libraries(iblgf INTERFACE xsimd)
target_link_libraries(iblgf INTERFACE xtl)
target_link_libraries(iblgf INTERFACE xtensor)
target_link_libraries(iblgf INTERFACE xsimd)
target_link_libraries(iblgf INTERFACE ${HDF5_LIBRARIES})

target_link_libraries(iblgf INTERFACE hdf5)
target_link_libraries(iblgf INTERFACE ${BLAS_LIBRARIES})
target_link_libraries(iblgf INTERFACE ${FFTW_LIBRARIES})
target_compile_features(iblgf INTERFACE cxx_std_17)

enable_testing()
set(CTEST_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/unit_testing)

if (IBLGF_TESTING )
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.10.0
    )
    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
        add_library(GTest::gtest ALIAS gtest)
    endif()

    if (IBLGF_TESTING)
        add_library(iblgf_test_main ./src/gtest_main_test.cpp)
        target_link_libraries(iblgf_test_main GTest::gtest iblgf)
        add_subdirectory(tests)
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS iblgf EXPORT iblgf-targets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT iblgf-targets
  FILE iblgf-targets.cmake
  NAMESPACE iblgf::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
  )
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/iblgfConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/iblgfConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)
write_basic_package_version_file(iblgfConfigVersion.cmake VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/iblgfConfig.cmake
    DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake)
