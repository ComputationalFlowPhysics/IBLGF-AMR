# Minimum CMake version required
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Project name
PROJECT(IBLGF C CXX)

# Enable testing framework
INCLUDE(CTest)

# Add support for CMAKE_DEPENDENT_OPTION
INCLUDE(CMakeDependentOption)



# Make sure default compilation is RELEASE
# ------------------------------------------------------------------------------
IF (NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are: None Debug Release." FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
# ------------------------------------------------------------------------------



# Define path where to search for customised "Find.cmake" files
# ------------------------------------------------------------------------------
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# ------------------------------------------------------------------------------



# Set default install path if not provided on the command-line
# ------------------------------------------------------------------------------
IF (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE PATH "" FORCE)
ENDIF()
# ------------------------------------------------------------------------------

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++14")



# Set various ThirdParty locations
# ------------------------------------------------------------------------------
SET(TPURL   http://www.iblgf.caltech.edu/thirdparty)
SET(TPSRC   ${CMAKE_SOURCE_DIR}/ThirdParty)
SET(TPBUILD ${CMAKE_BINARY_DIR}/ThirdParty)
SET(TPDIST  ${CMAKE_BINARY_DIR}/ThirdParty/dist)
SET(TPINC   ${CMAKE_INSTALL_PREFIX}/include/ThirdParty)
# ------------------------------------------------------------------------------



# Find required FFTW package or install it
# ------------------------------------------------------------------------------
OPTION(IBLGF_USE_FFTW
"Use FFTW routines for performing the Fast Fourier Transform." ON)

IF (IBLGF_USE_FFTW)
    FIND_PACKAGE(FFTW REQUIRED)
ENDIF()
# ------------------------------------------------------------------------------



# Add relevant subdirectories for compilation
# ------------------------------------------------------------------------------
ADD_SUBDIRECTORY(SpatialDomain)
# ------------------------------------------------------------------------------



# Add executable
# ------------------------------------------------------------------------------
add_executable(IBLGFSolver main.cpp)
target_link_libraries(IBLGFSolver
		      SpatialDomain
                      ${FFTW_LIBRARY})
# ------------------------------------------------------------------------------



# Add dependencies
# ------------------------------------------------------------------------------
ADD_DEPENDENCIES(SpatialDomain
                 fftw-3.3.5)
# ------------------------------------------------------------------------------
