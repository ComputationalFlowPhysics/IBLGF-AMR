########################################################################
#
# ATLAS configuration for IBLGF
#
# - FindATLAS
# Find the native ATLAS includes and library
#
#  ATLAS_INCLUDES    - where to find cblas.h
#  ATLAS_LIBRARIES   - List of libraries when using FFTW.
#  ATLAS_FOUND       - True if ATLAS found.
########################################################################

FIND_PATH(ATLAS_INCLUDE_DIR cblas.h
          /usr/include
          /usr/local/include
          /usr/local/include/atlas)

SET(ATLAS_LIB_SEARCH_DIRS
    /usr/lib/sse2
    /usr/lib/atlas/sse2
    /usr/local/lib/atlas)

FIND_LIBRARY(ATLAS NAMES atlas PATHS ${ATLAS_LIB_SEARCH_DIRS})
FIND_LIBRARY(ATLAS_CBLAS NAMES cblas PATHS ${ATLAS_LIB_SEARCH_DIRS})
FIND_LIBRARY(ATLAS_LAPACK NAMES lapack PATHS ${ATLAS_LIB_SEARCH_DIRS})

SET(ATLAS_INCLUDE_FILE ${ATLAS_INCLUDE_DIR}/cblas.h)
SET(ATLAS_LIBRARIES ${ATLAS} ${ATLAS_CBLAS} ${ATLAS_LAPACK})

IF (ATLAS_INCLUDE_DIR)
    SET(ATLAS_FOUND ON)
ENDIF()

IF (ATLAS_FOUND)
    MESSAGE(STATUS "Found ATLAS: ${ATLAS_INCLUDE_DIR}")
ELSE()
    IF (ATLAS_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find ATLAS")
    ENDIF()
ENDIF()
