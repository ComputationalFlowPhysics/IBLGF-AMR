########################################################################
#
# Blas/Lapack configuration for IBLGF
#
# - FindNativeBlasLapack
# Find the native Blas/Lapack libraries.
#
#  BLAS_LIBRARIES           - Blas libraries.
#  LAPACK_LIBRARIES         - Lapack libraries.
#  NATIVE_BLAS_LAPACK_FOUND - True if native Blas/Lapack found.
########################################################################


SET(NATIVE_BLAS_LAPACK_SEARCH_PATHS
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64)

FIND_LIBRARY(NATIVE_BLAS NAMES blas
             PATHS ${NATIVE_BLAS_LAPACK_SEARCH_PATHS})
FIND_LIBRARY(NATIVE_LAPACK NAMES lapack
             PATHS ${NATIVE_BLAS_LAPACK_SEARCH_PATHS})


IF (NATIVE_BLAS AND NATIVE_LAPACK)
    SET(NATIVE_BLAS_LAPACK_FOUND ON)
    SET(BLAS_LIBRARIES ${NATIVE_BLAS} CACHE FILEPATH "BLAS libraries" FORCE)
    SET(LAPACK_LIBRARIES ${NATIVE_LAPACK} CACHE FILEPATH "LAPACK libraries" FORCE)
ENDIF()

IF (NATIVE_BLAS_LAPACK_FOUND)
    MESSAGE(STATUS "Found Native Blas and Lapack")
ELSE(NATIVE_BLAS_LAPACK_FOUND)
    MESSAGE(FATAL_ERROR "Could not find Native blas and lapack libraries.")
ENDIF()
