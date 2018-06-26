SET(OPEN_BLAS_SEARCH_PATHS
    /lib/ /lib64/ /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64
    /opt/OpenBLAS/lib $ENV{OPENBLAS_HOME}/lib)

FIND_LIBRARY(OPENBLAS NAMES openblas PATHS ${OPEN_BLAS_SEARCH_PATHS})

IF (OPENBLAS)
    SET(OPENBLAS_FOUND ON)
ENDIF()

IF (OPENBLAS_FOUND)
    MESSAGE(STATUS "Found OpenBLAS")
ELSE()
    MESSAGE(FATAL_ERROR "Could not find OpenBLAS libraries.")
ENDIF()
