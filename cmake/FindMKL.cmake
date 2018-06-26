########################################################################
#
# MKL configuration for IBLGF
#
# - FindMKL
# Find the native MKL includes and library
#
#  MKL_INCLUDES              - where to find fftw3.h
#  SCALAPACK_LIBRARIES       - List of required MKL/SCALAPACK libraries.
#  MKL_FOUND                 - True if MKL/Scalapack libraries found.
########################################################################

FIND_PATH(MKL_INCLUDES mkl_cblas.h
    HINTS $ENV{MKL_HOME}/include
          $ENV{MKLROOT}/include)

IF (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    SET(INTEL_LIBDIR "intel64")
ELSE()
    SET(INTEL_LIBDIR "ia32")
ENDIF()

FIND_LIBRARY(MKL_LAPACK
    NAMES mkl_lapack
          mkl_lapack95_lp64
          mkl_lapack95_ilp64
    HINTS $ENV{MKL_HOME}/lib/${INTEL_LIBDIR} 
          $ENV{MKLROOT}/lib/${INTEL_LIBDIR})


IF (MKL_LAPACK_FOUND)
    # old MKL versions
    IF (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        FIND_LIBRARY(MKL
            NAMES mkl_ia64
            HINTS $ENV{MKL_HOME}/lib/${INTEL_LIBDIR} 
                  $ENV{MKLROOT}/lib/${INTEL_LIBDIR})
    ELSE()
        FIND_LIBRARY(MKL
            NAMES mkl_ia32
            HINTS $ENV{MKL_HOME}/lib/${INTEL_LIBDIR} 
                  $ENV{MKLROOT}/lib/${INTEL_LIBDIR})
    ENDIF()

    FIND_LIBRARY(MKL_GUIDE
        NAMES guide
        HINTS $ENV{MKL_HOME}/lib/${INTEL_LIBDIR} 
              $ENV{MKLROOT}/lib/${INTEL_LIBDIR})

    SET(SCALAPACK_LIBRARIES
        ${MKL}
        ${MKL_LAPACK}
        ${MKL_GUIDE})
ELSE()
    # newer MKL version
    SET (MKL_LAPACK "")

    FIND_LIBRARY(MKL_INTEL
        NAMES mkl_intel_lp64
        HINTS $ENV{MKL_HOME}/lib/${INTEL_LIBDIR} 
              $ENV{MKLROOT}/lib/${INTEL_LIBDIR})

    FIND_LIBRARY(MKL_SEQUENTIAL
        NAMES mkl_sequential
        HINTS $ENV{MKL_HOME}/lib/${INTEL_LIBDIR} 
              $ENV{MKLROOT}/lib/${INTEL_LIBDIR})

    FIND_LIBRARY(MKL_CORE
        NAMES mkl_core
        HINTS $ENV{MKL_HOME}/lib/${INTEL_LIBDIR} 
              $ENV{MKLROOT}/lib/${INTEL_LIBDIR})

    FIND_LIBRARY(MKL_BLACS
            NAMES mkl_blacs_intelmpi_lp64 
                  mkl_blacs_intelmpi_ilp64
            HINTS $ENV{MKL_HOME}/lib/${INTEL_LIBDIR} 
                  $ENV{MKLROOT}/lib/${INTEL_LIBDIR})

    FIND_LIBRARY(MKL_SCALAPACK
            NAMES mkl_scalapack_lp64
                  mkl_scalapack_ilp64
            HINTS $ENV{MKL_HOME}/lib/${INTEL_LIBDIR} 
                  $ENV{MKLROOT}/lib/${INTEL_LIBDIR})

    SET(SCALAPACK_LIBRARIES
        ${MKL_INTEL}
        ${MKL_SEQUENTIAL}
        ${MKL_CORE}
        ${MKL_BLACS}
        ${MKL_SCALAPACK})
ENDIF()

SET(MKL_BLAS_INCLUDE_FILE ${MKL_INCLUDES}/mkl_blas.h)
SET(MKL_LAPACK_INCLUDE_FILE ${MKL_INCLUDES}/mkl_lapack.h)


IF (MKL_INCLUDES)
    SET(MKL_FOUND ON)
ENDIF()


IF (MKL_FOUND OR MKL_INTEL_FOUND)
    IF (NOT MKL_FIND_QUIETLY)
        MESSAGE(STATUS "Found MKL: ${MKL_INCLUDES}")
    ENDIF()
ELSE()
    SET(MKL_FOUND OFF)
ENDIF()
