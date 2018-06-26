########################################################################
#
# SCALAPACK configuration for IBLGF
#
# - FindSCALAPACK
# Find the native SCALAPACK library.
#
#  SCALAPACK_LIBRARIES   - List of libraries when using SCALAPACK.
#  SCALAPACK_FOUND       - True if SCALAPACK found.
########################################################################


SET(SCALAPACK_SEARCH_PATHS
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64)

FIND_LIBRARY(NATIVE_SCALAPACK
             NAMES
                scalapack
                scalapack-2.0.2
             PATHS
                ${SCALAPACK_SEARCH_PATHS})

IF(NATIVE_SCALAPACK)

    SET(SCALAPACK_FOUND YES)

    SET(SCALAPACK_LIBRARIES
        "${NATIVE_SCALAPACK}"
        "${BLAS_LIBRARIES}"
        "${LAPACK_LIBRARIES}"
        CACHE STRING "" FORCE)

    ADD_CUSTOM_TARGET(scalapack-2.0.2 ALL)

ELSE()

    INCLUDE(ExternalProject)

    EXTERNALPROJECT_ADD(
        scalapack-2.0.2
        STAMP_DIR ${TPBUILD}/stamp
        DOWNLOAD_DIR ${TPSRC}
        DOWNLOAD_COMMAND tar -xvf ${TPSRC}/scalapack-2.0.2.tar.gz
        SOURCE_DIR ${TPSRC}/scalapack-2.0.2
        BINARY_DIR ${TPBUILD}/scalapack-2.0.2
        TMP_DIR ${TPBUILD}/scalapack-2.0.2-tmp
        INSTALL_DIR ${TPDIST}
        CONFIGURE_COMMAND
            cmake ${TPSRC}/scalapack-2.0.2 -DCMAKE_INSTALL_PREFIX=${TPDIST}
        BUILD_COMMAND make install)

    LINK_DIRECTORIES(${TPDIST}/lib)
    MESSAGE(STATUS "Built SCALAPACK: ${TPDIST}/lib/lib${SCALAPACK_LIBRARY}.a")

    SET(SCALAPACK ${TPDIST}/lib/libscalapack.a
        CACHE FILEPATH "SCALAPACK library" FORCE)

    SET(SCALAPACK_FOUND YES)

    SET(SCALAPACK_LIBRARIES
        "${SCALAPACK}"
        "${BLAS_LIBRARIES}"
        "${LAPACK_LIBRARIES}"
        CACHE STRING "" FORCE)

ENDIF()





