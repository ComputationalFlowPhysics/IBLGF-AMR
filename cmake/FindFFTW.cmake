########################################################################
#
# FFTW configuration for IBLGF
#
# - FindFFTW
# Find the native FFTW includes and library
#
#  FFTW_INCLUDES    - where to find fftw3.h
#  FFTW_LIBRARIES   - List of libraries when using FFTW.
#  FFTW_FOUND       - True if FFTW found.
########################################################################

IF (IBLGF_USE_FFTW)

    # Set some common FFTW search paths.
    SET(FFTW_SEARCH_PATHS
        $ENV{LD_LIBRARY_PATH}
        $ENV{FFTW_HOME}/lib
        $ENV{FFTWHOME}/lib)

    FIND_LIBRARY(FFTW_LIBRARY NAMES fftw3 fftw3f PATHS ${FFTW_SEARCH_PATHS})

    IF (FFTW_LIBRARY)
        GET_FILENAME_COMPONENT(FFTW_PATH ${FFTW_LIBRARY} PATH)
        SET(FFTW_INCLUDE_DIR ${FFTW_PATH}/../include CACHE FILEPATH
            "FFTW include directory.")
        SET(BUILD_FFTW OFF)
    ELSE()
        SET(BUILD_FFTW ON)
    ENDIF()

    INCLUDE_DIRECTORIES(SYSTEM ${FFTW_INCLUDE_DIR})
    MARK_AS_ADVANCED(FFTW_LIBRARY)
    MARK_AS_ADVANCED(FFTW_INCLUDE_DIR)

    CMAKE_DEPENDENT_OPTION(THIRDPARTY_BUILD_FFTW
    "Build FFTW from ThirdParty" ${BUILD_FFTW}
    "IBLGF_USE_FFTW" OFF)

    IF (THIRDPARTY_BUILD_FFTW)
        INCLUDE(ExternalProject)
        EXTERNALPROJECT_ADD(
            fftw-3.3.5
            STAMP_DIR ${TPBUILD}/stamp
            DOWNLOAD_DIR ${TPSRC}
            DOWNLOAD_COMMAND tar -xvf ${TPSRC}/fftw-3.3.5.tar.gz
            SOURCE_DIR ${TPSRC}/fftw-3.3.5
            BINARY_DIR ${TPBUILD}/fftw-3.3.5
            TMP_DIR ${TPBUILD}/fftw-3.3.5-tmp
            INSTALL_DIR ${TPDIST}
            CONFIGURE_COMMAND ${TPSRC}/fftw-3.3.5/configure --prefix=${TPDIST}
                        --quiet --enable-shared --disable-dependency-tracking)

        SET(FFTW_LIBRARY fftw3 CACHE FILEPATH "FFTW library" FORCE)
        SET(FFTW_INCLUDE_DIR ${TPDIST}/include CACHE FILEPATH "FFTW include" FORCE)
        LINK_DIRECTORIES(${TPDIST}/lib)
        MESSAGE(STATUS "Build FFTW: ${TPDIST}/lib/lib${FFTW_LIBRARY}.so")
        SET(FFTW_CONFIG_INCLUDE_DIR ${TPINC})

    ELSE()
        ADD_CUSTOM_TARGET(fftw-3.3.5 ALL)
        MESSAGE(STATUS "Found FFTW: ${FFTW_LIBRARY}")
        SET(FFTW_CONFIG_INCLUDE_DIR ${FFTW_INCLUDE_DIR})
    ENDIF()
ELSE()
    MESSAGE(FATAL_ERROR "REQUIRED THIRDPARTY LIBRARY FFTW NOT FOUND." )
ENDIF(IBLGF_USE_FFTW)



