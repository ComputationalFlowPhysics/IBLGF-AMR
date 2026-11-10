#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )
include_directories($CRAY_LIBSCI_PREFIX/include)
link_directories($CRAY_LIBSCI_PREFIX/lib)

set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
message("${PETSC}")
message("${PKG_CONFIG_PATH}")
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)

INCLUDE_DIRECTORIES(${PETSC_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${SLEPC_INC})
add_definitions(${PETSC_CFLAGS_OTHER})
link_directories(${PETSC_LIBRARY_DIRS})

message("PETSC_INCLUDE_DIRS: ${PETSC_INCLUDE_DIRS}")
message("PETSC_LIBRARIES: ${PETSC_LIBRARIES}")
message("PETSC_DEFINITIONS: ${PETSC_DEFINITIONS}")

set(NAME BaseFlow2D)

# add the executable
add_executable(BaseFlow2D.x BaseFlow2D.cpp ${SOURCES})
target_link_libraries(BaseFlow2D.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      ${PETSC_LIBRARIES}
                      xtensor
                      xsimd slepc)
