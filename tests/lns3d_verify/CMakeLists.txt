include_directories(${IBLGF_SOURCE_DIR}/src )
include_directories($CRAY_LIBSCI_PREFIX/include)
link_directories($CRAY_LIBSCI_PREFIX/lib)

set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)

INCLUDE_DIRECTORIES(${PETSC_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${SLEPC_INC})
add_definitions(${PETSC_CFLAGS_OTHER})
link_directories(${PETSC_LIBRARY_DIRS})

set(NAME lns3d_verify)

add_executable(lns3d_verify.x lns3d_verify.cpp ${SOURCES})
target_link_libraries(lns3d_verify.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      ${PETSC_LIBRARIES}
                      xtensor
                      xsimd slepc)
