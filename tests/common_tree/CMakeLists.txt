#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )
add_executable(symgrid3D.x symgrid.cpp ${SOURCES})
target_link_libraries(symgrid3D.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)
# add the executable
add_executable(common_tree.x common_tree.cpp ${SOURCES})
target_link_libraries(common_tree.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)

add_executable(adapt_to_ref.x adapt_to_ref.cpp ${SOURCES})
target_link_libraries(adapt_to_ref.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)

add_executable(read_write_test.x read_write_test.cpp ${SOURCES})
target_link_libraries(read_write_test.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)

set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
message("${PETSC}")
message("${PKG_CONFIG_PATH}")
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)


#INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
INCLUDE_DIRECTORIES(${PETSC_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${SLEPC_INC})
#ADD_DEFINITIONS(${PETSC_DEFINITIONS})
add_definitions(${PETSC_CFLAGS_OTHER})
link_directories(${PETSC_LIBRARY_DIRS})

add_executable(test_POD.x test_POD.cpp ${SOURCES})
target_link_libraries(test_POD.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd
                      ${PETSC_LIBRARIES}
                      slepc)