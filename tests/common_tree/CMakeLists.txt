#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

set(NAME common_tree_3D)
add_custom_target(copy_files_common_tree ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CTEST_WORKING_DIRECTORY}/${NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/configs
            ${CTEST_WORKING_DIRECTORY}/${NAME}/configs
)

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
pkg_search_module(SLEPC REQUIRED IMPORTED_TARGET SLEPc)

#INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
INCLUDE_DIRECTORIES(${PETSC_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${SLEPC_INCLUDE_DIRS})
#ADD_DEFINITIONS(${PETSC_DEFINITIONS})
add_definitions(${PETSC_CFLAGS_OTHER})
add_definitions(${SLEPC_CFLAGS_OTHER})
link_directories(${PETSC_LIBRARY_DIRS})
link_directories(${SLEPC_LIBRARY_DIRS})

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
                      ${SLEPC_LIBRARIES})

add_executable(common_tree_overlap3D_Test common_tree_overlap_Test.cpp ${SOURCES})
add_dependencies(common_tree_overlap3D_Test copy_files_common_tree)
target_link_libraries(common_tree_overlap3D_Test iblgf_test_main)
add_test(
    NAME common_tree_overlap_Test_3D
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} ${IBLGF_MPI_NP} $<TARGET_FILE:common_tree_overlap3D_Test>
    WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
)

add_executable(adapt_to_ref_overlap3D_Test adapt_to_ref_overlap_Test.cpp ${SOURCES})
add_dependencies(adapt_to_ref_overlap3D_Test copy_files_common_tree)
target_link_libraries(adapt_to_ref_overlap3D_Test iblgf_test_main)
add_test(
    NAME adapt_to_ref_overlap_Test_3D
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} ${IBLGF_MPI_NP} $<TARGET_FILE:adapt_to_ref_overlap3D_Test>
    WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
)

add_executable(symgrid3D_Test symgrid_Test.cpp ${SOURCES})
add_dependencies(symgrid3D_Test copy_files_common_tree)
target_link_libraries(symgrid3D_Test iblgf_test_main)
add_test(
    NAME symgrid_Test_3D
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:symgrid3D_Test>
    WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
)
