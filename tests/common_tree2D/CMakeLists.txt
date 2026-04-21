#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )
# copy config files
set(NAME common_tree_2D)
add_custom_target(copy_files_petsc ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CTEST_WORKING_DIRECTORY}/${NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/configs
            ${CTEST_WORKING_DIRECTORY}/${NAME}/configs
)
# add the executable
add_executable(symgrid.x symgrid.cpp ${SOURCES})
target_link_libraries(symgrid.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)
add_executable(adapt_to_ref2D.x adapt_to_ref.cpp ${SOURCES})
target_link_libraries(adapt_to_ref2D.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)
add_executable(common_tree2D.x common_tree.cpp ${SOURCES})
target_link_libraries(common_tree2D.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)
                      
set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH}
    "${PETSC}/lib/pkgconfig:${PETSC}/lib64/pkgconfig:/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig:$ENV{PKG_CONFIG_PATH}")
message("${PETSC}")
message("$ENV{PKG_CONFIG_PATH}")
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc petsc)
pkg_search_module(SLEPC REQUIRED IMPORTED_TARGET SLEPc slepc)


#INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
INCLUDE_DIRECTORIES(${PETSC_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${SLEPC_INCLUDE_DIRS})
#ADD_DEFINITIONS(${PETSC_DEFINITIONS})
add_definitions(${PETSC_CFLAGS_OTHER})
add_definitions(${SLEPC_CFLAGS_OTHER})
link_directories(${PETSC_LIBRARY_DIRS})
link_directories(${SLEPC_LIBRARY_DIRS})

add_executable(test_POD2D.x test_POD.cpp ${SOURCES})
target_link_libraries(test_POD2D.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd
                      ${PETSC_LIBRARIES}
                      ${SLEPC_LIBRARIES})

add_executable(common_tree_Test common_tree_Test.cpp ${SOURCES})
add_dependencies(common_tree_Test copy_files_petsc)
target_link_libraries(common_tree_Test iblgf_test_main)
add_test(
    NAME common_tree_Test
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:common_tree_Test>
    WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
)

add_executable(adapt_to_ref_overlap2D_Test adapt_to_ref_overlap_Test.cpp ${SOURCES})
add_dependencies(adapt_to_ref_overlap2D_Test copy_files_petsc)
target_link_libraries(adapt_to_ref_overlap2D_Test iblgf_test_main)
add_test(
    NAME adapt_to_ref_overlap_Test_2D
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:adapt_to_ref_overlap2D_Test>
    WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
)

add_executable(symgrid2D_Test symgrid_Test.cpp ${SOURCES})
add_dependencies(symgrid2D_Test copy_files_petsc)
target_link_libraries(symgrid2D_Test iblgf_test_main)
add_test(
    NAME symgrid_Test_2D
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:symgrid2D_Test>
    WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
)

add_executable(symfield2D_Test symfield_Test.cpp ${SOURCES})
add_dependencies(symfield2D_Test copy_files_petsc)
target_link_libraries(symfield2D_Test iblgf_test_main)
add_test(
    NAME symfield_Test_2D
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:symfield2D_Test>
    WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
)

add_executable(restart2D_Test restart_Test.cpp ${SOURCES})
add_dependencies(restart2D_Test copy_files_petsc)
target_link_libraries(restart2D_Test iblgf_test_main)
add_test(
    NAME restart_Test_2D
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:restart2D_Test>
    WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
)

# add_executable(test_POD_Test test_POD_Test.cpp ${SOURCES})
# target_link_libraries(test_POD_Test
#                       iblgf_test_main
#                       ${PETSC_LIBRARIES}
#                       ${SLEPC_LIBRARIES})
# add_test(
#     NAME test_POD_Test_parallel_np13
#     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} ${IBLGF_MPI_NP} $<TARGET_FILE:test_POD_Test>
#     WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
# )
