#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )
include_directories($CRAY_LIBSCI_PREFIX/include)
link_directories($CRAY_LIBSCI_PREFIX/lib)

#find_package(PETSC)
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

message("PETSC_INCLUDE_DIRS: ${PETSC_INCLUDE_DIRS}")
message("PETSC_LIBRARIES: ${PETSC_LIBRARIES}")
message("PETSC_DEFINITIONS: ${PETSC_DEFINITIONS}")
set(NAME lns_stab_petsc)
set(MKL_LINK static)
# add_custom_target(copy_ifherk_files2D ALL
#                       COMMAND ${CMAKE_COMMAND} -E copy_directory
#                       ${CMAKE_CURRENT_SOURCE_DIR}/configs
#                       ${CTEST_WORKING_DIRECTORY}/${NAME}
#                       )

# set(Ptest_list ns_amr_lgf_Test2D)

# add the executable
add_executable(lns_stab_petsc.x lns_stab_petsc.cpp ${SOURCES})
target_link_libraries(lns_stab_petsc.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      ${PETSC_LIBRARIES}
                      xtensor
                      xsimd slepc)

add_executable(lns_stab_petscTest.x lns_stab_petscTest.cpp ${SOURCES})
target_link_libraries(lns_stab_petscTest.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      ${PETSC_LIBRARIES}
                      xtensor
                      xsimd slepc)

foreach (item ${Ptest_list})
    add_executable(${item} ${item}.cpp ${SOURCES})
    add_dependencies(${item} copy_ifherk_files2D)
    target_link_libraries(${item} iblgf_test_main)
    add_test(
        NAME ${item}_parallel_np13
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
    )
endforeach()
