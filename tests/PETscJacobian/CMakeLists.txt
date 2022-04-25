#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

#set(PETSC_DIR "/home/whou/lib/PETSc/petsc")
#set(PETSC_ARCH "arch-linux-c-debug")
#include_directories(${PETSC_DIR}/include )
#include_directories(${PETSC_DIR}/arch-linux-c-debug/include )
#include_directories(${PETSC_DIR}/lib/petsc/conf/variables)

#include_directories(${PETSC_DIR}/lib/petsc/conf/rules)
#include_directories(${PETSC_DIR}/lib/petsc/conf/tests)
find_package(PETSc)

INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})





#pkg_search_module(PETSC PETSC IMPORTED_TARGET REQUIRED)
#message(STATUS "Found PETSc ${PETSC_VERSION}")
#set(MKLROOT "/opt/intel-2019/mkl")


set(NAME PETscJacobian)
add_custom_target(copy_PETscJacobian ALL
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      ${CMAKE_CURRENT_SOURCE_DIR}/configs
                      ${CTEST_WORKING_DIRECTORY}/${NAME}
                      )

set(Ptest_list PETscJacobian_Test)

# add the executable
add_executable(PETscJacobian.x PETscJacobian.cpp ${SOURCES})
target_link_libraries(PETscJacobian.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      ${PETSC_LIBRARIES}
                      xtensor
                      xsimd)
# target_link_libraries(MKLSolver.x PUBLIC $<LINK_ONLY:MKL::MKL_C>)

foreach (item ${Ptest_list})
    add_executable(${item} ${item}.cpp ${SOURCES})
    add_dependencies(${item} copy_PETscJacobian)
    target_link_libraries(${item} iblgf_test_main)
    add_test(
        NAME ${item}_parallel_np13
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
    )
endforeach()
