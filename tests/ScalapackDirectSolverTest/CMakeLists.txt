#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )
#set(MKLROOT "/home/root/intel-oneAPI/oneAPI/mkl/latest")

find_package(PETSc)
#find_package(SLEPc)
#find_package(PkgConfig REQUIRED) 
#pkg_check_modules(SLEPC_PKG REQUIRED IMPORTED_TARGET SLEPc)

INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
INCLUDE_DIRECTORIES(${SLEPC_INC})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})
#set(MKLROOT "/opt/intel-2019/mkl")
#find_package(MKL CONFIG REQUIRED)

set(NAME ScalapackSolverTest)
set(MKL_LINK static)
add_custom_target(copy_ScalapackLU ALL
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      ${CMAKE_CURRENT_SOURCE_DIR}/configs
                      ${CTEST_WORKING_DIRECTORY}/${NAME}
                      )

set(Ptest_list ScalapackLU_Test)

# add the executable
add_executable(ScalapackLU.x ScalapackLU.cpp ${SOURCES})
target_link_libraries(ScalapackLU.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      ${PETSC_LIBRARIES}
                      xtensor
                      xsimd)
                      
#target_compile_options(ScalapackLU.x PUBLIC -DMKL_ILP64  -I"/home/root/intel-oneAPI/oneAPI/mkl/latest/include")
#target_include_directories(ScalapackLU.x PUBLIC ${MKLROOT})
# target_link_libraries(MKLSolver.x PUBLIC $<LINK_ONLY:MKL::MKL_C>)

foreach (item ${Ptest_list})
    add_executable(${item} ${item}.cpp ${SOURCES})
    add_dependencies(${item} copy_ScalapackLU)
    target_link_libraries(${item} iblgf_test_main)
    add_test(
        NAME ${item}_parallel_np13
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
    )
endforeach()
