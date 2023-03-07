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
find_package(SLEPc)
#find_package(PkgConfig REQUIRED) 
#pkg_check_modules(SLEPC_PKG REQUIRED IMPORTED_TARGET SLEPc)

INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
INCLUDE_DIRECTORIES(${SLEPC_INC})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})

# set(MKLROOT "/home/root/intel-oneAPI/oneAPI/mkl/latest")





#pkg_search_module(PETSC PETSC IMPORTED_TARGET REQUIRED)
#message(STATUS "Found PETSc ${PETSC_VERSION}")
#set(MKLROOT "/opt/intel-2019/mkl")


set(NAME SLEPcLNSRes_Test)
add_custom_target(copy_SLEPcTestLNSRes ALL
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      ${CMAKE_CURRENT_SOURCE_DIR}/configs
                      ${CTEST_WORKING_DIRECTORY}/${NAME}
                      )

set(Ptest_list SLEPcLNSRes_Test)

# add the executable
add_executable(SLEPcTestLNSRes.x SLEPcTestLNSRes.cpp ${SOURCES})
target_link_libraries(SLEPcTestLNSRes.x
                      "-Wl,--start-group /home/root/intel-oneAPI/oneAPI/mkl/latest/lib/intel64/libmkl_intel_ilp64.a /home/root/intel-oneAPI/oneAPI/mkl/latest/lib/intel64/libmkl_sequential.a /home/root/intel-oneAPI/oneAPI/mkl/latest/lib/intel64/libmkl_core.a /home/root/intel-oneAPI/oneAPI/mkl/latest/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -lpthread -lm -ldl"
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      ${PETSC_LIBRARIES}
                      ${SLEPC_LIB}
                      xtensor
                      xsimd)

# target_compile_options(SLEPcTestLNS_helm.x PUBLIC -DMKL_ILP64  -I"/home/root/intel-oneAPI/oneAPI/mkl/latest/include")
# target_include_directories(SLEPcTestLNS.x PUBLIC ${MKLROOT})
# target_link_libraries(MKLSolver.x PUBLIC $<LINK_ONLY:MKL::MKL_C>)

foreach (item ${Ptest_list})
    add_executable(${item} ${item}.cpp ${SOURCES})
    add_dependencies(${item} copy_SLEPcTest)
    target_link_libraries(${item} iblgf_test_main)
    add_test(
        NAME ${item}_parallel_np13
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
    )
endforeach()
