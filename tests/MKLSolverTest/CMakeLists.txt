#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )
set(MKLROOT "/home/root/intel-oneAPI/oneAPI/mkl/latest")
#set(MKLROOT "/opt/intel-2019/mkl")
#find_package(MKL CONFIG REQUIRED)

set(NAME MKLSolver)
set(MKL_LINK static)
add_custom_target(copy_MKLSolver ALL
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      ${CMAKE_CURRENT_SOURCE_DIR}/configs
                      ${CTEST_WORKING_DIRECTORY}/${NAME}
                      )

set(Ptest_list MKLSolver_Test)

# add the executable
add_executable(MKLSolver.x MKLSolver.cpp ${SOURCES})
target_link_libraries(MKLSolver.x
                      "-Wl,--start-group /home/root/intel-oneAPI/oneAPI/mkl/latest/lib/intel64/libmkl_intel_ilp64.a /home/root/intel-oneAPI/oneAPI/mkl/latest/lib/intel64/libmkl_sequential.a /home/root/intel-oneAPI/oneAPI/mkl/latest/lib/intel64/libmkl_core.a /home/root/intel-oneAPI/oneAPI/mkl/latest/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -lpthread -lm -ldl"
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)
                      

target_compile_options(MKLSolver.x PUBLIC -DMKL_ILP64  -I"/home/root/intel-oneAPI/oneAPI/mkl/latest/include")
target_include_directories(MKLSolver.x PUBLIC ${MKLROOT})
# target_link_libraries(MKLSolver.x PUBLIC $<LINK_ONLY:MKL::MKL_C>)

foreach (item ${Ptest_list})
    add_executable(${item} ${item}.cpp ${SOURCES})
    add_dependencies(${item} copy_MKLSolver)
    target_link_libraries(${item} iblgf_test_main)
    add_test(
        NAME ${item}_parallel_np13
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
    )
endforeach()
