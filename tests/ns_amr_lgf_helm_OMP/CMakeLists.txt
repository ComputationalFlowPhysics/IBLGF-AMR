#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

find_package(PETSc)
#find_package(SLEPc)

INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
INCLUDE_DIRECTORIES(${SLEPC_INC})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})

find_package(OpenMP)

set(NAME ns_amr_lgf_helm_OMP)
add_custom_target(copy_ifherk_files_helm_OMP ALL
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      ${CMAKE_CURRENT_SOURCE_DIR}/configs
                      ${CTEST_WORKING_DIRECTORY}/${NAME}
                      )

set(Ptest_list ns_amr_lgf_Test_helm_OMP)

# add the executable
add_executable(ns_amr_lgf_helm_OMP.x ns_amr_lgf_OMP.cpp ${SOURCES})
target_link_libraries(ns_amr_lgf_helm_OMP.x
                      PUBLIC ${MPI_LIBRARIES}
                      PUBLIC ${Boost_LIBRARIES}
                      PUBLIC ${FFTW_LIBRARIES}
                      PUBLIC ${HDF5_LIBRARIES} PUBLIC hdf5
                      PUBLIC ${BLAS_LIBRARIES}
                      PUBLIC ${PETSC_LIBRARIES}
                      PUBLIC xtensor
                      PUBLIC xsimd)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ns_amr_lgf_helm_OMP.x PUBLIC OpenMP::OpenMP_CXX)
endif()

foreach (item ${Ptest_list})
    add_executable(${item} ${item}.cpp ${SOURCES})
    add_dependencies(${item} copy_ifherk_files_helm_OMP)
    target_link_libraries(${item} PUBLIC iblgf_test_main)
    add_test(
        NAME ${item}_parallel_np13
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
    )
endforeach()
