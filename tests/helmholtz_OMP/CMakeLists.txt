#      ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
#     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
#      ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀
#          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌
#          ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄
#          ▐░▌     ▐░░░░░░░░░░▌ ▐░▌          ▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌
#          ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀
#          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌
#      ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌
#     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌
#      ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀

set(NAME vortexrings_OMP)
# Testing --------------------------------------------------------------------
# copy config files
#add_custom_target(copy_files ALL
#                      COMMAND ${CMAKE_COMMAND} -E copy_directory
#                      ${CMAKE_CURRENT_SOURCE_DIR}/configs
#                      ${CTEST_WORKING_DIRECTORY}/${NAME}/configs
#                      )

# add the executable

find_package(OpenMP)

find_package(OpenMP)


find_package(PETSc)
#find_package(SLEPc)

INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
INCLUDE_DIRECTORIES(${SLEPC_INC})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})


add_executable(helmholtzSolve_OMP.x helmholtz.cpp ${SOURCES})
target_link_libraries(helmholtzSolve_OMP.x
                      PUBLIC ${MPI_LIBRARIES}
                      PUBLIC ${Boost_LIBRARIES}
                      PUBLIC ${FFTW_LIBRARIES}
                      PUBLIC ${HDF5_LIBRARIES} PUBLIC hdf5
                      PUBLIC ${BLAS_LIBRARIES}
                      PUBLIC xtensor
                      PUBLIC xsimd)

target_link_libraries(helmholtzSolve_OMP.x PUBLIC iblgf_test_main PUBLIC ${PETSC_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(helmholtzSolve_OMP.x PUBLIC OpenMP::OpenMP_CXX)
endif()


                  #set(Ptest_list poisson)
                  #
                  #foreach (item ${Ptest_list})
                  #    add_executable(${item} ${item}.cpp ${SOURCES})
                  #    add_dependencies(${item} copy_files)
                  #    target_link_libraries(${item} iblgf_test_main)
                  #    add_test(
                  #        NAME ${item}
                  #        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
                  #    )
                  #    add_test(
                  #        NAME ${item}_parallel_np13
                  #        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 13 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
                  #    )
                  #endforeach()
