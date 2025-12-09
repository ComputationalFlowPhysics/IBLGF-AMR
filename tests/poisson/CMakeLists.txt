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

set(NAME vortexrings)
# Testing --------------------------------------------------------------------
# copy config files
#add_custom_target(copy_files ALL
#                      COMMAND ${CMAKE_COMMAND} -E copy_directory
#                      ${CMAKE_CURRENT_SOURCE_DIR}/configs
#                      ${CTEST_WORKING_DIRECTORY}/${NAME}/configs
#                      )

# add the executable
add_executable(poisson.x poisson.cpp ${SOURCES})
target_link_libraries(poisson.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)
set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --max-errors=1")

add_executable(poisson_gpu.x poisson_gpu.cu ${SOURCES} ${GPU_SOURCES})
target_link_libraries(poisson_gpu.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd
                      ${CUDA_CUFFT_LIBRARIES} cufft)
set_target_properties(poisson_gpu.x PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
)
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
