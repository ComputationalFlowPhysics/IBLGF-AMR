#Bring the headers into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

set(NAME ns_amr_lgf_gpu_new)
add_custom_target(copy_ifherk_files_gpu_new ALL
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      ${CMAKE_CURRENT_SOURCE_DIR}/configs
                      ${CTEST_WORKING_DIRECTORY}/${NAME}
                      )
set(Ptest_list_gpu ns_amr_lgf_gpu_new_Test)
if(USE_GPU)
    add_custom_command(TARGET copy_ifherk_files_gpu_new POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CTEST_WORKING_DIRECTORY}/${NAME}
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/configs
                ${CTEST_WORKING_DIRECTORY}/${NAME}
    )

    set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --max-errors=1")

    add_executable(${NAME}.x ns_amr_lgf_gpu_new.cu ${SOURCES} ${GPU_SOURCES})
    add_dependencies(${NAME}.x copy_ifherk_files_gpu_new)
    target_compile_definitions(${NAME}.x PUBLIC IBLGF_GPU_RESIDENT)
    target_link_libraries(${NAME}.x
                        ${MPI_LIBRARIES}
                        ${Boost_LIBRARIES}
                        ${FFTW_LIBRARIES}
                        ${HDF5_LIBRARIES} hdf5
                        ${BLAS_LIBRARIES}
                        xtensor
                        xsimd
                        ${CUDA_CUFFT_LIBRARIES} cufft)
    set_target_properties(${NAME}.x PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
    )

endif()
if(USE_GPU)
    foreach (item ${Ptest_list_gpu})
        add_executable(${item} ${item}.cu ${SOURCES} ${GPU_SOURCES})
        add_dependencies(${item} copy_ifherk_files_gpu_new)
        target_compile_definitions(${item} PUBLIC IBLGF_GPU_RESIDENT)
        target_link_libraries(${item} iblgf_test_main
                            ${MPI_LIBRARIES}
                            ${Boost_LIBRARIES}
                            ${FFTW_LIBRARIES}
                            ${HDF5_LIBRARIES} hdf5
                            ${BLAS_LIBRARIES}
                            xtensor
                            xsimd
                            ${CUDA_CUFFT_LIBRARIES} cufft)
        set_target_properties(${item} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 17
        )
        add_test(
            NAME ${item}_parallel_np13
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} ${IBLGF_MPI_NP} $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
            #COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 10 $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
        )
    endforeach()
endif()
