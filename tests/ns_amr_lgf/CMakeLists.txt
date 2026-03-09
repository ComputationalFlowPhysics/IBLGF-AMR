#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

set(NAME ns_amr_lgf)
add_custom_target(copy_ifherk_files ALL
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      ${CMAKE_CURRENT_SOURCE_DIR}/configs
                      ${CTEST_WORKING_DIRECTORY}/${NAME}
                      )

set(Ptest_list ns_amr_lgf_Test)
set(Ptest_list_gpu ns_amr_lgf_gpu_Test)
find_package(OpenMP)
if(USE_GPU)
    add_custom_command(TARGET copy_ifherk_files POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CTEST_WORKING_DIRECTORY}/${NAME}_gpu
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/configs
                ${CTEST_WORKING_DIRECTORY}/${NAME}_gpu
    )
endif()
# add the executable
add_executable(ns_amr_lgf.x ns_amr_lgf.cpp ${SOURCES})
target_link_libraries(ns_amr_lgf.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ns_amr_lgf.x PUBLIC OpenMP::OpenMP_CXX)
endif()
if(USE_GPU)
        
    set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --max-errors=1")

    add_executable(ns_amr_lgf_gpu.x ns_amr_lgf_gpu.cu ${SOURCES} ${GPU_SOURCES})
    target_link_libraries(ns_amr_lgf_gpu.x
                        ${MPI_LIBRARIES}
                        ${Boost_LIBRARIES}
                        ${FFTW_LIBRARIES}
                        ${HDF5_LIBRARIES} hdf5
                        ${BLAS_LIBRARIES}
                        xtensor
                        xsimd
                        ${CUDA_CUFFT_LIBRARIES} cufft)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ns_amr_lgf_gpu.x PUBLIC OpenMP::OpenMP_CXX)
    endif()
    set_target_properties(ns_amr_lgf_gpu.x PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
    )
endif()
foreach (item ${Ptest_list})
    add_executable(${item} ${item}.cpp ${SOURCES})
    add_dependencies(${item} copy_ifherk_files)
    target_link_libraries(${item} iblgf_test_main)
    add_test(
        NAME ${item}_parallel_np13
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} ${IBLGF_MPI_NP} $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
        #COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 10 $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
    )
endforeach()
if(USE_GPU)
    foreach (item ${Ptest_list_gpu})
        add_executable(${item} ${item}.cu ${SOURCES} ${GPU_SOURCES})
        add_dependencies(${item} copy_ifherk_files)
        target_link_libraries(${item} iblgf_test_main
                            ${CUDA_CUFFT_LIBRARIES} cufft)
        set_target_properties(${item} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 17
        )
        add_test(
            NAME ${item}_parallel_np13_gpu
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} ${IBLGF_MPI_NP} $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}_gpu
            #COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 10 $<TARGET_FILE:${item}> WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}_gpu
        )
    endforeach()
endif()
