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

# GPU-enabled Navier-Stokes solver test
include_directories(${IBLGF_SOURCE_DIR}/src)

set(NAME ns_amr_lgf_gpu)

# Copy config files
add_custom_target(copy_ns_gpu_files ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CTEST_WORKING_DIRECTORY}/${NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/configs
            ${CTEST_WORKING_DIRECTORY}/${NAME}
)

# Only build GPU version if USE_GPU is ON
if(USE_GPU)
    set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --max-errors=1")

    # Main GPU executable
    add_executable(ns_amr_lgf_gpu.x ns_amr_lgf_gpu.cu ${SOURCES} ${GPU_SOURCES})
    add_dependencies(ns_amr_lgf_gpu.x copy_ns_gpu_files)
    
    target_link_libraries(ns_amr_lgf_gpu.x
                        ${MPI_LIBRARIES}
                        ${Boost_LIBRARIES}
                        ${FFTW_LIBRARIES}
                        ${HDF5_LIBRARIES} hdf5
                        ${BLAS_LIBRARIES}
                        xtensor
                        xsimd
                        ${CUDA_CUFFT_LIBRARIES} cufft)
    
    set_target_properties(ns_amr_lgf_gpu.x PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
    )

    # Add test (optional - for automated testing)
    # Note: GPU tests may require specific hardware, so we make it optional
    if(DEFINED ENV{IBLGF_RUN_GPU_TESTS})
        add_test(
            NAME ns_amr_lgf_gpu_test
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} 
                    ${MPIEXEC_NUMPROC_FLAG} ${IBLGF_MPI_NP} 
                    $<TARGET_FILE:ns_amr_lgf_gpu.x>
            WORKING_DIRECTORY ${CTEST_WORKING_DIRECTORY}/${NAME}
        )
    endif()
else()
    message(STATUS "Skipping ns_amr_lgf_gpu - USE_GPU is OFF")
endif()
