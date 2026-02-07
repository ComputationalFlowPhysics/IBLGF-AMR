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

set(NAME convolution)

# Build the test executable
add_executable(convolution_test.x convolution_test.cpp ${SOURCES})

target_link_libraries(convolution_test.x
    iblgf_test_main
    ${MPI_LIBRARIES}
    ${Boost_LIBRARIES}
    ${FFTW_LIBRARIES}
    ${HDF5_LIBRARIES}
    hdf5
    ${BLAS_LIBRARIES}
    xtensor
    xsimd
)

# Register the test with CTest
add_test(
    NAME convolution_test
    COMMAND $<TARGET_FILE:convolution_test.x>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# LGF Convolution Test
add_executable(convolution_lgf_test.x convolution_lgf_test.cpp ${SOURCES})

target_link_libraries(convolution_lgf_test.x
    iblgf_test_main
    ${MPI_LIBRARIES}
    ${Boost_LIBRARIES}
    ${FFTW_LIBRARIES}
    ${HDF5_LIBRARIES}
    hdf5
    ${BLAS_LIBRARIES}
    xtensor
    xsimd
)

add_test(
    NAME convolution_lgf_test
    COMMAND $<TARGET_FILE:convolution_lgf_test.x>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# GPU Test (optional, only if CUDA is enabled)
if(USE_GPU)
    enable_language(CUDA)
    set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --max-errors=1")
    
    add_executable(convolution_gpu_test.x convolution_gpu_test.cu ${SOURCES} ${GPU_SOURCES})
    target_link_libraries(convolution_gpu_test.x
        iblgf_test_main
        ${MPI_LIBRARIES}
        ${Boost_LIBRARIES}
        ${FFTW_LIBRARIES}
        ${HDF5_LIBRARIES}
        hdf5
        ${BLAS_LIBRARIES}
        xtensor
        xsimd
        ${CUDA_CUFFT_LIBRARIES} cufft
    )
    set_target_properties(convolution_gpu_test.x PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
    )
    
    # Register GPU test with CTest
    add_test(
        NAME convolution_gpu_test
        COMMAND $<TARGET_FILE:convolution_gpu_test.x>
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()
