#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

add_executable(convTest.x convTest.cpp ${SOURCES})
target_link_libraries(convTest.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd)

add_executable(convTest_gpu.x convTest_gpu.cu ${SOURCES} ${GPU_SOURCES})
target_link_libraries(convTest_gpu.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES} fftw3
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      xtensor
                      xsimd
                      ${CUDA_CUFFT_LIBRARIES} cufft)
set_target_properties(convTest_gpu.x PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
)
