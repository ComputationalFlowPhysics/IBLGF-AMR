//      ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
//      ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌
//          ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄
//          ▐░▌     ▐░░░░░░░░░░▌ ▐░▌          ▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌
//          ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌
//      ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌
//      ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀

#ifndef IBLGF_INCLUDED_OPERATORS_GPU_CUH
#define IBLGF_INCLUDED_OPERATORS_GPU_CUH

#include <cuda_runtime.h>
#include <iblgf/types.hpp>

namespace iblgf
{
namespace domain
{
namespace gpu
{

/**
 * @brief CUDA kernel for computing curl transpose on face-centered data
 * Accelerates velocity field computation from streamfunction
 */
__global__ void curl_transpose_kernel_3d(
    const double* __restrict__ stream_x,
    const double* __restrict__ stream_y,
    const double* __restrict__ stream_z,
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    int nx, int ny, int nz,
    double inv_dx,
    double scale);

/**
 * @brief CUDA kernel for computing Laplacian on cell-centered data
 * Accelerates source term computation in Poisson solver
 */
__global__ void laplace_kernel_3d(
    const double* __restrict__ input,
    double* __restrict__ output,
    int nx, int ny, int nz,
    double inv_dx2);

/**
 * @brief CUDA kernel for max norm computation
 * Used for AMR refinement criteria
 */
__global__ void maxnorm_reduction_kernel(
    const double* __restrict__ data,
    double* __restrict__ partial_max,
    int n);

/**
 * @brief CUDA kernel for field coarsening (restriction operator)
 * Accelerates multigrid-style coarsening in FMM
 */
__global__ void coarsen_field_kernel_3d(
    const double* __restrict__ fine,
    double* __restrict__ coarse,
    int nx_fine, int ny_fine, int nz_fine);

/**
 * @brief CUDA kernel for field interpolation (prolongation operator)
 * Accelerates multigrid-style interpolation in FMM
 */
__global__ void interpolate_field_kernel_3d(
    const double* __restrict__ coarse,
    double* __restrict__ fine,
    int nx_coarse, int ny_coarse, int nz_coarse);

/**
 * @brief CUDA kernel for element-wise field addition with scaling
 * Accelerates field accumulation operations
 */
__global__ void field_axpy_kernel(
    const double* __restrict__ x,
    double* __restrict__ y,
    double alpha,
    int n);

/**
 * @brief CUDA kernel for field zeroing
 * Faster than host-side memset for large arrays
 */
__global__ void field_zero_kernel(
    double* __restrict__ data,
    int n);

/**
 * @brief Helper class for managing GPU memory pools
 * Reduces allocation overhead during solver iterations
 */
class GPUMemoryPool
{
public:
    GPUMemoryPool(size_t initial_capacity = 1024*1024*128); // 128 MB default
    ~GPUMemoryPool();
    
    void* allocate(size_t bytes);
    void deallocate(void* ptr);
    void clear();
    
private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<Block> blocks_;
    size_t capacity_;
    cudaStream_t stream_;
};

/**
 * @brief Helper class for asynchronous GPU operations
 * Manages CUDA streams for overlapping computation and communication
 */
class GPUStreamManager
{
public:
    GPUStreamManager(int num_streams = 4);
    ~GPUStreamManager();
    
    cudaStream_t get_stream(int idx);
    void synchronize_all();
    
private:
    std::vector<cudaStream_t> streams_;
};

} // namespace gpu
} // namespace domain
} // namespace iblgf

#endif // IBLGF_INCLUDED_OPERATORS_GPU_CUH
