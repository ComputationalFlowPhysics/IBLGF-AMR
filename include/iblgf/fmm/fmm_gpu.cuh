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

#ifndef IBLGF_INCLUDED_FMM_GPU_CUH
#define IBLGF_INCLUDED_FMM_GPU_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include <iblgf/types.hpp>

namespace iblgf
{
namespace fmm
{
namespace gpu
{

/**
 * @brief Structure representing an octant on GPU
 * Simplified representation for GPU processing
 */
struct GPUOctant
{
    int level;
    int morton_key[3];  // Space-filling curve key
    double center[3];
    double* source_data;   // Device pointer to source field
    double* target_data;   // Device pointer to target field
    int data_size;
    bool is_leaf;
    bool is_source;
    bool is_target;
};

/**
 * @brief Structure for FMM interaction list on GPU
 */
struct GPUInteractionList
{
    int* source_indices;
    int* target_indices;
    int num_interactions;
};

/**
 * @brief CUDA kernel for parallel anterpolation (coarsening)
 * Processes multiple octants in parallel for FMM upward pass
 */
__global__ void fmm_anterpolation_kernel(
    const double* __restrict__ fine_data,
    double* __restrict__ coarse_data,
    const int* __restrict__ child_indices,
    const int* __restrict__ parent_indices,
    int num_octants,
    int block_size);

/**
 * @brief CUDA kernel for parallel interpolation
 * Processes multiple octants in parallel for FMM downward pass
 */
__global__ void fmm_interpolation_kernel(
    const double* __restrict__ coarse_data,
    double* __restrict__ fine_data,
    const int* __restrict__ parent_indices,
    const int* __restrict__ child_indices,
    int num_octants,
    int block_size);

/**
 * @brief CUDA kernel for FMM near-field interactions (direct evaluation)
 * Computes interactions between neighboring octants without FFT
 */
__global__ void fmm_near_field_kernel(
    const GPUOctant* __restrict__ octants,
    const GPUInteractionList* __restrict__ interaction_lists,
    const double* __restrict__ kernel_values,
    int num_target_octants,
    double dx,
    int block_size);

/**
 * @brief CUDA kernel for far-field LGF evaluation
 * Pre-computes LGF values for FFT-based far-field interactions
 */
__global__ void lgf_evaluation_kernel(
    double* __restrict__ lgf_values,
    int* __restrict__ offsets,
    int num_offsets,
    double dx,
    double alpha);  // Helmholtz parameter (0 for Laplace)

/**
 * @brief CUDA kernel for batched field copy operations
 * Efficiently copies multiple fields between octant data structures
 */
__global__ void batched_field_copy_kernel(
    const double* const* __restrict__ src_ptrs,
    double* const* __restrict__ dst_ptrs,
    const int* __restrict__ sizes,
    int num_copies);

/**
 * @brief CUDA kernel for mask-based field accumulation
 * Accumulates fields only for octants with specific masks set
 */
__global__ void masked_field_accumulate_kernel(
    const double* __restrict__ source,
    double* __restrict__ target,
    const bool* __restrict__ mask,
    double scale,
    int n);

/**
 * @brief Helper class for managing FMM tree data on GPU
 * Handles uploading/downloading octree structure to GPU
 */
class FMMTreeGPU
{
public:
    FMMTreeGPU();
    ~FMMTreeGPU();
    
    void upload_octants(const std::vector<GPUOctant>& octants);
    void upload_interaction_lists(const std::vector<GPUInteractionList>& lists);
    void download_results(std::vector<double>& target_data);
    
    GPUOctant* get_device_octants() { return d_octants_; }
    GPUInteractionList* get_device_lists() { return d_interaction_lists_; }
    int get_num_octants() const { return num_octants_; }
    
private:
    GPUOctant* d_octants_;
    GPUInteractionList* d_interaction_lists_;
    int num_octants_;
    int num_lists_;
    cudaStream_t stream_;
};

/**
 * @brief Helper class for batched FFT operations in FMM
 * Manages multiple FFT plans for different octant sizes
 */
class FMMBatchedFFT
{
public:
    FMMBatchedFFT(int max_block_size, int max_batch_size = 32);
    ~FMMBatchedFFT();
    
    void execute_forward_batch(
        double** source_ptrs,
        cufftDoubleComplex** spectrum_ptrs,
        int* sizes,
        int batch_size);
    
    void execute_inverse_batch(
        cufftDoubleComplex** spectrum_ptrs,
        double** target_ptrs,
        int* sizes,
        int batch_size);
    
private:
    cufftHandle plan_forward_;
    cufftHandle plan_inverse_;
    int max_block_size_;
    int max_batch_size_;
    cudaStream_t stream_;
};

/**
 * @brief Helper function to compute optimal CUDA launch configuration
 * Determines grid/block dimensions for FMM kernels
 */
inline void compute_fmm_launch_config(
    int num_elements,
    int& num_blocks,
    int& block_size)
{
    block_size = 256;  // Good default for most GPUs
    num_blocks = (num_elements + block_size - 1) / block_size;
    
    // Limit grid size for better occupancy
    const int max_blocks = 2048;
    if (num_blocks > max_blocks) {
        num_blocks = max_blocks;
    }
}

/**
 * @brief Helper function for GPU error checking
 */
inline void check_cuda_error(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

} // namespace gpu
} // namespace fmm
} // namespace iblgf

#endif // IBLGF_INCLUDED_FMM_GPU_CUH
