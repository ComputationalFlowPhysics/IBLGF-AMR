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

#include <iblgf/fmm/fmm_gpu.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace iblgf
{
namespace fmm
{
namespace gpu
{

// ============================================================================
// FMM Anterpolation Kernel - Upward Pass (leaf to root)
// ============================================================================
__global__ void fmm_anterpolation_kernel(
    const double* __restrict__ fine_data,
    double* __restrict__ coarse_data,
    const int* __restrict__ child_indices,
    const int* __restrict__ parent_indices,
    int num_octants,
    int block_size)
{
    int octant_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (octant_idx >= num_octants) return;
    
    int parent_idx = parent_indices[octant_idx];
    int child_offset = child_indices[octant_idx];
    
    // Each octant has 8 children in 3D
    int n_cells = block_size * block_size * block_size;
    
    // Trilinear restriction: average 8 fine cells into 1 coarse cell
    for (int local_idx = 0; local_idx < n_cells / 8; local_idx++)
    {
        double sum = 0.0;
        for (int child = 0; child < 8; child++)
        {
            int fine_idx = (child_offset + child) * n_cells + local_idx * 8;
            
            // Sum all 8 subcells
            for (int subcell = 0; subcell < 8; subcell++)
            {
                sum += fine_data[fine_idx + subcell];
            }
        }
        
        int coarse_idx = parent_idx * (n_cells / 8) + local_idx;
        atomicAdd(&coarse_data[coarse_idx], sum / 8.0);
    }
}

// ============================================================================
// FMM Interpolation Kernel - Downward Pass (root to leaf)
// ============================================================================
__global__ void fmm_interpolation_kernel(
    const double* __restrict__ coarse_data,
    double* __restrict__ fine_data,
    const int* __restrict__ parent_indices,
    const int* __restrict__ child_indices,
    int num_octants,
    int block_size)
{
    int octant_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (octant_idx >= num_octants) return;
    
    int parent_idx = parent_indices[octant_idx];
    int child_offset = child_indices[octant_idx];
    
    int n_cells = block_size * block_size * block_size;
    
    // Trilinear prolongation: interpolate 1 coarse cell to 8 fine cells
    for (int local_idx = 0; local_idx < n_cells / 8; local_idx++)
    {
        int coarse_idx = parent_idx * (n_cells / 8) + local_idx;
        double coarse_val = coarse_data[coarse_idx];
        
        // Distribute to 8 fine cells (simple injection for now)
        // Could be improved with higher-order interpolation
        for (int child = 0; child < 8; child++)
        {
            int fine_idx = (child_offset + child) * n_cells + local_idx * 8;
            for (int subcell = 0; subcell < 8; subcell++)
            {
                fine_data[fine_idx + subcell] += coarse_val;
            }
        }
    }
}

// ============================================================================
// FMM Near-Field Kernel - Direct Evaluation for Neighbor Interactions
// ============================================================================
__global__ void fmm_near_field_kernel(
    const GPUOctant* __restrict__ octants,
    const GPUInteractionList* __restrict__ interaction_lists,
    const double* __restrict__ kernel_values,
    int num_target_octants,
    double dx,
    int block_size)
{
    int target_idx = blockIdx.x;
    int local_idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    
    if (target_idx >= num_target_octants) return;
    
    const GPUOctant& target = octants[target_idx];
    const GPUInteractionList& list = interaction_lists[target_idx];
    
    int n_cells = block_size * block_size * block_size;
    
    if (local_idx >= n_cells) return;
    
    // Compute target cell position
    int iz = local_idx / (block_size * block_size);
    int iy = (local_idx % (block_size * block_size)) / block_size;
    int ix = local_idx % block_size;
    
    double target_pos[3] = {
        target.center[0] + (ix - block_size/2.0) * dx,
        target.center[1] + (iy - block_size/2.0) * dx,
        target.center[2] + (iz - block_size/2.0) * dx
    };
    
    double accumulated = 0.0;
    
    // Loop over all source octants in interaction list
    for (int s = 0; s < list.num_interactions; s++)
    {
        int source_idx = list.source_indices[s];
        const GPUOctant& source = octants[source_idx];
        
        // Loop over all source cells
        for (int src_cell = 0; src_cell < n_cells; src_cell++)
        {
            int sz = src_cell / (block_size * block_size);
            int sy = (src_cell % (block_size * block_size)) / block_size;
            int sx = src_cell % block_size;
            
            double source_pos[3] = {
                source.center[0] + (sx - block_size/2.0) * dx,
                source.center[1] + (sy - block_size/2.0) * dx,
                source.center[2] + (sz - block_size/2.0) * dx
            };
            
            // Compute distance
            double r2 = 0.0;
            for (int d = 0; d < 3; d++) {
                double diff = target_pos[d] - source_pos[d];
                r2 += diff * diff;
            }
            
            double r = sqrt(r2);
            if (r < 1e-10) continue;  // Skip self-interaction
            
            // Laplacian Green's function: -1/(4πr)
            double kernel = -0.25 / (M_PI * r);
            double source_val = source.source_data[src_cell];
            
            accumulated += kernel * source_val;
        }
    }
    
    // Accumulate to target
    atomicAdd(&target.target_data[local_idx], accumulated * dx * dx * dx);
}

// ============================================================================
// LGF Evaluation Kernel - Precompute Lattice Green's Function
// ============================================================================
__global__ void lgf_evaluation_kernel(
    double* __restrict__ lgf_values,
    int* __restrict__ offsets,
    int num_offsets,
    double dx,
    double alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_offsets) return;
    
    // Extract 3D offset from linearized index
    int oz = offsets[idx * 3 + 2];
    int oy = offsets[idx * 3 + 1];
    int ox = offsets[idx * 3 + 0];
    
    double r2 = (ox*ox + oy*oy + oz*oz) * dx * dx;
    double r = sqrt(r2);
    
    if (r < 1e-12) {
        lgf_values[idx] = 0.0;  // Regularize self-term
        return;
    }
    
    if (alpha < 1e-10) {
        // Laplacian: -1/(4πr)
        lgf_values[idx] = -0.25 / (M_PI * r);
    }
    else {
        // Helmholtz: -exp(-αr)/(4πr)
        lgf_values[idx] = -exp(-alpha * r) / (4.0 * M_PI * r);
    }
}

// ============================================================================
// Batched Field Copy Kernel
// ============================================================================
__global__ void batched_field_copy_kernel(
    const double* const* __restrict__ src_ptrs,
    double* const* __restrict__ dst_ptrs,
    const int* __restrict__ sizes,
    int num_copies)
{
    int copy_idx = blockIdx.y;
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (copy_idx >= num_copies) return;
    if (elem_idx >= sizes[copy_idx]) return;
    
    dst_ptrs[copy_idx][elem_idx] = src_ptrs[copy_idx][elem_idx];
}

// ============================================================================
// Masked Field Accumulate Kernel
// ============================================================================
__global__ void masked_field_accumulate_kernel(
    const double* __restrict__ source,
    double* __restrict__ target,
    const bool* __restrict__ mask,
    double scale,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    if (mask[idx]) {
        target[idx] += scale * source[idx];
    }
}

// ============================================================================
// FMMTreeGPU Implementation
// ============================================================================
FMMTreeGPU::FMMTreeGPU()
    : d_octants_(nullptr)
    , d_interaction_lists_(nullptr)
    , num_octants_(0)
    , num_lists_(0)
{
    cudaStreamCreate(&stream_);
}

FMMTreeGPU::~FMMTreeGPU()
{
    if (d_octants_) cudaFree(d_octants_);
    if (d_interaction_lists_) cudaFree(d_interaction_lists_);
    cudaStreamDestroy(stream_);
}

void FMMTreeGPU::upload_octants(const std::vector<GPUOctant>& octants)
{
    num_octants_ = octants.size();
    size_t bytes = num_octants_ * sizeof(GPUOctant);
    
    if (d_octants_) cudaFree(d_octants_);
    cudaMalloc(&d_octants_, bytes);
    cudaMemcpyAsync(d_octants_, octants.data(), bytes, 
                    cudaMemcpyHostToDevice, stream_);
}

void FMMTreeGPU::upload_interaction_lists(const std::vector<GPUInteractionList>& lists)
{
    num_lists_ = lists.size();
    size_t bytes = num_lists_ * sizeof(GPUInteractionList);
    
    if (d_interaction_lists_) cudaFree(d_interaction_lists_);
    cudaMalloc(&d_interaction_lists_, bytes);
    cudaMemcpyAsync(d_interaction_lists_, lists.data(), bytes,
                    cudaMemcpyHostToDevice, stream_);
}

void FMMTreeGPU::download_results(std::vector<double>& target_data)
{
    cudaStreamSynchronize(stream_);
    // Download would need to gather target data from all octants
    // Implementation depends on data layout
}

// ============================================================================
// FMMBatchedFFT Implementation
// ============================================================================
FMMBatchedFFT::FMMBatchedFFT(int max_block_size, int max_batch_size)
    : max_block_size_(max_block_size)
    , max_batch_size_(max_batch_size)
{
    cudaStreamCreate(&stream_);
    
    // Create batched 3D FFT plans
    int n[3] = {max_block_size, max_block_size, max_block_size};
    cufftPlanMany(&plan_forward_, 3, n,
                  NULL, 1, max_block_size * max_block_size * max_block_size,
                  NULL, 1, (max_block_size/2 + 1) * max_block_size * max_block_size,
                  CUFFT_D2Z, max_batch_size);
    
    cufftPlanMany(&plan_inverse_, 3, n,
                  NULL, 1, (max_block_size/2 + 1) * max_block_size * max_block_size,
                  NULL, 1, max_block_size * max_block_size * max_block_size,
                  CUFFT_Z2D, max_batch_size);
    
    cufftSetStream(plan_forward_, stream_);
    cufftSetStream(plan_inverse_, stream_);
}

FMMBatchedFFT::~FMMBatchedFFT()
{
    cufftDestroy(plan_forward_);
    cufftDestroy(plan_inverse_);
    cudaStreamDestroy(stream_);
}

void FMMBatchedFFT::execute_forward_batch(
    double** source_ptrs,
    cufftDoubleComplex** spectrum_ptrs,
    int* sizes,
    int batch_size)
{
    // Execute batched forward FFT
    // Note: This is simplified - real implementation would handle
    // different sizes and proper data staging
    for (int i = 0; i < batch_size; i++) {
        cufftExecD2Z(plan_forward_, 
                     (cufftDoubleReal*)source_ptrs[i],
                     spectrum_ptrs[i]);
    }
}

void FMMBatchedFFT::execute_inverse_batch(
    cufftDoubleComplex** spectrum_ptrs,
    double** target_ptrs,
    int* sizes,
    int batch_size)
{
    // Execute batched inverse FFT
    for (int i = 0; i < batch_size; i++) {
        cufftExecZ2D(plan_inverse_,
                     spectrum_ptrs[i],
                     (cufftDoubleReal*)target_ptrs[i]);
    }
}

} // namespace gpu
} // namespace fmm
} // namespace iblgf
