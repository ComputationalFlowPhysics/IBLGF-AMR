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

#include <iblgf/operators/operators_gpu.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace iblgf
{
namespace domain
{
namespace gpu
{

// ============================================================================
// Curl transpose kernel - computes velocity from streamfunction
// ============================================================================
__global__ void curl_transpose_kernel_3d(
    const double* __restrict__ stream_x,
    const double* __restrict__ stream_y,
    const double* __restrict__ stream_z,
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    int nx, int ny, int nz,
    double inv_dx,
    double scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Interior points only (exclude boundaries)
    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1)
    {
        int idx = i + j * nx + k * nx * ny;
        
        // Compute curl: u = curl(stream) with face-centered staggering
        // u_x = d(stream_z)/dy - d(stream_y)/dz
        double dsy_dz = (stream_y[idx + nx*ny] - stream_y[idx - nx*ny]) * 0.5 * inv_dx;
        double dsz_dy = (stream_z[idx + nx] - stream_z[idx - nx]) * 0.5 * inv_dx;
        vel_x[idx] = scale * (dsz_dy - dsy_dz);
        
        // u_y = d(stream_x)/dz - d(stream_z)/dx
        double dsx_dz = (stream_x[idx + nx*ny] - stream_x[idx - nx*ny]) * 0.5 * inv_dx;
        double dsz_dx = (stream_z[idx + 1] - stream_z[idx - 1]) * 0.5 * inv_dx;
        vel_y[idx] = scale * (dsx_dz - dsz_dx);
        
        // u_z = d(stream_y)/dx - d(stream_x)/dy
        double dsy_dx = (stream_y[idx + 1] - stream_y[idx - 1]) * 0.5 * inv_dx;
        double dsx_dy = (stream_x[idx + nx] - stream_x[idx - nx]) * 0.5 * inv_dx;
        vel_z[idx] = scale * (dsy_dx - dsx_dy);
    }
}

// ============================================================================
// Laplacian kernel - 7-point stencil for 3D
// ============================================================================
__global__ void laplace_kernel_3d(
    const double* __restrict__ input,
    double* __restrict__ output,
    int nx, int ny, int nz,
    double inv_dx2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1)
    {
        int idx = i + j * nx + k * nx * ny;
        
        double center = input[idx];
        double lap = -6.0 * center
                   + input[idx - 1]      // x-1
                   + input[idx + 1]      // x+1
                   + input[idx - nx]     // y-1
                   + input[idx + nx]     // y+1
                   + input[idx - nx*ny]  // z-1
                   + input[idx + nx*ny]; // z+1
        
        output[idx] = lap * inv_dx2;
    }
}

// ============================================================================
// Max norm reduction kernel with warp-level optimizations
// ============================================================================
__global__ void maxnorm_reduction_kernel(
    const double* __restrict__ data,
    double* __restrict__ partial_max,
    int n)
{
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data and perform first level of reduction in global memory
    double max_val = 0.0;
    if (idx < n) {
        max_val = fabs(data[idx]);
    }
    
    // Grid-stride loop for better load balancing
    for (int i = idx + blockDim.x * gridDim.x; i < n; i += blockDim.x * gridDim.x) {
        max_val = fmax(max_val, fabs(data[i]));
    }
    
    sdata[tid] = max_val;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed)
    if (tid < 32) {
        volatile double* smem = sdata;
        smem[tid] = fmax(smem[tid], smem[tid + 32]);
        smem[tid] = fmax(smem[tid], smem[tid + 16]);
        smem[tid] = fmax(smem[tid], smem[tid + 8]);
        smem[tid] = fmax(smem[tid], smem[tid + 4]);
        smem[tid] = fmax(smem[tid], smem[tid + 2]);
        smem[tid] = fmax(smem[tid], smem[tid + 1]);
    }
    
    // Write result for this block
    if (tid == 0) {
        partial_max[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Field coarsening kernel (restriction) - trilinear averaging
// ============================================================================
__global__ void coarsen_field_kernel_3d(
    const double* __restrict__ fine,
    double* __restrict__ coarse,
    int nx_fine, int ny_fine, int nz_fine)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    int nx_coarse = nx_fine / 2;
    int ny_coarse = ny_fine / 2;
    int nz_coarse = nz_fine / 2;
    
    if (i < nx_coarse && j < ny_coarse && k < nz_coarse)
    {
        int idx_coarse = i + j * nx_coarse + k * nx_coarse * ny_coarse;
        
        // Trilinear averaging (8-point stencil)
        int i_fine = 2 * i;
        int j_fine = 2 * j;
        int k_fine = 2 * k;
        
        double sum = 0.0;
        for (int dk = 0; dk < 2; dk++) {
            for (int dj = 0; dj < 2; dj++) {
                for (int di = 0; di < 2; di++) {
                    int idx_fine = (i_fine + di) + (j_fine + dj) * nx_fine + (k_fine + dk) * nx_fine * ny_fine;
                    sum += fine[idx_fine];
                }
            }
        }
        
        coarse[idx_coarse] = sum * 0.125; // Divide by 8
    }
}

// ============================================================================
// Field interpolation kernel (prolongation) - trilinear interpolation
// ============================================================================
__global__ void interpolate_field_kernel_3d(
    const double* __restrict__ coarse,
    double* __restrict__ fine,
    int nx_coarse, int ny_coarse, int nz_coarse)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    int nx_fine = nx_coarse * 2;
    int ny_fine = ny_coarse * 2;
    int nz_fine = nz_coarse * 2;
    
    if (i < nx_fine && j < ny_fine && k < nz_fine)
    {
        // Determine coarse grid cell
        int i_coarse = i / 2;
        int j_coarse = j / 2;
        int k_coarse = k / 2;
        
        // Interpolation weights
        double wx = (i % 2) * 0.5;
        double wy = (j % 2) * 0.5;
        double wz = (k % 2) * 0.5;
        
        // Bounds checking for interpolation
        int i_c1 = min(i_coarse + 1, nx_coarse - 1);
        int j_c1 = min(j_coarse + 1, ny_coarse - 1);
        int k_c1 = min(k_coarse + 1, nz_coarse - 1);
        
        // Trilinear interpolation
        double c000 = coarse[i_coarse + j_coarse * nx_coarse + k_coarse * nx_coarse * ny_coarse];
        double c100 = coarse[i_c1 + j_coarse * nx_coarse + k_coarse * nx_coarse * ny_coarse];
        double c010 = coarse[i_coarse + j_c1 * nx_coarse + k_coarse * nx_coarse * ny_coarse];
        double c110 = coarse[i_c1 + j_c1 * nx_coarse + k_coarse * nx_coarse * ny_coarse];
        double c001 = coarse[i_coarse + j_coarse * nx_coarse + k_c1 * nx_coarse * ny_coarse];
        double c101 = coarse[i_c1 + j_coarse * nx_coarse + k_c1 * nx_coarse * ny_coarse];
        double c011 = coarse[i_coarse + j_c1 * nx_coarse + k_c1 * nx_coarse * ny_coarse];
        double c111 = coarse[i_c1 + j_c1 * nx_coarse + k_c1 * nx_coarse * ny_coarse];
        
        double c00 = c000 * (1.0 - wx) + c100 * wx;
        double c10 = c010 * (1.0 - wx) + c110 * wx;
        double c01 = c001 * (1.0 - wx) + c101 * wx;
        double c11 = c011 * (1.0 - wx) + c111 * wx;
        
        double c0 = c00 * (1.0 - wy) + c10 * wy;
        double c1 = c01 * (1.0 - wy) + c11 * wy;
        
        int idx_fine = i + j * nx_fine + k * nx_fine * ny_fine;
        fine[idx_fine] = c0 * (1.0 - wz) + c1 * wz;
    }
}

// ============================================================================
// AXPY kernel: y = alpha * x + y
// ============================================================================
__global__ void field_axpy_kernel(
    const double* __restrict__ x,
    double* __restrict__ y,
    double alpha,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        y[i] += alpha * x[i];
    }
}

// ============================================================================
// Field zeroing kernel
// ============================================================================
__global__ void field_zero_kernel(
    double* __restrict__ data,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        data[i] = 0.0;
    }
}

// ============================================================================
// GPU Memory Pool Implementation
// ============================================================================
GPUMemoryPool::GPUMemoryPool(size_t initial_capacity)
    : capacity_(initial_capacity)
{
    cudaStreamCreate(&stream_);
}

GPUMemoryPool::~GPUMemoryPool()
{
    for (auto& block : blocks_) {
        if (block.ptr) {
            cudaFree(block.ptr);
        }
    }
    cudaStreamDestroy(stream_);
}

void* GPUMemoryPool::allocate(size_t bytes)
{
    // Find first available block of sufficient size
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= bytes) {
            block.in_use = true;
            return block.ptr;
        }
    }
    
    // Allocate new block if none available
    void* ptr;
    cudaMalloc(&ptr, bytes);
    blocks_.push_back({ptr, bytes, true});
    return ptr;
}

void GPUMemoryPool::deallocate(void* ptr)
{
    for (auto& block : blocks_) {
        if (block.ptr == ptr) {
            block.in_use = false;
            return;
        }
    }
}

void GPUMemoryPool::clear()
{
    for (auto& block : blocks_) {
        block.in_use = false;
    }
}

// ============================================================================
// GPU Stream Manager Implementation
// ============================================================================
GPUStreamManager::GPUStreamManager(int num_streams)
{
    streams_.resize(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams_[i]);
    }
}

GPUStreamManager::~GPUStreamManager()
{
    for (auto stream : streams_) {
        cudaStreamDestroy(stream);
    }
}

cudaStream_t GPUStreamManager::get_stream(int idx)
{
    return streams_[idx % streams_.size()];
}

void GPUStreamManager::synchronize_all()
{
    for (auto stream : streams_) {
        cudaStreamSynchronize(stream);
    }
}

} // namespace gpu
} // namespace domain
} // namespace iblgf
