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

#include <iblgf/utilities/convolution_GPU.hpp>

namespace iblgf
{
namespace fft
{
// CUDA kernel for element-wise complex multiplication and addition (all on GPU)
// Highly optimized for batched operation with coalesced memory access
// __global__ void
// prod_complex_add_ptr(const cuDoubleComplex* const* f0_ptrs, const cuDoubleComplex* output, 
//                      cuDoubleComplex* result, const size_t* f0_sizes, int batch_size, size_t output_size_per_batch)
// {
//     // Cache f0_sizes in shared memory to avoid repeated global memory reads
//     extern __shared__ size_t shared_f0_sizes[];
    
//     // Cooperatively load f0_sizes into shared memory
//     for (int i = threadIdx.x; i < batch_size; i += blockDim.x)
//     {
//         shared_f0_sizes[i] = f0_sizes[i];
//     }
//     __syncthreads();
    
//     // Grid-stride loop for better memory coalescing and load balancing
//     size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     size_t stride = blockDim.x * gridDim.x;
//     size_t total_elements = static_cast<size_t>(batch_size) * output_size_per_batch;
    
//     for (size_t linear_idx = global_idx; linear_idx < total_elements; linear_idx += stride)
//     {
//         // Decompose linear index into batch and element index
//         // Using integer division and modulo
//         int batch = linear_idx / output_size_per_batch;
//         size_t idx = linear_idx % output_size_per_batch;
        
//         // Read f0_size from shared memory (much faster than global)
//         size_t f0_size = shared_f0_sizes[batch];
        
//         if (idx < f0_size)
//         {
//             // Use __ldg for read-only data to leverage texture cache
//             cuDoubleComplex f0_val = __ldg(&f0_ptrs[batch][idx]);
//             cuDoubleComplex out_val = __ldg(&output[linear_idx]);
            
//             // Multiply and store result
//             result[linear_idx] = cuCmul(f0_val, out_val);
//         }
//         else
//         {
//             // Zero out elements beyond f0_size for proper padding
//             result[linear_idx] = make_cuDoubleComplex(0.0, 0.0);
//         }
//     }
// }
__global__ void prod_complex_optimized(
    const cuDoubleComplex* const* __restrict__ f0_ptrs, 
    const cuDoubleComplex* __restrict__ output, 
    cuDoubleComplex* __restrict__ result, 
    const size_t* __restrict__ f0_sizes, 
    int batch_size, 
    size_t output_size_per_batch) 
{
    // 1. Shared memory for both sizes AND base pointers
    extern __shared__ char s_mem[];
    size_t* s_f0_sizes = (size_t*)s_mem;
    cuDoubleComplex** s_f0_ptrs = (cuDoubleComplex**)&s_f0_sizes[batch_size];

    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        s_f0_sizes[i] = f0_sizes[i];
        s_f0_ptrs[i] = (cuDoubleComplex*)f0_ptrs[i];
    }
    __syncthreads();

    size_t total_elements = (size_t)batch_size * output_size_per_batch;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         linear_idx < total_elements; 
         linear_idx += stride) 
    {
        // 2. Faster index decomposition
        int batch = linear_idx / output_size_per_batch; 
        size_t idx = linear_idx - ((size_t)batch * output_size_per_batch);
        
        cuDoubleComplex res = make_cuDoubleComplex(0.0, 0.0);
        
        // 3. Direct shared memory access for the pointer and size
        if (idx < s_f0_sizes[batch]) {
            cuDoubleComplex f0_val = s_f0_ptrs[batch][idx];
            cuDoubleComplex out_val = output[linear_idx];
            res = cuCmul(f0_val, out_val);
        }
        
        result[linear_idx] = res;
    }
}
__global__ void
sum_batches(const cuDoubleComplex* input, cuDoubleComplex* output, int batch_size, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop for better load balancing
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < size; i += stride)
    {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        
        // Unroll small batch loops for better performance
        #pragma unroll 4
        for (int b = 0; b < batch_size; ++b)
        {
            // Use __ldg for read-only input data
            sum = cuCadd(sum, __ldg(&input[i + b * size]));
        }
        
        // Atomic add for accumulation (safer for concurrent access)
        cuDoubleComplex old_val = output[i];
        output[i] = cuCadd(old_val, sum);
    }
}

// Scale complex array on device: data[i] *= alpha
__global__ void
scale_complex(cuDoubleComplex* data, size_t size, double alpha)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        cuDoubleComplex v = data[idx];
        data[idx] = make_cuDoubleComplex(alpha * cuCreal(v), alpha * cuCimag(v));
    }
}

__global__ void
pack_field_device_kernel(const float_type* src, int src_nx, int src_ny, int src_nz,
    float_type* dst, int dst_nx, int dst_ny, int dst_nz,
    int copy_nx, int copy_ny, int copy_nz, size_t batch_offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(copy_nx) * copy_ny * copy_nz;
    if (idx >= total) return;

    int i = static_cast<int>(idx % copy_nx);
    int j = static_cast<int>((idx / copy_nx) % copy_ny);
    int k = static_cast<int>(idx / (static_cast<size_t>(copy_nx) * copy_ny));

    size_t src_idx = (static_cast<size_t>(k) * src_ny + j) * src_nx + i;
    size_t dst_idx = batch_offset + (static_cast<size_t>(k) * dst_ny + j) * dst_nx + i;

    dst[dst_idx] = src[src_idx];
}

__global__ void
add_solution_device_kernel(const float_type* src, int src_nx, int src_ny, int src_nz,
    float_type* dst, int dst_nx, int dst_ny, int dst_nz,
    int copy_nx, int copy_ny, int copy_nz, int src_i0, int src_j0, int src_k0)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(copy_nx) * copy_ny * copy_nz;
    if (idx >= total) return;

    int i = static_cast<int>(idx % copy_nx);
    int j = static_cast<int>((idx / copy_nx) % copy_ny);
    int k = static_cast<int>(idx / (static_cast<size_t>(copy_nx) * copy_ny));

    size_t src_idx = (static_cast<size_t>(k + src_k0) * src_ny + (j + src_j0)) * src_nx + (i + src_i0);
    size_t dst_idx = (static_cast<size_t>(k) * dst_ny + j) * dst_nx + i;

    dst[dst_idx] += src[src_idx];
}
} // namespace fft
} // namespace iblgf

namespace iblgf
{
namespace fft
{
dfft_r2c_gpu::dfft_r2c_gpu(dims_3D _dims_padded, dims_3D _dims_non_zero)
: dims_input_3D(_dims_padded)
, input_(_dims_padded[2] * _dims_padded[1] * _dims_padded[0], 0.0)
, output_(_dims_padded[2] * _dims_padded[1] * ((_dims_padded[0] / 2) + 1))
{
    const int NX = _dims_padded[0];
    const int NY = _dims_padded[1];
    const int NZ = _dims_padded[2];

    const int NX_out = NX / 2 + 1;

    const size_t real_size = sizeof(float_type) * NX * NY * NZ;
    const size_t complex_size = sizeof(cufftDoubleComplex) * NX_out * NY * NZ;

    cudaMalloc((void**)&input_cu_, real_size);
    cudaMalloc((void**)&output_cu_, complex_size);

    // Create a stream and bind cuFFT plan to it
    cudaStreamCreate(&stream_);
    cufftPlan3d(&plan, NZ, NY, NX, CUFFT_D2Z);
    cufftSetStream(plan, stream_);

    // Pin host buffers to accelerate transfers
    cudaHostRegister(input_.data(), input_.size() * sizeof(float_type), 0);
    cudaHostRegister(output_.data(), output_.size() * sizeof(std::complex<float_type>), 0);
}

template<class Vector>
void
dfft_r2c_gpu::copy_input(const Vector& _v, dims_3D _dims_v)
{
    if (_v.size() == input_.size()) { std::copy(_v.begin(), _v.end(), input_.begin()); }
    else
    {
        std::cout << " _v.size(): " << _v.size() << " input_.size(): " << input_.size() << std::endl;
        std::cout << "_dims_v: " << _dims_v << " dims_input_3D: " << dims_input_3D << std::endl;
        throw std::runtime_error("ERROR! LGF SIZE NOT MATCHING");
    }
}

template void dfft_r2c_gpu::copy_input<std::vector<double>>(const std::vector<double>& _v, dims_3D _dims_v);

void
dfft_r2c_gpu::execute_whole()
{
    cudaMemcpyAsync(input_cu_, input_.data(), input_.size() * sizeof(float_type), cudaMemcpyHostToDevice, stream_);
    cufftExecD2Z(plan, (cufftDoubleReal*)input_cu_, (cufftDoubleComplex*)output_cu_);
    cudaStreamSynchronize(stream_);
    // i think we want to add
    // cudaMemcpy(output_.data(), output_cu_, output_.size() * sizeof(std::complex<float_type>), cudaMemcpyDeviceToHost);
}

dfft_r2c_gpu::~dfft_r2c_gpu()
{
    // Unpin host buffers
    if (!input_.empty())
    {
        cudaError_t err = cudaHostUnregister(input_.data());
        if (err != cudaSuccess) std::cerr << "cudaHostUnregister(input_) failed: " << cudaGetErrorString(err) << "\n";
    }
    if (!output_.empty())
    {
        cudaError_t err = cudaHostUnregister(output_.data());
        if (err != cudaSuccess) std::cerr << "cudaHostUnregister(output_) failed: " << cudaGetErrorString(err) << "\n";
    }
    // Free device memory if allocated
    if (input_cu_)
    {
        cudaError_t err = cudaFree(input_cu_);
        if (err != cudaSuccess) std::cerr << "cudaFree(input_cu_) failed: " << cudaGetErrorString(err) << "\n";
        input_cu_ = nullptr;
    }

    if (output_cu_)
    {
        cudaError_t err = cudaFree(output_cu_);
        if (err != cudaSuccess) std::cerr << "cudaFree(output_cu_) failed: " << cudaGetErrorString(err) << "\n";
        output_cu_ = nullptr;
    }

    // Destroy cuFFT plan if it exists
    if (plan != 0)
    { // plan default-initialized to 0 or CUFFT_INVALID_PLAN
        cufftResult res = cufftDestroy(plan);
        if (res != CUFFT_SUCCESS) std::cerr << "cufftDestroy(plan) failed: " << res << "\n";
        plan = 0;
    }

    // Destroy stream
    if (stream_)
    {
        cudaError_t err = cudaStreamDestroy(stream_);
        if (err != cudaSuccess) std::cerr << "cudaStreamDestroy(stream_) failed: " << cudaGetErrorString(err) << "\n";
        stream_ = nullptr;
    }
}

void
dfft_r2c_gpu::execute()
{
    cudaMemcpyAsync(input_cu_, input_.data(), input_.size() * sizeof(float_type), cudaMemcpyHostToDevice, stream_);
    cufftExecD2Z(plan, (cufftDoubleReal*)input_cu_, (cufftDoubleComplex*)output_cu_);
    // Copy back on same stream and synchronize
    cudaMemcpyAsync(output_.data(), output_cu_, output_.size() * sizeof(std::complex<float_type>), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
}

void
dfft_r2c_gpu::execute_ptr()
{
    // Async copy on bound stream and execute FFT
    cudaMemcpyAsync(input_cu_, input_.data(), input_.size() * sizeof(float_type),
        cudaMemcpyHostToDevice, stream_);
    cufftExecD2Z(plan, (cufftDoubleReal*)input_cu_, (cufftDoubleComplex*)output_cu_);
    cudaStreamSynchronize(stream_);
}

dfft_c2r_gpu::dfft_c2r_gpu(dims_3D _dims, dims_3D _dims_small)
: input_(_dims[2] * _dims[1] * ((_dims[0] / 2) + 1), std::complex<float_type>(0.0))
, output_(_dims[2] * _dims[1] * _dims[0], 0.0)
{
    const int NX = _dims[0];
    const int NY = _dims[1];
    const int NZ = _dims[2];

    const int NX_out = NX / 2 + 1;

    const size_t real_size = sizeof(float_type) * NX * NY * NZ;
    const size_t complex_size = sizeof(cufftDoubleComplex) * NX_out * NY * NZ;

    cudaMalloc((void**)&input_cu_, complex_size);
    cudaMalloc((void**)&output_cu_, real_size);

    // Create a stream and bind cuFFT plan to it
    cudaStreamCreate(&stream_);
    cufftPlan3d(&plan, NZ, NY, NX, CUFFT_Z2D);
    cufftSetStream(plan, stream_);

    // Pin host buffers to accelerate transfers
    cudaHostRegister(input_.data(), input_.size() * sizeof(std::complex<float_type>), 0);
    cudaHostRegister(output_.data(), output_.size() * sizeof(float_type), 0);
}
dfft_c2r_gpu::~dfft_c2r_gpu()
{
    // Unpin host buffers
    if (!input_.empty())
    {
        cudaError_t err = cudaHostUnregister(input_.data());
        if (err != cudaSuccess) std::cerr << "cudaHostUnregister(input_) failed: " << cudaGetErrorString(err) << "\n";
    }
    if (!output_.empty())
    {
        cudaError_t err = cudaHostUnregister(output_.data());
        if (err != cudaSuccess) std::cerr << "cudaHostUnregister(output_) failed: " << cudaGetErrorString(err) << "\n";
    }
    // Free device memory if allocated
    if (input_cu_)
    {
        cudaError_t err = cudaFree(input_cu_);
        if (err != cudaSuccess) std::cerr << "cudaFree(input_cu_) failed: " << cudaGetErrorString(err) << "\n";
        input_cu_ = nullptr;
    }

    if (output_cu_)
    {
        cudaError_t err = cudaFree(output_cu_);
        if (err != cudaSuccess) std::cerr << "cudaFree(output_cu_) failed: " << cudaGetErrorString(err) << "\n";
        output_cu_ = nullptr;
    }

    // Destroy cuFFT plan if it exists
    if (plan != 0)
    { // plan default-initialized to 0 or CUFFT_INVALID_PLAN
        cufftResult res = cufftDestroy(plan);
        if (res != CUFFT_SUCCESS) std::cerr << "cufftDestroy(plan) failed: " << res << "\n";
        plan = 0;
    }

    // Destroy stream
    if (stream_)
    {
        cudaError_t err = cudaStreamDestroy(stream_);
        if (err != cudaSuccess) std::cerr << "cudaStreamDestroy(stream_) failed: " << cudaGetErrorString(err) << "\n";
        stream_ = nullptr;
    }
}

void
dfft_c2r_gpu::execute()
{
    cudaMemcpyAsync(input_cu_, input_.data(), input_.size() * sizeof(std::complex<float_type>), cudaMemcpyHostToDevice, stream_);
    cufftExecZ2D(plan, (cufftDoubleComplex*)input_cu_, (cufftDoubleReal*)output_cu_);
    cudaMemcpyAsync(output_.data(), output_cu_, output_.size() * sizeof(float_type), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
}

void
dfft_c2r_gpu::execute_device()
{
    // Run FFT using existing device input buffer
    // Keep result on GPU - only copy to host when explicitly requested
    // This avoids unnecessary synchronization and DtoH overhead
    cufftExecZ2D(plan, (cufftDoubleComplex*)input_cu_, (cufftDoubleReal*)output_cu_);
    // Do NOT copy to host here - caller manages when/if to copy
}

// New method: Copy output from device to host (call only when needed)
void
dfft_c2r_gpu::copy_output_to_host()
{
    // Async copy on same stream - can overlap with other work
    cudaMemcpyAsync(output_.data(), output_cu_, output_.size() * sizeof(float_type), cudaMemcpyDeviceToHost, stream_);
}

// New method: Get raw device pointer for GPU-direct access
float_type*
dfft_c2r_gpu::output_cu_ptr()
{
    return output_cu_;
}

dfft_r2c_gpu_batch::dfft_r2c_gpu_batch(dims_3D _dims_padded, dims_3D _dims_non_zero, int _batch_size)
: dims_input_3D(_dims_padded)
, max_batch_size_(_batch_size)
, input_(nullptr)
, output_(_dims_padded[2] * _dims_padded[1] * ((_dims_padded[0] / 2) + 1) * max_batch_size_)
, stream_(nullptr)
, transfer_stream_(nullptr)
, transfer_ready_event_(nullptr)
{
    const int NX = _dims_padded[0];
    const int NY = _dims_padded[1];
    const int NZ = _dims_padded[2];

    const int NX_out = NX / 2 + 1;

    const size_t real_size = sizeof(float_type) * NX * NY * NZ * max_batch_size_;
    const size_t complex_size = sizeof(cufftDoubleComplex) * NX_out * NY * NZ * max_batch_size_;

    // Allocate pinned host memory for faster HtoD transfers
    cudaHostAlloc((void**)&input_, real_size, cudaHostAllocDefault);
    cudaMalloc((void**)&input_cu_, real_size);
    cudaMalloc((void**)&output_cu_, complex_size);
    cudaMalloc((void**)&result_cu_, complex_size);

    // Zero both buffers at init for clean state
    std::memset(input_, 0, real_size);
    cudaMemset(input_cu_, 0, real_size);

    // Create CUDA streams for asynchronous operations
    cudaStreamCreate(&stream_);
    cudaStreamCreate(&transfer_stream_);

    int n[3] = {NZ, NY, NX};
    cufftPlanMany(&plan, 3, n, NULL, 1, NZ * NY * NX, NULL, 1, NX_out * NY * NZ, CUFFT_D2Z,
        max_batch_size_);
    // Bind cuFFT plan to main compute stream after plan creation
    cufftSetStream(plan, stream_);

    // Event used to order compute stream after transfer stream without host blocking
    cudaEventCreateWithFlags(&transfer_ready_event_, cudaEventDisableTiming);
    // Event used to signal compute completion for safe buffer reuse
    cudaEventCreateWithFlags(&batch_done_event_, cudaEventDisableTiming);
    // Mark event complete initially so first batch doesn't wait
    cudaEventRecord(batch_done_event_, stream_);
}

dfft_r2c_gpu_batch::~dfft_r2c_gpu_batch()
{
    // Synchronize both streams before cleanup
    if (stream_)
    {
        cudaError_t err = cudaStreamSynchronize(stream_);
        if (err != cudaSuccess) std::cerr << "cudaStreamSynchronize(stream_) failed: " << cudaGetErrorString(err) << "\n";
    }
    if (transfer_stream_)
    {
        cudaError_t err = cudaStreamSynchronize(transfer_stream_);
        if (err != cudaSuccess) std::cerr << "cudaStreamSynchronize(transfer_stream_) failed: " << cudaGetErrorString(err) << "\n";
    }

    // Free pinned host memory if allocated
    if (input_)
    {
        cudaError_t err = cudaFreeHost(input_);
        if (err != cudaSuccess) std::cerr << "cudaFreeHost(input_) failed: " << cudaGetErrorString(err) << "\n";
        input_ = nullptr;
    }

    if (input_cu_)
    {
        cudaError_t err = cudaFree(input_cu_);
        if (err != cudaSuccess) std::cerr << "cudaFree(input_cu_) failed: " << cudaGetErrorString(err) << "\n";
        input_cu_ = nullptr;
    }

    if (output_cu_)
    {
        cudaError_t err = cudaFree(output_cu_);
        if (err != cudaSuccess) std::cerr << "cudaFree(output_cu_) failed: " << cudaGetErrorString(err) << "\n";
        output_cu_ = nullptr;
    }

    if (result_cu_)
    {
        cudaError_t err = cudaFree(result_cu_);
        if (err != cudaSuccess) std::cerr << "cudaFree(result_cu_) failed: " << cudaGetErrorString(err) << "\n";
        result_cu_ = nullptr;
    }

    if (transfer_ready_event_)
    {
        cudaError_t err = cudaEventDestroy(transfer_ready_event_);
        if (err != cudaSuccess) std::cerr << "cudaEventDestroy(transfer_ready_event_) failed: " << cudaGetErrorString(err) << "\n";
        transfer_ready_event_ = nullptr;
    }
    if (batch_done_event_)
    {
        cudaError_t err = cudaEventDestroy(batch_done_event_);
        if (err != cudaSuccess) std::cerr << "cudaEventDestroy(batch_done_event_) failed: " << cudaGetErrorString(err) << "\n";
        batch_done_event_ = nullptr;
    }

    // Destroy CUDA streams if they exist
    if (stream_)
    {
        cudaError_t err = cudaStreamDestroy(stream_);
        if (err != cudaSuccess) std::cerr << "cudaStreamDestroy(stream_) failed: " << cudaGetErrorString(err) << "\n";
        stream_ = nullptr;
    }
    if (transfer_stream_)
    {
        cudaError_t err = cudaStreamDestroy(transfer_stream_);
        if (err != cudaSuccess) std::cerr << "cudaStreamDestroy(transfer_stream_) failed: " << cudaGetErrorString(err) << "\n";
        transfer_stream_ = nullptr;
    }

    // Destroy cuFFT plan if it exists
    if (plan != 0)
    { // plan default-initialized to 0 or CUFFT_INVALID_PLAN
        cufftResult res = cufftDestroy(plan);
        if (res != CUFFT_SUCCESS) std::cerr << "cufftDestroy(plan) failed: " << res << "\n";
        plan = 0;
    }
}

void dfft_r2c_gpu_batch::execute_ptr(int current_batch)
{
    const size_t slot_elems = static_cast<size_t>(dims_input_3D[0]) * dims_input_3D[1] * dims_input_3D[2];
    const size_t copy_elems = static_cast<size_t>(current_batch) * slot_elems;
    const size_t remaining_elems = static_cast<size_t>(max_batch_size_) * slot_elems - copy_elems;
    
    // NOTE: Data is already on GPU via copy_field_gpu() which uses cudaMemcpy3D directly to input_cu_
    // We just need to wait for those async copies to complete and zero remaining slots
    
    // Order compute stream after transfer stream without blocking the host
    cudaEventRecord(transfer_ready_event_, transfer_stream_);
    cudaStreamWaitEvent(stream_, transfer_ready_event_, 0);
    
    // Zero remaining slots on main stream to ensure clean padding for FFT
    if (remaining_elems > 0)
    {
        cudaMemsetAsync(input_cu_ + copy_elems, 0, remaining_elems * sizeof(float_type), stream_);
    }
    
    // Execute batched FFT on compute stream
    cufftExecD2Z(plan, (cufftDoubleReal*)input_cu_, (cufftDoubleComplex*)output_cu_);
    // No sync needed - subsequent kernels launch on same stream with implicit ordering
}

void dfft_r2c_gpu_batch::copy_field_gpu_device(const float_type* src_device, dims_3D src_ext, dims_3D dims_v, int _batch_idx) noexcept
{
    const int dst_nx = dims_input_3D[0];
    const int dst_ny = dims_input_3D[1];
    const int dst_nz = dims_input_3D[2];

    const size_t slot_size = static_cast<size_t>(dst_nx) * dst_ny * dst_nz;
    const size_t batch_offset_elems = static_cast<size_t>(_batch_idx) * slot_size;

    cudaStreamWaitEvent(transfer_stream_, batch_done_event_, 0);

    const size_t total = static_cast<size_t>(dims_v[0]) * dims_v[1] * dims_v[2];
    if (total == 0) return;

    int blockSize = 256;
    int numBlocks = static_cast<int>((total + blockSize - 1) / blockSize);

    pack_field_device_kernel<<<numBlocks, blockSize, 0, transfer_stream_>>>(
        src_device,
        src_ext[0], src_ext[1], src_ext[2],
        input_cu_,
        dst_nx, dst_ny, dst_nz,
        dims_v[0], dims_v[1], dims_v[2],
        batch_offset_elems);
}

} //namespace fft
} // namespace iblgf
