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

#ifndef INCLUDED_CONVOLUTION_GPU_IBLGF_HPP
#define INCLUDED_CONVOLUTION_GPU_IBLGF_HPP
#include <complex>
#include <cstring>
#include <cufft.h>
#include <iblgf/types.hpp>
#include <vector>

namespace iblgf
{
namespace fft
{
using float_type = double;

// CUDA kernel for element-wise complex multiplication and addition (all on GPU)
// For batched operation with per-batch f0 pointers
__global__ void prod_complex_add_ptr(
    const cuDoubleComplex* const* f0_ptrs,
    const cuDoubleComplex* output,
    cuDoubleComplex* result,
    const size_t* f0_sizes,
    int batch_size,
    size_t output_size_per_batch);
__global__ void sum_batches(
    const cuDoubleComplex* input,
    cuDoubleComplex* output,
    int n_batches,
    size_t size_per_batch);

// Scale complex array on device: data[i] *= alpha
__global__ void scale_complex(
    cuDoubleComplex* data,
    size_t size,
    double alpha);


class dfft_r2c_gpu
{
  public:
    using float_type = double;
    // using complex_vector_t = std::vector<std::complex<float_type>,
    //     xsimd::aligned_allocator<std::complex<float_type>, 32>>;
    // using real_vector_t =
    //     std::vector<float_type, xsimd::aligned_allocator<float_type, 32>>;
    using dims_3D = types::vector_type<int, 3>;
    using dims_2D = types::vector_type<int, 2>;

  public: //Ctors:
    dfft_r2c_gpu(const dfft_r2c_gpu& other) = default;
    dfft_r2c_gpu(dfft_r2c_gpu&& other) = default;
    dfft_r2c_gpu& operator=(const dfft_r2c_gpu& other) & = default;
    dfft_r2c_gpu& operator=(dfft_r2c_gpu&& other) & = default;
    ~dfft_r2c_gpu();

    dfft_r2c_gpu(dims_3D _dims_padded, dims_3D _dims_non_zero);
    // dfft_r2c_gpu(dims_2D _dims_padded, dims_2D _dims_non_zero);

  public: //Interface
    void         execute_whole();
    void         execute();
    void         execute_ptr();
    inline auto& input() { return input_; }
    inline auto& output() { return output_; }
    inline const auto& output() const { return output_; }
    inline auto  output_copy() { return output_; }
        inline auto& output_cu() { return output_cu_; }

    template<class Vector>
    void copy_input(const Vector& _v, dims_3D _dims_v);
    template<class Vector>
    void copy_field(const Vector& _v, dims_3D _dims_v) noexcept
    {
        const int dim0 = dims_input_3D[0];
        const int dim1 = dims_input_3D[1];
        const int plane_size = dim0 * dim1;
        
        for (int k = 0; k < _dims_v[2]; ++k)
        {
            for (int j = 0; j < _dims_v[1]; ++j)
            {
                const int base_idx = j * dim0 + k * plane_size;
                for (int i = 0; i < _dims_v[0]; ++i)
                {
                    input_[base_idx + i] = _v.get_real_local(i, j, k);
                }
            }
        }
    }

  private:
    dims_3D                               dims_input_3D;
    dims_2D                               dims_input_2D;
    std::vector<float_type>               input_;
    std::vector<std::complex<float_type>> output_;
    float_type*                           input_cu_;
    cufftDoubleComplex*                   output_cu_;

    cufftHandle plan;
        cudaStream_t                          stream_;
};

class dfft_r2c_gpu_batch
{
  public:
    using dims_3D = types::vector_type<int, 3>;

  public: //Ctors:
    dfft_r2c_gpu_batch(const dfft_r2c_gpu_batch& other) = default;
    dfft_r2c_gpu_batch(dfft_r2c_gpu_batch&& other) = default;
    dfft_r2c_gpu_batch& operator=(const dfft_r2c_gpu_batch& other) & = default;
    dfft_r2c_gpu_batch& operator=(dfft_r2c_gpu_batch&& other) & = default;
    ~dfft_r2c_gpu_batch();

    dfft_r2c_gpu_batch(dims_3D _dims_padded, dims_3D _dims_non_zero, int batch_size);

  public: //Interface
    // void         execute_whole();
    inline auto& input() { return input_; }
    inline auto& output() { return output_; }
    inline const auto& output() const { return output_; }
    inline auto  output_copy() { return output_; }
    inline auto& output_cu() { return output_cu_; }
    inline auto& f0_ptrs() { return f0_ptrs_; }
    inline auto& f0_sizes() { return f0_sizes_; }
    inline auto& input_cu() { return input_cu_; }
    inline auto& result_cu() { return result_cu_; }
    inline size_t result_size() const { return output_.size() * max_batch_size_; }
    inline size_t input_size() const { return static_cast<size_t>(dims_input_3D[0]) * dims_input_3D[1] * dims_input_3D[2] * max_batch_size_; }
    inline auto& stream() { return stream_; }
    inline auto& transfer_stream() { return transfer_stream_; }
    
    // GPU-direct copy using cudaMemcpy3D - bypasses CPU loops entirely
    template<class Vector>
    void copy_field_gpu(const Vector& _v, dims_3D _dims_v, int _batch_idx) noexcept
    {
        // Source dimensions (DataField's actual allocated extent including ghost zones)
        const auto& src_block = _v.real_block();
        const auto& src_ext = src_block.extent();
        
        // Destination dimensions (GPU padded buffer)
        const int dst_nx = dims_input_3D[0];
        const int dst_ny = dims_input_3D[1];
        const int dst_nz = dims_input_3D[2];
        
        // Calculate byte offset for this batch in the GPU buffer
        const size_t slot_size = static_cast<size_t>(dst_nx) * dst_ny * dst_nz;
        const size_t batch_offset_elems = static_cast<size_t>(_batch_idx) * slot_size;
        
        // NOTE: No per-batch memset needed - buffer initialized once at construction
        // and cleaned up after batch execution. This eliminates 9% memset overhead.
        
        cudaMemcpy3DParms p = {0};
        
        // Source: DataField host memory with its pitch/stride
        p.srcPtr = make_cudaPitchedPtr(
            (void*)&_v.get_real_local(0, 0, 0),
            src_ext[0] * sizeof(float_type),  // pitch in bytes
            src_ext[0],                        // width in elements
            src_ext[1]                         // height in elements
        );
        
        // Destination: GPU buffer for this batch slot
        p.dstPtr = make_cudaPitchedPtr(
            (void*)(input_cu_ + batch_offset_elems),
            dst_nx * sizeof(float_type),       // pitch in bytes
            dst_nx,                            // width in elements  
            dst_ny                             // height in elements
        );
        
        // Extent: actual data to copy (not including padding)
        p.extent = make_cudaExtent(
            _dims_v[0] * sizeof(float_type),   // width in bytes
            _dims_v[1],                        // height
            _dims_v[2]                         // depth
        );
        
        p.kind = cudaMemcpyHostToDevice;
        
        // Execute asynchronously on transfer stream
        cudaMemcpy3DAsync(&p, transfer_stream_);
    }
    
    // Original CPU-based copy (fallback or for debugging)
    template<class Vector>
    void copy_field(const Vector& _v, dims_3D _dims_v, int _batch_idx) noexcept
    {
        const int dim0 = dims_input_3D[0];
        const int dim1 = dims_input_3D[1];
        const int dim2 = dims_input_3D[2];
        
        const size_t plane_size = static_cast<size_t>(dim0) * dim1;
        const size_t slot_size = plane_size * dim2;
        const size_t batch_offset = static_cast<size_t>(_batch_idx) * slot_size;

        // std::cout<<"Copy field batch "<<_batch_idx<<" dims_v "<<_dims_v[0]<<" "<<_dims_v[1]<<" "<<_dims_v[2]<<std::endl;
        for (int k = 0; k < _dims_v[2]; ++k)
        {
            for (int j = 0; j < _dims_v[1]; ++j)
            {
                // Calculate starting positions for this row
                // Destination: index in your input_ array
                const size_t dest_row_start = batch_offset + (k * plane_size) + (j * dim0);
                
                // Source: Get pointer to the start of the row (i=0) in the source vector
                // This assumes _v is a DataField and i is the contiguous dimension
                const auto* src_ptr = &_v.get_real_local(0, j, k);
                auto* dest_ptr = &input_[dest_row_start];

                // Use std::copy for potential SIMD optimization
                std::copy(src_ptr, src_ptr + _dims_v[0], dest_ptr);
            }
        }
    }
    void execute_ptr(int current_batch);

  private:
    dims_3D                               dims_input_3D;
    int                                   max_batch_size_;
    float_type*                           input_;          // Pinned host memory (allocated via cudaHostAlloc)
    std::vector<std::complex<float_type>> output_;
    float_type*                           input_cu_;
    cufftDoubleComplex*                   output_cu_;
    std::vector<cufftDoubleComplex*>      f0_ptrs_;        // Pointers to external f0 data (device)
    std::vector<size_t>                   f0_sizes_;       // Sizes of each f0 entry
    cufftDoubleComplex*                   result_cu_;
    cufftHandle                           plan;
    cudaStream_t                          stream_;         // CUDA stream for asynchronous FFT/kernel operations
    cudaStream_t                          transfer_stream_; // Separate stream for HtoD transfers to enable overlap
    cudaEvent_t                           transfer_ready_event_; // Signals transfer stream completion to compute stream
};

class dfft_c2r_gpu
{
  public:
    using dims_3D = types::vector_type<int, 3>;
    using dims_2D = types::vector_type<int, 2>;

  public: //Ctors:
    dfft_c2r_gpu(const dfft_c2r_gpu& other) = default;
    dfft_c2r_gpu(dfft_c2r_gpu&& other) = default;
    dfft_c2r_gpu& operator=(const dfft_c2r_gpu& other) & = default;
    dfft_c2r_gpu& operator=(dfft_c2r_gpu&& other) & = default;
    ~dfft_c2r_gpu();

    dfft_c2r_gpu(dims_3D _dims, dims_3D _dims_small);
    // dfft_c2r_gpu(dims_2D _dims, dims_2D _dims_small);

  public: //Interface
    void         execute();
    void         execute_device();
    void         copy_output_to_host();  // Explicit async copy when needed
    float_type*  output_cu_ptr();        // Get GPU pointer for direct access
    inline auto& input() { return input_; }
    inline auto& input_cu() { return input_cu_; }
    inline auto& output() { return output_; }
    inline auto& stream() { return stream_; }  // Access CUDA stream

  private:
    std::vector<std::complex<float_type>> input_;
    std::vector<float_type>               output_;
    cufftDoubleComplex*                   input_cu_;
    float_type*                           output_cu_;

    cufftHandle plan;
        cudaStream_t                          stream_;
};

template<int Dim>
class Convolution_GPU
{
  public:
    static constexpr int dimension = Dim;
    using dims_t = types::vector_type<int, Dim>;
    using complex_vector_t = std::vector<std::complex<float_type>>;
    using real_vector_t = std::vector<float_type>;

  public:
    Convolution_GPU(const Convolution_GPU& other) = delete;
    Convolution_GPU(Convolution_GPU&& other) = default;
    Convolution_GPU& operator=(const Convolution_GPU& other) & = delete;
    Convolution_GPU& operator=(Convolution_GPU&& other) & = default;
    ~Convolution_GPU() 
    {
        if (um_f0_ptrs_) cudaFree(um_f0_ptrs_);
        if (um_f0_sizes_) cudaFree(um_f0_sizes_);
    }

    Convolution_GPU(dims_t _dims0, dims_t _dims1, int batch_size = 10)
    : padded_dims_(_dims0 + _dims1 - 1)
    , padded_dims_next_pow_2_(helper_next_pow_2(padded_dims_))
    , dims0_(_dims0)
    , dims1_(_dims1)
    , fft_forward0_(padded_dims_next_pow_2_, _dims0)
    , fft_forward1_(padded_dims_next_pow_2_, _dims1)
    , fft_forward1_batch(padded_dims_next_pow_2_, _dims1, (batch_size > 0 ? batch_size : 1))
    , fft_backward_(padded_dims_next_pow_2_, _dims1)
    , padded_size_(helper_all_prod(padded_dims_next_pow_2_))
    , tmp_prod(padded_size_, std::complex<float_type>(0.0))
    , current_batch_size_(0)
    , max_batch_size_((batch_size > 0 ? batch_size : 1))
    , um_f0_ptrs_(nullptr)
    , um_f0_sizes_(nullptr)
    , um_capacity_((batch_size > 0 ? batch_size : 1))
    {
        // Allocate unified memory for LGF pointers and sizes
        cudaMallocManaged(&um_f0_ptrs_, um_capacity_ * sizeof(cufftDoubleComplex*));
        cudaMallocManaged(&um_f0_sizes_, um_capacity_ * sizeof(size_t));
    }

    dims_t helper_next_pow_2(dims_t v)
    {
        dims_t tmp;
        for (int i = 0; i < dimension; i++) { tmp[i] = v[i]; }
        return tmp;
    }
    int helper_all_prod(dims_t v)
    {
        int tmp = 1;
        for (int i = 0; i < dimension; i++) { tmp *= v[i]; }
        return tmp;
    }

  public: //Members:
    cufftDoubleComplex* dft_r2c(std::vector<float_type>& _vec)
    {
        fft_forward0_.copy_input(_vec, dims0_);
        fft_forward0_.execute_whole();
        return fft_forward0_.output_cu();
    }

    size_t dft_r2c_size() const
    {
        return fft_forward0_.output().size();
    }

    void fft_backward_field_clean()
    {
        number_fwrd_executed = 0;
        current_batch_size_ = 0;
        std::fill(fft_backward_.input().begin(), fft_backward_.input().end(), 0);
        // Clean device buffers so no stale accumulation carries between solves
        cudaMemset(fft_backward_.input_cu(), 0, fft_backward_.input().size() * sizeof(cufftDoubleComplex));
        cudaMemset(fft_forward1_batch.result_cu(), 0, fft_forward1_batch.result_size() * sizeof(cufftDoubleComplex));
    }

    template<typename Source, typename BlockType, class Kernel>
    void apply_forward_add(const BlockType& _lgf_block, Kernel* _kernel, int _level_diff, const Source& _source)
    {
        execute_fwrd_field_ptr(_lgf_block, _kernel, _level_diff, _source);
    }

    // template<class Field, class BlockType, class Kernel>
    // void execute_fwrd_field(const BlockType& _lgf_block, Kernel* _kernel, int _level_diff, const Field& _b)
    // {
    //     auto& f0 = _kernel->dft_gpu(_lgf_block, padded_dims_next_pow_2_, this,
    //         _level_diff); // gets dft of LGF, checks if already computed

    //     if (f0.size() == 0) return;
    //     number_fwrd_executed++;

    //     fft_forward1_.copy_field(_b, dims1_);
    //     fft_forward1_.execute(); // forward DFT of source
    //     auto& f1 = fft_forward1_.output();

    //     complex_vector_t prod(f0.size());
    //     complex_vector_t in_backward(f0.size());
    //     prod_complex_add(f0, f1,
    //         fft_backward_.input()); // convolution by multiplication in Fourier space, placed in backward input
    //     // // all induced fields are accumulated in the backward input and then one backward FFT is performed later in fmm.compute_influence_field()
    // }

    template<class Field, class BlockType, class Kernel>
    void execute_fwrd_field_ptr(const BlockType& _lgf_block, Kernel* _kernel, int _level_diff, const Field& _b)
    {
        auto* f0_entry = _kernel->dft_gpu(_lgf_block, padded_dims_next_pow_2_, this,
            _level_diff); // gets dft of LGF, cached with device buffer
        
        if (!f0_entry || f0_entry->size == 0 || f0_entry->device == nullptr)
        {
            return;
        }
        
        // Store reference to LGF spectrum (no copy needed) and source field to batch buffers
        fft_forward1_batch.f0_ptrs().push_back(f0_entry->device);
        fft_forward1_batch.f0_sizes().push_back(f0_entry->size);
        
        // Use GPU-direct copy for maximum performance
        fft_forward1_batch.copy_field_gpu(_b, dims1_, current_batch_size_);
        
        current_batch_size_++;
        number_fwrd_executed++;
        
        // If batch is full, process it immediately
        if (current_batch_size_ >= max_batch_size_)
        {
            flush_batch();
        }
    }
    
    void flush_batch()
    {
        if (current_batch_size_ == 0) return;
        
        // Copy pointers and sizes to unified memory (accessible from both host and device)
        std::memcpy(um_f0_ptrs_, fft_forward1_batch.f0_ptrs().data(), 
                    current_batch_size_ * sizeof(cufftDoubleComplex*));
        std::memcpy(um_f0_sizes_, fft_forward1_batch.f0_sizes().data(), 
                    current_batch_size_ * sizeof(size_t));
        
        // Execute FFTs for the current batch (copy only active slots inside execute)
        fft_forward1_batch.execute_ptr(current_batch_size_);
        
        // Compute products and accumulate
        // Optimized kernel launch configuration for total workload
        size_t size_per_fft = fft_backward_.input().size();
        size_t total_elements = current_batch_size_ * size_per_fft;
        int blockSize = 256;
        int numBlocks = (total_elements + blockSize - 1) / blockSize;
        int numBlocksSum = (size_per_fft + blockSize - 1) / blockSize;
        
        // Limit grid size to avoid excessive blocks (GPU occupancy optimization)
        int maxBlocks = 2048; // Typical value for good occupancy
        numBlocks = (numBlocks > maxBlocks) ? maxBlocks : numBlocks;
        numBlocksSum = (numBlocksSum > maxBlocks) ? maxBlocks : numBlocksSum;
        
        // Allocate shared memory for caching f0_sizes (batch_size * sizeof(size_t))
        size_t shared_mem_size = current_batch_size_ * sizeof(size_t);
        
        // Launch kernels on the same stream as FFT to ensure proper ordering
        prod_complex_add_ptr<<<numBlocks, blockSize, shared_mem_size, fft_forward1_batch.stream()>>>(
            um_f0_ptrs_, 
            fft_forward1_batch.output_cu(), 
            fft_forward1_batch.result_cu(), 
            um_f0_sizes_,
            current_batch_size_,
            size_per_fft);
        
        // Sum batches into backward input on same stream
        sum_batches<<<numBlocksSum, blockSize, 0, fft_forward1_batch.stream()>>>(
            fft_forward1_batch.result_cu(), 
            fft_backward_.input_cu(), 
            current_batch_size_, 
            size_per_fft);
        
        // Ensure all GPU operations complete before clearing host-side state
        cudaStreamSynchronize(fft_forward1_batch.stream());
        
        // Reset batch counter and clear pointers
        current_batch_size_ = 0;
        fft_forward1_batch.f0_ptrs().clear();
        fft_forward1_batch.f0_sizes().clear();
    }   


    void prod_complex_add(const complex_vector_t& a, const complex_vector_t& b, complex_vector_t& res)
    {
        std::size_t size = a.size();

        for (std::size_t i = 0; i < size; ++i) { res[i] += a[i] * b[i]; }
    }

    

    auto& output() { return fft_backward_.output(); }

    template<typename Target, typename BlockType>
    void apply_backward(const BlockType& _extractor, Target& _target, float_type _extra_scale)
    {
        if (number_fwrd_executed == 0) return;
        
        // Flush any remaining items in the batch
        flush_batch();
        
        // Scale on device to avoid extra DtoH/HtoD
        float_type scale = 1.0;
        for (int i = 0; i < dimension; i++) { scale /= padded_dims_next_pow_2_[i]; }
        scale *= _extra_scale;
        const size_t size_per_fft = fft_backward_.input().size();
        int blockSize = 256;
        int numBlocks = (size_per_fft + blockSize - 1) / blockSize;
        scale_complex<<<numBlocks, blockSize, 0, fft_backward_.stream()>>>(
            fft_backward_.input_cu(), size_per_fft, scale);

        // Execute backward transform directly from device input
        fft_backward_.execute_device();
        
        // Copy result to host asynchronously
        fft_backward_.copy_output_to_host();
        
        // MUST sync before accessing host data in add_solution()
        cudaStreamSynchronize(fft_backward_.stream());
        
        // Now process results (data copy is complete)
        add_solution(_extractor, _target);
    }

    template<class Block, class Field>
    void add_solution(const Block& _b, Field& _F)
    {
        if (dimension == 3)
        {
            for (int k = dims0_[2] - 1; k < dims0_[2] + _b.extent()[2] - 1; ++k)
            {
                for (int j = dims0_[1] - 1; j < dims0_[1] + _b.extent()[1] - 1; ++j)
                {
                    for (int i = dims0_[0] - 1; i < dims0_[0] + _b.extent()[0] - 1; ++i)
                    {
                        _F.get_real_local(i - dims0_[0] + 1, j - dims0_[1] + 1, k - dims0_[2] + 1) +=
                            fft_backward_.output()[i + j * padded_dims_next_pow_2_[0] +
                                                   k * padded_dims_next_pow_2_[0] * padded_dims_next_pow_2_[1]];
                    }
                }
            }
        }
        else
        {
            for (int j = dims0_[1] - 1; j < dims0_[1] + _b.extent()[1] - 1; ++j)
            {
                for (int i = dims0_[0] - 1; i < dims0_[0] + _b.extent()[0] - 1; ++i)
                {
                    _F.get_real_local(i - dims0_[0] + 1, j - dims0_[1] + 1) +=
                        fft_backward_.output()[i + j * padded_dims_next_pow_2_[0]];
                }
            }
        }
    }

  public:
    int fft_count_ = 0;
    int number_fwrd_executed =
        0; //register the number of forward procedure applied, if no forward applied, then don't need to apply backward one as well.
    int current_batch_size_ = 0;
    int max_batch_size_ = 0;

  private:
    dims_t padded_dims_;
    dims_t padded_dims_next_pow_2_;
    dims_t dims0_;
    dims_t dims1_;

    dfft_r2c_gpu fft_forward0_;
    dfft_r2c_gpu fft_forward1_;
    dfft_r2c_gpu_batch fft_forward1_batch;
    dfft_c2r_gpu fft_backward_;

    unsigned int     padded_size_;
    complex_vector_t tmp_prod;
    
    // Unified memory for LGF pointers and sizes (no HtoD transfer needed)
    cufftDoubleComplex** um_f0_ptrs_;
    size_t*              um_f0_sizes_;
    int                  um_capacity_;
};
} // namespace fft
} // namespace iblgf

#endif //INCLUDED_CONVOLUTION_GPU_IBLGF_HPP
