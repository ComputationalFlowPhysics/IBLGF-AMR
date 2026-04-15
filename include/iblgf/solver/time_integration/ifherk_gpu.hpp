#ifndef IBLGF_INCLUDED_IFHERK_GPU_HPP
#define IBLGF_INCLUDED_IFHERK_GPU_HPP

#ifdef IBLGF_COMPILE_CUDA

#include <cuda_runtime.h>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <iblgf/global.hpp>
#include <iblgf/operators/operators_gpu_kernels.hpp>

namespace iblgf
{
namespace solver
{
namespace ifherk_gpu
{

inline void cuda_check(cudaError_t err, const char* msg)
{
    if (err == cudaSuccess) return;
    std::ostringstream oss;
    oss << msg << ": " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
}

inline std::vector<float_type>& h2d_pack_buf()
{
    static thread_local std::vector<float_type> buf;
    return buf;
}

inline std::vector<float_type>& d2h_pack_buf()
{
    static thread_local std::vector<float_type> buf;
    return buf;
}

inline float_type*& d_h2d_pack_ptr()
{
    static thread_local float_type* ptr = nullptr;
    return ptr;
}

inline float_type*& d_d2h_pack_ptr()
{
    static thread_local float_type* ptr = nullptr;
    return ptr;
}

inline std::size_t& h2d_pack_capacity()
{
    static thread_local std::size_t cap = 0;
    return cap;
}

inline std::size_t& d2h_pack_capacity()
{
    static thread_local std::size_t cap = 0;
    return cap;
}

inline void ensure_pack_capacity(std::vector<float_type>& host_buf,
    float_type*& device_buf, std::size_t& capacity, std::size_t needed)
{
    if (capacity >= needed) return;
    host_buf.resize(needed);
    if (device_buf) cudaFree(device_buf);
    cuda_check(
        cudaMalloc(&device_buf, needed * sizeof(float_type)),
        "Allocate IFHERK packed transfer buffer");
    capacity = needed;
}

template<typename PtrType>
struct pointer_table_cache_slot_t
{
    std::string key;
    PtrType* d_ptrs = nullptr;
    std::size_t capacity = 0;
    std::vector<PtrType> host_shadow;
};

template<typename PtrType>
inline std::vector<pointer_table_cache_slot_t<PtrType>>& pointer_table_cache()
{
    static thread_local std::vector<pointer_table_cache_slot_t<PtrType>> cache;
    return cache;
}

template<typename PtrType>
inline PtrType* upload_pointer_table(
    const std::vector<PtrType>& host_ptrs, const char* context)
{
    auto& cache = pointer_table_cache<PtrType>();
    auto it = std::find_if(cache.begin(), cache.end(),
        [context](const auto& s) { return s.key == context; });
    if (it == cache.end())
    {
        cache.push_back(pointer_table_cache_slot_t<PtrType>{});
        it = std::prev(cache.end());
        it->key = context;
    }

    const std::size_t n = host_ptrs.size();
    if (it->capacity < n)
    {
        if (it->d_ptrs) cudaFree(it->d_ptrs);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&it->d_ptrs),
                       n * sizeof(PtrType)),
            context);
        it->capacity = n;
        it->host_shadow.clear();
    }

    const bool changed = (it->host_shadow.size() != n) ||
                         !std::equal(host_ptrs.begin(), host_ptrs.end(),
                             it->host_shadow.begin());
    if (changed && n > 0)
    {
        cuda_check(cudaMemcpy(it->d_ptrs, host_ptrs.data(),
                       n * sizeof(PtrType), cudaMemcpyHostToDevice),
            context);
        it->host_shadow.assign(host_ptrs.begin(), host_ptrs.end());
    }

    return it->d_ptrs;
}

__global__ void copy_scale_kernel(
    const float_type* src, float_type* dst, std::size_t n, float_type scale)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          static_cast<std::size_t>(threadIdx.x);
    if (i >= n) return;
    dst[i] = src[i] * scale;
}

__global__ void axpy_kernel(
    const float_type* src, float_type* dst, std::size_t n, float_type scale)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          static_cast<std::size_t>(threadIdx.x);
    if (i >= n) return;
    dst[i] += src[i] * scale;
}

__global__ void copy_scale_kernel_batched(
    const float_type* const* src, float_type* const* dst, int n_blocks,
    std::size_t n_per_block, float_type scale)
{
    const std::size_t gid = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            static_cast<std::size_t>(threadIdx.x);
    const std::size_t total = static_cast<std::size_t>(n_blocks) * n_per_block;
    if (gid >= total) return;
    const int b = static_cast<int>(gid / n_per_block);
    const std::size_t i = gid - static_cast<std::size_t>(b) * n_per_block;
    dst[b][i] = src[b][i] * scale;
}

__global__ void axpy_kernel_batched(const float_type* const* src,
    float_type* const* dst, int n_blocks, std::size_t n_per_block,
    float_type scale)
{
    const std::size_t gid = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            static_cast<std::size_t>(threadIdx.x);
    const std::size_t total = static_cast<std::size_t>(n_blocks) * n_per_block;
    if (gid >= total) return;
    const int b = static_cast<int>(gid / n_per_block);
    const std::size_t i = gid - static_cast<std::size_t>(b) * n_per_block;
    dst[b][i] += src[b][i] * scale;
}

template<class Block, class From, class To>
inline void copy_block(Block& block, float_type scale)
{
    static_assert(From::nFields() == To::nFields(),
        "number of fields doesn't match when copy");
    constexpr int threads = 256;
    for (std::size_t field_idx = 0; field_idx < From::nFields(); ++field_idx)
    {
        auto& src = block(From::tag(), field_idx);
        auto& dst = block(To::tag(), field_idx);
        src.update_device(nullptr, true);
        const auto* p_src = src.device_ptr();
        auto* p_dst = dst.device_ptr();
        const std::size_t n = src.data().size();
        if (!p_src || !p_dst)
            throw std::runtime_error("ifherk copy_block: null device pointer");
        if (n == 0) continue;
        const int blocks_1d = static_cast<int>((n + threads - 1) / threads);
        copy_scale_kernel<<<blocks_1d, threads>>>(p_src, p_dst, n, scale);
        cuda_check(cudaGetLastError(), "ifherk copy_block kernel");
        cuda_check(cudaMemcpy(dst.data().data(), p_dst, n * sizeof(float_type),
                       cudaMemcpyDeviceToHost),
            "ifherk copy_block D2H");
    }
}

template<class Block, class From, class To>
inline void add_block(Block& block, float_type scale)
{
    static_assert(From::nFields() == To::nFields(),
        "number of fields doesn't match when add");
    constexpr int threads = 256;
    for (std::size_t field_idx = 0; field_idx < From::nFields(); ++field_idx)
    {
        auto& src = block(From::tag(), field_idx);
        auto& dst = block(To::tag(), field_idx);
        src.update_device(nullptr, true);
        dst.update_device(nullptr, true);
        const auto* p_src = src.device_ptr();
        auto* p_dst = dst.device_ptr();
        const std::size_t n = src.data().size();
        if (!p_src || !p_dst)
            throw std::runtime_error("ifherk add_block: null device pointer");
        if (n == 0) continue;
        const int blocks_1d = static_cast<int>((n + threads - 1) / threads);
        axpy_kernel<<<blocks_1d, threads>>>(p_src, p_dst, n, scale);
        cuda_check(cudaGetLastError(), "ifherk add_block kernel");
        cuda_check(cudaMemcpy(dst.data().data(), p_dst, n * sizeof(float_type),
                       cudaMemcpyDeviceToHost),
            "ifherk add_block D2H");
    }
}

template<class BlockPtr, class From, class To>
inline void copy_level_batched(const std::vector<BlockPtr>& blocks,
    float_type scale)
{
    static_assert(From::nFields() == To::nFields(),
        "number of fields doesn't match when copy");
    if (blocks.empty()) return;
    constexpr int threads = 256;
    const int n_blocks = static_cast<int>(blocks.size());

    for (std::size_t field_idx = 0; field_idx < From::nFields(); ++field_idx)
    {
        std::vector<float_type*> h_src;
        std::vector<float_type*> h_dst;
        h_src.reserve(blocks.size());
        h_dst.reserve(blocks.size());

        std::size_t n_per_block = 0;
        for (std::size_t b = 0; b < blocks.size(); ++b)
        {
            auto& src = (*blocks[b])(From::tag(), field_idx);
            auto& dst = (*blocks[b])(To::tag(), field_idx);
            src.update_device(nullptr, true);
            auto* p_src = const_cast<float_type*>(src.device_ptr());
            auto* p_dst = dst.device_ptr();
            if (!p_src || !p_dst)
                throw std::runtime_error(
                    "ifherk copy_level_batched: null device pointer");
            const std::size_t n_src = src.data().size();
            const std::size_t n_dst = dst.data().size();
            if (b == 0) n_per_block = n_src;
            if (n_src != n_dst || n_src != n_per_block)
                throw std::runtime_error(
                    "ifherk copy_level_batched: heterogeneous block size");
            h_src.push_back(p_src);
            h_dst.push_back(p_dst);
        }
        if (n_per_block == 0) continue;

        float_type** d_src = upload_pointer_table(
            h_src, "Upload IFHERK copy batched src pointer table");
        float_type** d_dst = upload_pointer_table(
            h_dst, "Upload IFHERK copy batched dst pointer table");

        const std::size_t total = static_cast<std::size_t>(n_blocks) * n_per_block;
        const int grid = static_cast<int>((total + threads - 1) / threads);
        copy_scale_kernel_batched<<<grid, threads>>>(
            reinterpret_cast<const float_type* const*>(d_src),
            reinterpret_cast<float_type* const*>(d_dst), n_blocks, n_per_block,
            scale);
        cuda_check(cudaGetLastError(), "IFHERK copy batched kernel");

        auto& d2h_pack = d2h_pack_buf();
        auto& d_d2h_pack = d_d2h_pack_ptr();
        auto& d2h_cap = d2h_pack_capacity();
        ensure_pack_capacity(d2h_pack, d_d2h_pack, d2h_cap, total);

        operators_gpu_cuda::gather1_to_slab<<<grid, threads>>>(
            reinterpret_cast<const float_type* const*>(d_dst), d_d2h_pack,
            n_blocks, n_per_block);
        cuda_check(cudaGetLastError(), "IFHERK copy batched gather");
        cuda_check(cudaMemcpy(d2h_pack.data(), d_d2h_pack,
                       total * sizeof(float_type), cudaMemcpyDeviceToHost),
            "IFHERK copy batched D2H");

        for (std::size_t b = 0; b < blocks.size(); ++b)
        {
            auto& dst = (*blocks[b])(To::tag(), field_idx);
            std::copy(d2h_pack.begin() + b * n_per_block,
                d2h_pack.begin() + (b + 1) * n_per_block, dst.data().begin());
        }
    }
}

template<class BlockPtr, class From, class To>
inline void add_level_batched(const std::vector<BlockPtr>& blocks,
    float_type scale)
{
    static_assert(From::nFields() == To::nFields(),
        "number of fields doesn't match when add");
    if (blocks.empty()) return;
    constexpr int threads = 256;
    const int n_blocks = static_cast<int>(blocks.size());

    for (std::size_t field_idx = 0; field_idx < From::nFields(); ++field_idx)
    {
        std::vector<float_type*> h_src;
        std::vector<float_type*> h_dst;
        h_src.reserve(blocks.size());
        h_dst.reserve(blocks.size());

        std::size_t n_per_block = 0;
        for (std::size_t b = 0; b < blocks.size(); ++b)
        {
            auto& src = (*blocks[b])(From::tag(), field_idx);
            auto& dst = (*blocks[b])(To::tag(), field_idx);
            src.update_device(nullptr, true);
            dst.update_device(nullptr, true);
            auto* p_src = const_cast<float_type*>(src.device_ptr());
            auto* p_dst = dst.device_ptr();
            if (!p_src || !p_dst)
                throw std::runtime_error(
                    "ifherk add_level_batched: null device pointer");
            const std::size_t n_src = src.data().size();
            const std::size_t n_dst = dst.data().size();
            if (b == 0) n_per_block = n_src;
            if (n_src != n_dst || n_src != n_per_block)
                throw std::runtime_error(
                    "ifherk add_level_batched: heterogeneous block size");
            h_src.push_back(p_src);
            h_dst.push_back(p_dst);
        }
        if (n_per_block == 0) continue;

        float_type** d_src = upload_pointer_table(
            h_src, "Upload IFHERK add batched src pointer table");
        float_type** d_dst = upload_pointer_table(
            h_dst, "Upload IFHERK add batched dst pointer table");

        const std::size_t total = static_cast<std::size_t>(n_blocks) * n_per_block;
        const int grid = static_cast<int>((total + threads - 1) / threads);
        axpy_kernel_batched<<<grid, threads>>>(
            reinterpret_cast<const float_type* const*>(d_src),
            reinterpret_cast<float_type* const*>(d_dst), n_blocks, n_per_block,
            scale);
        cuda_check(cudaGetLastError(), "IFHERK add batched kernel");

        auto& d2h_pack = d2h_pack_buf();
        auto& d_d2h_pack = d_d2h_pack_ptr();
        auto& d2h_cap = d2h_pack_capacity();
        ensure_pack_capacity(d2h_pack, d_d2h_pack, d2h_cap, total);

        operators_gpu_cuda::gather1_to_slab<<<grid, threads>>>(
            reinterpret_cast<const float_type* const*>(d_dst), d_d2h_pack,
            n_blocks, n_per_block);
        cuda_check(cudaGetLastError(), "IFHERK add batched gather");
        cuda_check(cudaMemcpy(d2h_pack.data(), d_d2h_pack,
                       total * sizeof(float_type), cudaMemcpyDeviceToHost),
            "IFHERK add batched D2H");

        for (std::size_t b = 0; b < blocks.size(); ++b)
        {
            auto& dst = (*blocks[b])(To::tag(), field_idx);
            std::copy(d2h_pack.begin() + b * n_per_block,
                d2h_pack.begin() + (b + 1) * n_per_block, dst.data().begin());
        }
    }
}

template<class Block, class Source, class Target>
inline void curl_block(Block& block, float_type dx_level)
{
    auto& src0 = block(Source::tag(), 0);
    auto& src1 = block(Source::tag(), 1);
    auto& src2 = block(Source::tag(), 2);
    auto& dst0 = block(Target::tag(), 0);
    auto& dst1 = block(Target::tag(), 1);
    auto& dst2 = block(Target::tag(), 2);

    src0.update_device(nullptr, true);
    src1.update_device(nullptr, true);
    src2.update_device(nullptr, true);

    // Ensure device buffers exist for all fields used by the kernel.
    auto* p_src0 = src0.device_ptr();
    auto* p_src1 = src1.device_ptr();
    auto* p_src2 = src2.device_ptr();
    auto* p_dst0 = dst0.device_ptr();
    auto* p_dst1 = dst1.device_ptr();
    auto* p_dst2 = dst2.device_ptr();
    if (!p_src0 || !p_src1 || !p_src2 || !p_dst0 || !p_dst1 || !p_dst2)
    {
        throw std::runtime_error("ifherk curl: null device pointer");
    }

    const auto ext = src0.real_block().extent();
    const int ex = ext[0];
    const int ey = ext[1];
    const int ez = ext[2];

    const auto interior = block.descriptor().extent();
    const int nx = interior[0];
    const int ny = interior[1];
    const int nz = interior[2];

    // Some correction/non-leaf blocks can have empty interior extents.
    // CUDA launch with any zero grid dimension is invalid.
    if (nx <= 0 || ny <= 0 || nz <= 0) return;

    const auto lb = src0.lbuffer();
    const int ox = lb[0];
    const int oy = lb[1];
    const int oz = lb[2];

    const dim3 threads(8, 8, 8);
    const dim3 blocks((nx + threads.x - 1) / threads.x,
        (ny + threads.y - 1) / threads.y, (nz + threads.z - 1) / threads.z);

    // Clear any prior sticky CUDA error before launch.
    (void)cudaGetLastError();
    operators_gpu_cuda::curl_kernel<<<blocks, threads>>>(p_src0, p_src1, p_src2, p_dst0, p_dst1,
        p_dst2, ex, ey, ez, ox, oy, oz, nx, ny, nz,
        1.0 / dx_level);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        std::ostringstream oss;
        oss << "ifherk curl kernel launch: " << cudaGetErrorString(launch_err)
            << " ex/ey/ez=" << ex << "/" << ey << "/" << ez
            << " nx/ny/nz=" << nx << "/" << ny << "/" << nz
            << " ox/oy/oz=" << ox << "/" << oy << "/" << oz
            << " src0=" << static_cast<const void*>(p_src0)
            << " src1=" << static_cast<const void*>(p_src1)
            << " src2=" << static_cast<const void*>(p_src2)
            << " dst0=" << static_cast<void*>(p_dst0)
            << " dst1=" << static_cast<void*>(p_dst1)
            << " dst2=" << static_cast<void*>(p_dst2);
        throw std::runtime_error(oss.str());
    }
    cuda_check(cudaDeviceSynchronize(), "ifherk curl kernel sync");

    cuda_check(cudaMemcpy(dst0.data().data(), dst0.device_ptr(),
                   dst0.data().size() * sizeof(float_type),
                   cudaMemcpyDeviceToHost),
        "ifherk curl target x D2H");
    cuda_check(cudaMemcpy(dst1.data().data(), dst1.device_ptr(),
                   dst1.data().size() * sizeof(float_type),
                   cudaMemcpyDeviceToHost),
        "ifherk curl target y D2H");
    cuda_check(cudaMemcpy(dst2.data().data(), dst2.device_ptr(),
                   dst2.data().size() * sizeof(float_type),
                   cudaMemcpyDeviceToHost),
        "ifherk curl target z D2H");
}

template<class Block, class FaceSource, class EdgeSource, class Target>
inline void nonlinear_block(Block& block)
{
    auto& face0 = block(FaceSource::tag(), 0);
    auto& face1 = block(FaceSource::tag(), 1);
    auto& face2 = block(FaceSource::tag(), 2);
    auto& edge0 = block(EdgeSource::tag(), 0);
    auto& edge1 = block(EdgeSource::tag(), 1);
    auto& edge2 = block(EdgeSource::tag(), 2);
    auto& dst0 = block(Target::tag(), 0);
    auto& dst1 = block(Target::tag(), 1);
    auto& dst2 = block(Target::tag(), 2);

    face0.update_device(nullptr, true);
    face1.update_device(nullptr, true);
    face2.update_device(nullptr, true);
    edge0.update_device(nullptr, true);
    edge1.update_device(nullptr, true);
    edge2.update_device(nullptr, true);

    // Ensure device buffers exist for all fields used by the kernel.
    auto* p_face0 = face0.device_ptr();
    auto* p_face1 = face1.device_ptr();
    auto* p_face2 = face2.device_ptr();
    auto* p_edge0 = edge0.device_ptr();
    auto* p_edge1 = edge1.device_ptr();
    auto* p_edge2 = edge2.device_ptr();
    auto* p_dst0 = dst0.device_ptr();
    auto* p_dst1 = dst1.device_ptr();
    auto* p_dst2 = dst2.device_ptr();
    if (!p_face0 || !p_face1 || !p_face2 || !p_edge0 || !p_edge1 || !p_edge2 ||
        !p_dst0 || !p_dst1 || !p_dst2)
    {
        throw std::runtime_error("ifherk nonlinear: null device pointer");
    }

    const auto ext = face0.real_block().extent();
    const int ex = ext[0];
    const int ey = ext[1];
    const int ez = ext[2];

    const auto interior = block.descriptor().extent();
    const int nx = interior[0];
    const int ny = interior[1];
    const int nz = interior[2];

    // Some correction/non-leaf blocks can have empty interior extents.
    // CUDA launch with any zero grid dimension is invalid.
    if (nx <= 0 || ny <= 0 || nz <= 0) return;

    const auto lb = face0.lbuffer();
    const int ox = lb[0];
    const int oy = lb[1];
    const int oz = lb[2];

    const dim3 threads(8, 8, 8);
    const dim3 blocks((nx + threads.x - 1) / threads.x,
        (ny + threads.y - 1) / threads.y, (nz + threads.z - 1) / threads.z);

    // Clear any prior sticky CUDA error before launch.
    (void)cudaGetLastError();
    operators_gpu_cuda::nonlinear_kernel<<<blocks, threads>>>(p_face0, p_face1, p_face2, p_edge0,
        p_edge1, p_edge2, p_dst0, p_dst1, p_dst2, ex, ey, ez, ox, oy, oz, nx,
        ny, nz);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        std::ostringstream oss;
        oss << "ifherk nonlinear kernel launch: "
            << cudaGetErrorString(launch_err) << " ex/ey/ez=" << ex << "/"
            << ey << "/" << ez << " nx/ny/nz=" << nx << "/" << ny << "/"
            << nz << " ox/oy/oz=" << ox << "/" << oy << "/" << oz;
        throw std::runtime_error(oss.str());
    }
    cuda_check(cudaDeviceSynchronize(), "ifherk nonlinear kernel sync");

    cuda_check(cudaMemcpy(dst0.data().data(), dst0.device_ptr(),
                   dst0.data().size() * sizeof(float_type),
                   cudaMemcpyDeviceToHost),
        "ifherk nonlinear target x D2H");
    cuda_check(cudaMemcpy(dst1.data().data(), dst1.device_ptr(),
                   dst1.data().size() * sizeof(float_type),
                   cudaMemcpyDeviceToHost),
        "ifherk nonlinear target y D2H");
    cuda_check(cudaMemcpy(dst2.data().data(), dst2.device_ptr(),
                   dst2.data().size() * sizeof(float_type),
                   cudaMemcpyDeviceToHost),
        "ifherk nonlinear target z D2H");
}

template<class Block, class Source, class Target>
inline void divergence_block(Block& block, float_type dx_level)
{
    auto& src0 = block(Source::tag(), 0);
    auto& src1 = block(Source::tag(), 1);
    auto& src2 = block(Source::tag(), 2);
    auto& dst = block(Target::tag(), 0);

    src0.update_device(nullptr, true);
    src1.update_device(nullptr, true);
    src2.update_device(nullptr, true);

    auto* p_src0 = src0.device_ptr();
    auto* p_src1 = src1.device_ptr();
    auto* p_src2 = src2.device_ptr();
    auto* p_dst = dst.device_ptr();
    if (!p_src0 || !p_src1 || !p_src2 || !p_dst)
        throw std::runtime_error("ifherk divergence: null device pointer");

    const auto ext = src0.real_block().extent();
    const int ex = ext[0];
    const int ey = ext[1];
    const int ez = ext[2];

    const auto interior = block.descriptor().extent();
    const int nx = interior[0];
    const int ny = interior[1];
    const int nz = interior[2];
    if (nx <= 0 || ny <= 0 || nz <= 0) return;

    const auto lb = src0.lbuffer();
    const int ox = lb[0];
    const int oy = lb[1];
    const int oz = lb[2];

    const dim3 threads(8, 8, 8);
    const dim3 blocks((nx + threads.x - 1) / threads.x,
        (ny + threads.y - 1) / threads.y, (nz + threads.z - 1) / threads.z);

    (void)cudaGetLastError();
    operators_gpu_cuda::divergence_kernel<<<blocks, threads>>>(p_src0, p_src1, p_src2, p_dst, ex,
        ey, ez, ox, oy, oz, nx, ny, nz, 1.0 / dx_level);
    cuda_check(cudaGetLastError(), "ifherk divergence kernel launch");
    cuda_check(cudaDeviceSynchronize(), "ifherk divergence kernel sync");

    cuda_check(cudaMemcpy(dst.data().data(), p_dst,
                   dst.data().size() * sizeof(float_type),
                   cudaMemcpyDeviceToHost),
        "ifherk divergence target D2H");
}

template<class Block, class Source, class Target>
inline void gradient_block(Block& block, float_type dx_level)
{
    auto& src = block(Source::tag(), 0);
    auto& dst0 = block(Target::tag(), 0);
    auto& dst1 = block(Target::tag(), 1);
    auto& dst2 = block(Target::tag(), 2);

    src.update_device(nullptr, true);

    auto* p_src = src.device_ptr();
    auto* p_dst0 = dst0.device_ptr();
    auto* p_dst1 = dst1.device_ptr();
    auto* p_dst2 = dst2.device_ptr();
    if (!p_src || !p_dst0 || !p_dst1 || !p_dst2)
        throw std::runtime_error("ifherk gradient: null device pointer");

    const auto ext = src.real_block().extent();
    const int ex = ext[0];
    const int ey = ext[1];
    const int ez = ext[2];

    const auto interior = block.descriptor().extent();
    const int nx = interior[0];
    const int ny = interior[1];
    const int nz = interior[2];
    if (nx <= 0 || ny <= 0 || nz <= 0) return;

    const auto lb = src.lbuffer();
    const int ox = lb[0];
    const int oy = lb[1];
    const int oz = lb[2];

    const dim3 threads(8, 8, 8);
    const dim3 blocks((nx + threads.x - 1) / threads.x,
        (ny + threads.y - 1) / threads.y, (nz + threads.z - 1) / threads.z);

    (void)cudaGetLastError();
    operators_gpu_cuda::gradient_kernel<<<blocks, threads>>>(p_src, p_dst0, p_dst1, p_dst2, ex, ey,
        ez, ox, oy, oz, nx, ny, nz, 1.0 / dx_level);
    cuda_check(cudaGetLastError(), "ifherk gradient kernel launch");
    cuda_check(cudaDeviceSynchronize(), "ifherk gradient kernel sync");

    cuda_check(cudaMemcpy(dst0.data().data(), p_dst0,
                   dst0.data().size() * sizeof(float_type),
                   cudaMemcpyDeviceToHost),
        "ifherk gradient target x D2H");
    cuda_check(cudaMemcpy(dst1.data().data(), p_dst1,
                   dst1.data().size() * sizeof(float_type),
                   cudaMemcpyDeviceToHost),
        "ifherk gradient target y D2H");
    cuda_check(cudaMemcpy(dst2.data().data(), p_dst2,
                   dst2.data().size() * sizeof(float_type),
                   cudaMemcpyDeviceToHost),
        "ifherk gradient target z D2H");
}

template<class BlockPtr, class Source, class Target>
inline void curl_level_batched(const std::vector<BlockPtr>& blocks,
    float_type dx_level)
{
    if (blocks.empty()) return;
    std::vector<BlockPtr> work_blocks;
    work_blocks.reserve(blocks.size());
    for (auto* b : blocks)
    {
        if (!b) continue;
        if ((*b)(Source::tag(), 0).data().empty()) continue;
        work_blocks.push_back(b);
    }
    if (work_blocks.empty()) return;

    const auto ext = (*work_blocks.front())(Source::tag(), 0).real_block().extent();
    const int ex = ext[0];
    const int ey = ext[1];
    const auto interior = work_blocks.front()->descriptor().extent();
    const int nx = interior[0], ny = interior[1], nz = interior[2];
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    const auto lb = (*work_blocks.front())(Source::tag(), 0).lbuffer();
    const int ox = lb[0], oy = lb[1], oz = lb[2];
    const size_t n_src_per_block =
        (*work_blocks.front())(Source::tag(), 0).data().size();
    const size_t n_dst_per_block =
        (*work_blocks.front())(Target::tag(), 0).data().size();
    if (n_src_per_block == 0 || n_dst_per_block == 0) return;

    for (auto* b : work_blocks)
    {
        const auto ext_b = (*b)(Source::tag(), 0).real_block().extent();
        const auto int_b = b->descriptor().extent();
        const auto lb_b = (*b)(Source::tag(), 0).lbuffer();
        const size_t n_src_b = (*b)(Source::tag(), 0).data().size();
        const size_t n_dst_b = (*b)(Target::tag(), 0).data().size();
        if (ext_b[0] != ex || ext_b[1] != ey || int_b[0] != nx || int_b[1] != ny ||
            int_b[2] != nz || lb_b[0] != ox || lb_b[1] != oy || lb_b[2] != oz ||
            n_src_b != n_src_per_block || n_dst_b != n_dst_per_block)
        {
            for (auto* bb : work_blocks)
            {
                auto& rb = *bb;
                curl_block<std::remove_reference_t<decltype(rb)>, Source, Target>(
                    rb, dx_level);
            }
            return;
        }
    }

    std::vector<float_type*> h_src0, h_src1, h_src2;
    std::vector<float_type*> h_dst0, h_dst1, h_dst2;
    h_src0.reserve(work_blocks.size());
    h_src1.reserve(work_blocks.size());
    h_src2.reserve(work_blocks.size());
    h_dst0.reserve(work_blocks.size());
    h_dst1.reserve(work_blocks.size());
    h_dst2.reserve(work_blocks.size());

    const size_t dst_slab_items = n_dst_per_block * work_blocks.size();
    if (dst_slab_items == 0) return;

    auto& d2h_pack = d2h_pack_buf();
    auto& d_d2h_pack = d_d2h_pack_ptr();
    auto& d2h_cap = d2h_pack_capacity();
    ensure_pack_capacity(d2h_pack, d_d2h_pack, d2h_cap, 3 * dst_slab_items);

    for (size_t b = 0; b < work_blocks.size(); ++b)
    {
        auto& s0 = (*work_blocks[b])(Source::tag(), 0);
        auto& s1 = (*work_blocks[b])(Source::tag(), 1);
        auto& s2 = (*work_blocks[b])(Source::tag(), 2);
        auto& d0 = (*work_blocks[b])(Target::tag(), 0);
        auto& d1 = (*work_blocks[b])(Target::tag(), 1);
        auto& d2 = (*work_blocks[b])(Target::tag(), 2);
        s0.update_device(nullptr, true);
        s1.update_device(nullptr, true);
        s2.update_device(nullptr, true);

        auto* p_s0 = s0.device_ptr();
        auto* p_s1 = s1.device_ptr();
        auto* p_s2 = s2.device_ptr();
        auto* p_d0 = d0.device_ptr();
        auto* p_d1 = d1.device_ptr();
        auto* p_d2 = d2.device_ptr();
        if (!p_s0 || !p_s1 || !p_s2 || !p_d0 || !p_d1 || !p_d2)
            throw std::runtime_error(
                "IFHERK batched curl: null device pointer");

        h_src0.push_back(const_cast<float_type*>(p_s0));
        h_src1.push_back(const_cast<float_type*>(p_s1));
        h_src2.push_back(const_cast<float_type*>(p_s2));
        h_dst0.push_back(p_d0);
        h_dst1.push_back(p_d1);
        h_dst2.push_back(p_d2);
    }

    float_type** d_src0 =
        upload_pointer_table(h_src0, "Upload IFHERK curl src0 pointer table");
    float_type** d_src1 =
        upload_pointer_table(h_src1, "Upload IFHERK curl src1 pointer table");
    float_type** d_src2 =
        upload_pointer_table(h_src2, "Upload IFHERK curl src2 pointer table");
    float_type** d_dst0 =
        upload_pointer_table(h_dst0, "Upload IFHERK curl dst0 pointer table");
    float_type** d_dst1 =
        upload_pointer_table(h_dst1, "Upload IFHERK curl dst1 pointer table");
    float_type** d_dst2 =
        upload_pointer_table(h_dst2, "Upload IFHERK curl dst2 pointer table");

    const int threads = 256;
    const int grid_gather =
        static_cast<int>((dst_slab_items + threads - 1) / threads);
    if (grid_gather <= 0) return;

    const size_t interior_size = static_cast<size_t>(nx) * ny * nz;
    const size_t total_threads = interior_size * work_blocks.size();
    const int grid = static_cast<int>((total_threads + threads - 1) / threads);
    if (grid <= 0) return;
    operators_gpu_cuda::curl_kernel_batched<<<grid, threads>>>(
        reinterpret_cast<const float_type* const*>(d_src0),
        reinterpret_cast<const float_type* const*>(d_src1),
        reinterpret_cast<const float_type* const*>(d_src2),
        reinterpret_cast<float_type* const*>(d_dst0),
        reinterpret_cast<float_type* const*>(d_dst1),
        reinterpret_cast<float_type* const*>(d_dst2),
        static_cast<int>(work_blocks.size()), ex, ey, ox, oy, oz, nx, ny,
        nz, 1.0 / dx_level);
    cuda_check(cudaGetLastError(), "IFHERK batched curl kernel");

    operators_gpu_cuda::gather3_to_slab<<<grid_gather, threads>>>(
        reinterpret_cast<const float_type* const*>(d_dst0),
        reinterpret_cast<const float_type* const*>(d_dst1),
        reinterpret_cast<const float_type* const*>(d_dst2), d_d2h_pack,
        static_cast<int>(work_blocks.size()), n_dst_per_block);
    cuda_check(cudaGetLastError(), "IFHERK curl slab gather");
    cuda_check(cudaMemcpy(d2h_pack.data(), d_d2h_pack,
                   3 * dst_slab_items * sizeof(float_type),
                   cudaMemcpyDeviceToHost),
        "IFHERK curl slab D2H");
    cuda_check(cudaDeviceSynchronize(), "IFHERK batched curl sync");

    for (size_t b = 0; b < work_blocks.size(); ++b)
    {
        auto& d0 = (*work_blocks[b])(Target::tag(), 0);
        auto& d1 = (*work_blocks[b])(Target::tag(), 1);
        auto& d2 = (*work_blocks[b])(Target::tag(), 2);
        std::copy(d2h_pack.begin() + 0 * dst_slab_items + b * n_dst_per_block,
            d2h_pack.begin() + 0 * dst_slab_items +
                (b + 1) * n_dst_per_block,
            d0.data().begin());
        std::copy(d2h_pack.begin() + 1 * dst_slab_items + b * n_dst_per_block,
            d2h_pack.begin() + 1 * dst_slab_items +
                (b + 1) * n_dst_per_block,
            d1.data().begin());
        std::copy(d2h_pack.begin() + 2 * dst_slab_items + b * n_dst_per_block,
            d2h_pack.begin() + 2 * dst_slab_items +
                (b + 1) * n_dst_per_block,
            d2.data().begin());
    }

}

template<class BlockPtr, class FaceSource, class EdgeSource, class Target>
inline void nonlinear_level_batched(const std::vector<BlockPtr>& blocks,
    float_type scale)
{
    if (blocks.empty()) return;
    std::vector<BlockPtr> work_blocks;
    work_blocks.reserve(blocks.size());
    for (auto* b : blocks)
    {
        if (!b) continue;
        if ((*b)(FaceSource::tag(), 0).data().empty()) continue;
        work_blocks.push_back(b);
    }
    if (work_blocks.empty()) return;

    const auto ext =
        (*work_blocks.front())(FaceSource::tag(), 0).real_block().extent();
    const int ex = ext[0];
    const int ey = ext[1];
    const auto interior = work_blocks.front()->descriptor().extent();
    const int nx = interior[0], ny = interior[1], nz = interior[2];
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    const auto lb = (*work_blocks.front())(FaceSource::tag(), 0).lbuffer();
    const int ox = lb[0], oy = lb[1], oz = lb[2];
    const size_t n_per_block =
        (*work_blocks.front())(FaceSource::tag(), 0).data().size();
    if (n_per_block == 0) return;

    for (auto* b : work_blocks)
    {
        const auto ext_b = (*b)(FaceSource::tag(), 0).real_block().extent();
        const auto int_b = b->descriptor().extent();
        const auto lb_b = (*b)(FaceSource::tag(), 0).lbuffer();
        const size_t n_b = (*b)(FaceSource::tag(), 0).data().size();
        if (ext_b[0] != ex || ext_b[1] != ey || int_b[0] != nx || int_b[1] != ny ||
            int_b[2] != nz || lb_b[0] != ox || lb_b[1] != oy || lb_b[2] != oz ||
            n_b != n_per_block)
        {
            for (auto* bb : work_blocks)
            {
                auto& rb = *bb;
                nonlinear_block<std::remove_reference_t<decltype(rb)>,
                    FaceSource, EdgeSource, Target>(rb);
                for (std::size_t field_idx = 0; field_idx < Target::nFields();
                     ++field_idx)
                {
                    rb(Target::tag(), field_idx).linalg_data() *= scale;
                }
            }
            return;
        }
    }

    std::vector<float_type*> h_face0, h_face1, h_face2;
    std::vector<float_type*> h_edge0, h_edge1, h_edge2;
    std::vector<float_type*> h_dst0, h_dst1, h_dst2;
    h_face0.reserve(work_blocks.size());
    h_face1.reserve(work_blocks.size());
    h_face2.reserve(work_blocks.size());
    h_edge0.reserve(work_blocks.size());
    h_edge1.reserve(work_blocks.size());
    h_edge2.reserve(work_blocks.size());
    h_dst0.reserve(work_blocks.size());
    h_dst1.reserve(work_blocks.size());
    h_dst2.reserve(work_blocks.size());

    const size_t slab_items = n_per_block * work_blocks.size();
    if (slab_items == 0) return;

    auto& d2h_pack = d2h_pack_buf();
    auto& d_d2h_pack = d_d2h_pack_ptr();
    auto& d2h_cap = d2h_pack_capacity();
    ensure_pack_capacity(d2h_pack, d_d2h_pack, d2h_cap, 3 * slab_items);

    for (size_t b = 0; b < work_blocks.size(); ++b)
    {
        auto& f0 = (*work_blocks[b])(FaceSource::tag(), 0);
        auto& f1 = (*work_blocks[b])(FaceSource::tag(), 1);
        auto& f2 = (*work_blocks[b])(FaceSource::tag(), 2);
        auto& e0 = (*work_blocks[b])(EdgeSource::tag(), 0);
        auto& e1 = (*work_blocks[b])(EdgeSource::tag(), 1);
        auto& e2 = (*work_blocks[b])(EdgeSource::tag(), 2);
        auto& d0 = (*work_blocks[b])(Target::tag(), 0);
        auto& d1 = (*work_blocks[b])(Target::tag(), 1);
        auto& d2 = (*work_blocks[b])(Target::tag(), 2);
        f0.update_device(nullptr, true);
        f1.update_device(nullptr, true);
        f2.update_device(nullptr, true);
        e0.update_device(nullptr, true);
        e1.update_device(nullptr, true);
        e2.update_device(nullptr, true);
        auto* p_f0 = f0.device_ptr();
        auto* p_f1 = f1.device_ptr();
        auto* p_f2 = f2.device_ptr();
        auto* p_e0 = e0.device_ptr();
        auto* p_e1 = e1.device_ptr();
        auto* p_e2 = e2.device_ptr();
        auto* p_d0 = d0.device_ptr();
        auto* p_d1 = d1.device_ptr();
        auto* p_d2 = d2.device_ptr();
        if (!p_f0 || !p_f1 || !p_f2 || !p_e0 || !p_e1 || !p_e2 || !p_d0 ||
            !p_d1 || !p_d2)
            throw std::runtime_error(
                "IFHERK batched nonlinear: null device pointer");

        h_face0.push_back(const_cast<float_type*>(p_f0));
        h_face1.push_back(const_cast<float_type*>(p_f1));
        h_face2.push_back(const_cast<float_type*>(p_f2));
        h_edge0.push_back(const_cast<float_type*>(p_e0));
        h_edge1.push_back(const_cast<float_type*>(p_e1));
        h_edge2.push_back(const_cast<float_type*>(p_e2));
        h_dst0.push_back(p_d0);
        h_dst1.push_back(p_d1);
        h_dst2.push_back(p_d2);
    }

    float_type** d_face0 = upload_pointer_table(
        h_face0, "Upload IFHERK nonlinear face0 pointer table");
    float_type** d_face1 = upload_pointer_table(
        h_face1, "Upload IFHERK nonlinear face1 pointer table");
    float_type** d_face2 = upload_pointer_table(
        h_face2, "Upload IFHERK nonlinear face2 pointer table");
    float_type** d_edge0 = upload_pointer_table(
        h_edge0, "Upload IFHERK nonlinear edge0 pointer table");
    float_type** d_edge1 = upload_pointer_table(
        h_edge1, "Upload IFHERK nonlinear edge1 pointer table");
    float_type** d_edge2 = upload_pointer_table(
        h_edge2, "Upload IFHERK nonlinear edge2 pointer table");
    float_type** d_dst0 = upload_pointer_table(
        h_dst0, "Upload IFHERK nonlinear dst0 pointer table");
    float_type** d_dst1 = upload_pointer_table(
        h_dst1, "Upload IFHERK nonlinear dst1 pointer table");
    float_type** d_dst2 = upload_pointer_table(
        h_dst2, "Upload IFHERK nonlinear dst2 pointer table");

    const int threads = 256;
    const int grid_slab = static_cast<int>((slab_items + threads - 1) / threads);
    if (grid_slab <= 0) return;

    const size_t interior_size = static_cast<size_t>(nx) * ny * nz;
    const size_t total_threads = interior_size * work_blocks.size();
    const int grid = static_cast<int>((total_threads + threads - 1) / threads);
    if (grid <= 0) return;
    operators_gpu_cuda::nonlinear_kernel_batched<<<grid, threads>>>(
        reinterpret_cast<const float_type* const*>(d_face0),
        reinterpret_cast<const float_type* const*>(d_face1),
        reinterpret_cast<const float_type* const*>(d_face2),
        reinterpret_cast<const float_type* const*>(d_edge0),
        reinterpret_cast<const float_type* const*>(d_edge1),
        reinterpret_cast<const float_type* const*>(d_edge2),
        reinterpret_cast<float_type* const*>(d_dst0),
        reinterpret_cast<float_type* const*>(d_dst1),
        reinterpret_cast<float_type* const*>(d_dst2),
        static_cast<int>(work_blocks.size()), ex, ey, ox, oy, oz, nx, ny, nz);
    cuda_check(cudaGetLastError(), "IFHERK batched nonlinear kernel");

    operators_gpu_cuda::gather3_to_slab<<<grid_slab, threads>>>(
        reinterpret_cast<const float_type* const*>(d_dst0),
        reinterpret_cast<const float_type* const*>(d_dst1),
        reinterpret_cast<const float_type* const*>(d_dst2), d_d2h_pack,
        static_cast<int>(work_blocks.size()), n_per_block);
    cuda_check(cudaGetLastError(), "IFHERK nonlinear slab gather");
    cuda_check(cudaMemcpy(d2h_pack.data(), d_d2h_pack,
                   3 * slab_items * sizeof(float_type), cudaMemcpyDeviceToHost),
        "IFHERK nonlinear slab D2H");
    cuda_check(cudaDeviceSynchronize(), "IFHERK batched nonlinear sync");

    for (size_t b = 0; b < work_blocks.size(); ++b)
    {
        auto& d0 = (*work_blocks[b])(Target::tag(), 0);
        auto& d1 = (*work_blocks[b])(Target::tag(), 1);
        auto& d2 = (*work_blocks[b])(Target::tag(), 2);
        std::copy(d2h_pack.begin() + 0 * slab_items + b * n_per_block,
            d2h_pack.begin() + 0 * slab_items + (b + 1) * n_per_block,
            d0.data().begin());
        std::copy(d2h_pack.begin() + 1 * slab_items + b * n_per_block,
            d2h_pack.begin() + 1 * slab_items + (b + 1) * n_per_block,
            d1.data().begin());
        std::copy(d2h_pack.begin() + 2 * slab_items + b * n_per_block,
            d2h_pack.begin() + 2 * slab_items + (b + 1) * n_per_block,
            d2.data().begin());
        for (std::size_t field_idx = 0; field_idx < Target::nFields();
             ++field_idx)
        {
            (*work_blocks[b])(Target::tag(), field_idx).linalg_data() *= scale;
        }
    }

}

template<class BlockPtr, class Source, class Target>
inline void divergence_level_batched(const std::vector<BlockPtr>& blocks,
    float_type dx_level)
{
    if (blocks.empty()) return;
    std::vector<BlockPtr> work_blocks;
    work_blocks.reserve(blocks.size());
    for (auto* b : blocks)
    {
        if (!b) continue;
        if ((*b)(Source::tag(), 0).data().empty()) continue;
        work_blocks.push_back(b);
    }
    if (work_blocks.empty()) return;

    const auto ext = (*work_blocks.front())(Source::tag(), 0).real_block().extent();
    const int ex = ext[0], ey = ext[1];
    const auto interior = work_blocks.front()->descriptor().extent();
    const int nx = interior[0], ny = interior[1], nz = interior[2];
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    const auto lb = (*work_blocks.front())(Source::tag(), 0).lbuffer();
    const int ox = lb[0], oy = lb[1], oz = lb[2];
    const size_t n_src = (*work_blocks.front())(Source::tag(), 0).data().size();
    const size_t n_dst = (*work_blocks.front())(Target::tag(), 0).data().size();
    if (n_src == 0 || n_dst == 0) return;

    for (auto* b : work_blocks)
    {
        const auto ext_b = (*b)(Source::tag(), 0).real_block().extent();
        const auto int_b = b->descriptor().extent();
        const auto lb_b = (*b)(Source::tag(), 0).lbuffer();
        if (ext_b[0] != ex || ext_b[1] != ey || int_b[0] != nx || int_b[1] != ny ||
            int_b[2] != nz || lb_b[0] != ox || lb_b[1] != oy || lb_b[2] != oz ||
            (*b)(Source::tag(), 0).data().size() != n_src ||
            (*b)(Target::tag(), 0).data().size() != n_dst)
        {
            for (auto* bb : work_blocks)
            {
                auto& rb = *bb;
                divergence_block<std::remove_reference_t<decltype(rb)>, Source,
                    Target>(rb, dx_level);
            }
            return;
        }
    }

    std::vector<float_type*> h_s0, h_s1, h_s2, h_d;
    h_s0.reserve(work_blocks.size());
    h_s1.reserve(work_blocks.size());
    h_s2.reserve(work_blocks.size());
    h_d.reserve(work_blocks.size());

    auto& d2h_pack = d2h_pack_buf();
    auto& d_d2h_pack = d_d2h_pack_ptr();
    auto& d2h_cap = d2h_pack_capacity();
    const size_t dst_slab_items = n_dst * work_blocks.size();
    ensure_pack_capacity(d2h_pack, d_d2h_pack, d2h_cap, dst_slab_items);

    for (auto* b : work_blocks)
    {
        auto& s0 = (*b)(Source::tag(), 0);
        auto& s1 = (*b)(Source::tag(), 1);
        auto& s2 = (*b)(Source::tag(), 2);
        auto& d = (*b)(Target::tag(), 0);
        s0.update_device(nullptr, true);
        s1.update_device(nullptr, true);
        s2.update_device(nullptr, true);
        h_s0.push_back(const_cast<float_type*>(s0.device_ptr()));
        h_s1.push_back(const_cast<float_type*>(s1.device_ptr()));
        h_s2.push_back(const_cast<float_type*>(s2.device_ptr()));
        h_d.push_back(d.device_ptr());
    }

    float_type** d_s0 = upload_pointer_table(h_s0, "Upload IFHERK div src0");
    float_type** d_s1 = upload_pointer_table(h_s1, "Upload IFHERK div src1");
    float_type** d_s2 = upload_pointer_table(h_s2, "Upload IFHERK div src2");
    float_type** d_d = upload_pointer_table(h_d, "Upload IFHERK div dst");

    const int threads = 256;
    const size_t interior_size = static_cast<size_t>(nx) * ny * nz;
    const size_t total_threads = interior_size * work_blocks.size();
    const int grid = static_cast<int>((total_threads + threads - 1) / threads);
    operators_gpu_cuda::divergence_kernel_batched<<<grid, threads>>>(
        reinterpret_cast<const float_type* const*>(d_s0),
        reinterpret_cast<const float_type* const*>(d_s1),
        reinterpret_cast<const float_type* const*>(d_s2),
        reinterpret_cast<float_type* const*>(d_d),
        static_cast<int>(work_blocks.size()), ex, ey, ox, oy, oz, nx, ny, nz,
        1.0 / dx_level);
    cuda_check(cudaGetLastError(), "IFHERK batched divergence kernel");

    const int grid_g = static_cast<int>((dst_slab_items + threads - 1) / threads);
    operators_gpu_cuda::gather1_to_slab<<<grid_g, threads>>>(
        reinterpret_cast<const float_type* const*>(d_d), d_d2h_pack,
        static_cast<int>(work_blocks.size()), n_dst);
    cuda_check(cudaGetLastError(), "IFHERK divergence slab gather");
    cuda_check(cudaMemcpy(d2h_pack.data(), d_d2h_pack,
                   dst_slab_items * sizeof(float_type), cudaMemcpyDeviceToHost),
        "IFHERK divergence slab D2H");
    cuda_check(cudaDeviceSynchronize(), "IFHERK batched divergence sync");

    for (size_t b = 0; b < work_blocks.size(); ++b)
    {
        auto& d = (*work_blocks[b])(Target::tag(), 0);
        std::copy(d2h_pack.begin() + b * n_dst,
            d2h_pack.begin() + (b + 1) * n_dst, d.data().begin());
    }

}

template<class BlockPtr, class Source, class Target>
inline void gradient_level_batched(const std::vector<BlockPtr>& blocks,
    float_type dx_level, float_type scale)
{
    if (blocks.empty()) return;
    std::vector<BlockPtr> work_blocks;
    work_blocks.reserve(blocks.size());
    for (auto* b : blocks)
    {
        if (!b) continue;
        if ((*b)(Source::tag(), 0).data().empty()) continue;
        work_blocks.push_back(b);
    }
    if (work_blocks.empty()) return;

    const auto ext = (*work_blocks.front())(Source::tag(), 0).real_block().extent();
    const int ex = ext[0], ey = ext[1];
    const auto interior = work_blocks.front()->descriptor().extent();
    const int nx = interior[0], ny = interior[1], nz = interior[2];
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    const auto lb = (*work_blocks.front())(Source::tag(), 0).lbuffer();
    const int ox = lb[0], oy = lb[1], oz = lb[2];
    const size_t n_src = (*work_blocks.front())(Source::tag(), 0).data().size();
    const size_t n_dst = (*work_blocks.front())(Target::tag(), 0).data().size();
    if (n_src == 0 || n_dst == 0) return;

    for (auto* b : work_blocks)
    {
        const auto ext_b = (*b)(Source::tag(), 0).real_block().extent();
        const auto int_b = b->descriptor().extent();
        const auto lb_b = (*b)(Source::tag(), 0).lbuffer();
        if (ext_b[0] != ex || ext_b[1] != ey || int_b[0] != nx || int_b[1] != ny ||
            int_b[2] != nz || lb_b[0] != ox || lb_b[1] != oy || lb_b[2] != oz ||
            (*b)(Source::tag(), 0).data().size() != n_src ||
            (*b)(Target::tag(), 0).data().size() != n_dst)
        {
            for (auto* bb : work_blocks)
            {
                auto& rb = *bb;
                gradient_block<std::remove_reference_t<decltype(rb)>, Source,
                    Target>(rb, dx_level);
                for (std::size_t field_idx = 0; field_idx < Target::nFields();
                     ++field_idx)
                    rb(Target::tag(), field_idx).linalg_data() *= scale;
            }
            return;
        }
    }

    std::vector<float_type*> h_s, h_d0, h_d1, h_d2;
    h_s.reserve(work_blocks.size());
    h_d0.reserve(work_blocks.size());
    h_d1.reserve(work_blocks.size());
    h_d2.reserve(work_blocks.size());

    auto& d2h_pack = d2h_pack_buf();
    auto& d_d2h_pack = d_d2h_pack_ptr();
    auto& d2h_cap = d2h_pack_capacity();
    const size_t dst_slab_items = n_dst * work_blocks.size();
    ensure_pack_capacity(d2h_pack, d_d2h_pack, d2h_cap, 3 * dst_slab_items);

    for (auto* b : work_blocks)
    {
        auto& s = (*b)(Source::tag(), 0);
        auto& d0 = (*b)(Target::tag(), 0);
        auto& d1 = (*b)(Target::tag(), 1);
        auto& d2 = (*b)(Target::tag(), 2);
        s.update_device(nullptr, true);
        h_s.push_back(const_cast<float_type*>(s.device_ptr()));
        h_d0.push_back(d0.device_ptr());
        h_d1.push_back(d1.device_ptr());
        h_d2.push_back(d2.device_ptr());
    }

    float_type** d_s = upload_pointer_table(h_s, "Upload IFHERK grad src");
    float_type** d_d0 = upload_pointer_table(h_d0, "Upload IFHERK grad dst0");
    float_type** d_d1 = upload_pointer_table(h_d1, "Upload IFHERK grad dst1");
    float_type** d_d2 = upload_pointer_table(h_d2, "Upload IFHERK grad dst2");

    const int threads = 256;
    const size_t interior_size = static_cast<size_t>(nx) * ny * nz;
    const size_t total_threads = interior_size * work_blocks.size();
    const int grid = static_cast<int>((total_threads + threads - 1) / threads);
    operators_gpu_cuda::gradient_kernel_batched<<<grid, threads>>>(
        reinterpret_cast<const float_type* const*>(d_s),
        reinterpret_cast<float_type* const*>(d_d0),
        reinterpret_cast<float_type* const*>(d_d1),
        reinterpret_cast<float_type* const*>(d_d2),
        static_cast<int>(work_blocks.size()), ex, ey, ox, oy, oz, nx, ny, nz,
        1.0 / dx_level);
    cuda_check(cudaGetLastError(), "IFHERK batched gradient kernel");

    const int grid_g = static_cast<int>((dst_slab_items + threads - 1) / threads);
    operators_gpu_cuda::gather3_to_slab<<<grid_g, threads>>>(
        reinterpret_cast<const float_type* const*>(d_d0),
        reinterpret_cast<const float_type* const*>(d_d1),
        reinterpret_cast<const float_type* const*>(d_d2), d_d2h_pack,
        static_cast<int>(work_blocks.size()), n_dst);
    cuda_check(cudaGetLastError(), "IFHERK gradient slab gather");
    cuda_check(cudaMemcpy(d2h_pack.data(), d_d2h_pack,
                   3 * dst_slab_items * sizeof(float_type),
                   cudaMemcpyDeviceToHost),
        "IFHERK gradient slab D2H");
    cuda_check(cudaDeviceSynchronize(), "IFHERK batched gradient sync");

    for (size_t b = 0; b < work_blocks.size(); ++b)
    {
        auto& d0 = (*work_blocks[b])(Target::tag(), 0);
        auto& d1 = (*work_blocks[b])(Target::tag(), 1);
        auto& d2 = (*work_blocks[b])(Target::tag(), 2);
        std::copy(d2h_pack.begin() + 0 * dst_slab_items + b * n_dst,
            d2h_pack.begin() + 0 * dst_slab_items + (b + 1) * n_dst,
            d0.data().begin());
        std::copy(d2h_pack.begin() + 1 * dst_slab_items + b * n_dst,
            d2h_pack.begin() + 1 * dst_slab_items + (b + 1) * n_dst,
            d1.data().begin());
        std::copy(d2h_pack.begin() + 2 * dst_slab_items + b * n_dst,
            d2h_pack.begin() + 2 * dst_slab_items + (b + 1) * n_dst,
            d2.data().begin());
        for (std::size_t field_idx = 0; field_idx < Target::nFields();
             ++field_idx)
            (*work_blocks[b])(Target::tag(), field_idx).linalg_data() *= scale;
    }

}

} // namespace ifherk_gpu
} // namespace solver
} // namespace iblgf

#endif // IBLGF_COMPILE_CUDA

#endif // IBLGF_INCLUDED_IFHERK_GPU_HPP
