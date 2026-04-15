#ifndef IBLGF_INCLUDED_OPERATORS_GPU_KERNELS_HPP
#define IBLGF_INCLUDED_OPERATORS_GPU_KERNELS_HPP

#ifdef IBLGF_COMPILE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <iblgf/global.hpp>

namespace iblgf
{
namespace operators_gpu_cuda
{
inline void cuda_check(cudaError_t err, const char* context)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(context) + ": " + cudaGetErrorString(err));
    }
}

__global__ void laplace_kernel(const float_type* src, float_type* dst,
    int ex, int ey, int ez, int ox, int oy, int oz, int nx, int ny, int nz,
    float_type fac)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    const int plane = ex * ey;
    const int idx = (i + ox) + ex * ((j + oy) + ey * (k + oz));

    dst[idx] = fac * (-6.0 * src[idx] + src[idx - 1] + src[idx + 1] +
                      src[idx - ex] + src[idx + ex] + src[idx - plane] +
                      src[idx + plane]);
}

__global__ void gradient_kernel(const float_type* src, float_type* dst0,
    float_type* dst1, float_type* dst2, int ex, int ey, int ez, int ox, int oy,
    int oz, int nx, int ny, int nz, float_type fac)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    const int plane = ex * ey;
    const int idx = (i + ox) + ex * ((j + oy) + ey * (k + oz));
    dst0[idx] = fac * (src[idx] - src[idx - 1]);
    dst1[idx] = fac * (src[idx] - src[idx - ex]);
    dst2[idx] = fac * (src[idx] - src[idx - plane]);
}

__global__ void divergence_kernel(const float_type* src0,
    const float_type* src1, const float_type* src2, float_type* dst, int ex,
    int ey, int ez, int ox, int oy, int oz, int nx, int ny, int nz,
    float_type fac)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    const int plane = ex * ey;
    const int idx = (i + ox) + ex * ((j + oy) + ey * (k + oz));

    dst[idx] = fac * (-src0[idx] - src1[idx] - src2[idx] + src0[idx + 1] +
                      src1[idx + ex] + src2[idx + plane]);
}

__global__ void curl_kernel(const float_type* src0,
    const float_type* src1, const float_type* src2, float_type* dst0,
    float_type* dst1, float_type* dst2, int ex, int ey, int ez, int ox, int oy,
    int oz, int nx, int ny, int nz, float_type fac)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    const int plane = ex * ey;
    const int idx = (i + ox) + ex * ((j + oy) + ey * (k + oz));

    dst0[idx] =
        fac * ((src2[idx] - src2[idx - ex]) - (src1[idx] - src1[idx - plane]));
    dst1[idx] =
        fac * ((src0[idx] - src0[idx - plane]) - (src2[idx] - src2[idx - 1]));
    dst2[idx] =
        fac * ((src1[idx] - src1[idx - 1]) - (src0[idx] - src0[idx - ex]));
}

__global__ void nonlinear_kernel(const float_type* face0, const float_type* face1,
    const float_type* face2, const float_type* edge0, const float_type* edge1,
    const float_type* edge2, float_type* dst0, float_type* dst1,
    float_type* dst2, int ex, int ey, int ez, int ox, int oy, int oz, int nx,
    int ny, int nz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    (void)ez;
    const int plane = ex * ey;
    const int idx = (i + ox) + ex * ((j + oy) + ey * (k + oz));

    dst0[idx] = 0.25 * (+edge1[idx] * (+face2[idx] + face2[idx - 1]) +
                        edge1[idx + plane] *
                            (+face2[idx + plane] + face2[idx + plane - 1]) -
                        edge2[idx] * (+face1[idx] + face1[idx - 1]) -
                        edge2[idx + ex] * (+face1[idx + ex] + face1[idx + ex - 1]));

    dst1[idx] = 0.25 * (+edge2[idx] * (+face0[idx] + face0[idx - ex]) +
                        edge2[idx + 1] * (+face0[idx + 1] + face0[idx + 1 - ex]) -
                        edge0[idx] * (+face2[idx] + face2[idx - ex]) -
                        edge0[idx + plane] *
                            (+face2[idx + plane] + face2[idx + plane - ex]));

    dst2[idx] = 0.25 * (+edge0[idx] * (+face1[idx] + face1[idx - plane]) +
                        edge0[idx + ex] *
                            (+face1[idx + ex] + face1[idx + ex - plane]) -
                        edge1[idx] * (+face0[idx] + face0[idx - plane]) -
                        edge1[idx + 1] *
                            (+face0[idx + 1] + face0[idx + 1 - plane]));
}

__global__ void unpack8_kernel(const float_type* packed, float_type* f0,
    float_type* f1, float_type* f2, float_type* f3, float_type* f4,
    float_type* f5, float_type* f6, float_type* f7, size_t n)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;
    f0[idx] = packed[idx];
    f1[idx] = packed[n + idx];
    f2[idx] = packed[2 * n + idx];
    f3[idx] = packed[3 * n + idx];
    f4[idx] = packed[4 * n + idx];
    f5[idx] = packed[5 * n + idx];
    f6[idx] = packed[6 * n + idx];
    f7[idx] = packed[7 * n + idx];
}

__global__ void unpack6_kernel(const float_type* packed, float_type* f0,
    float_type* f1, float_type* f2, float_type* f3, float_type* f4,
    float_type* f5, size_t n)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;
    f0[idx] = packed[idx];
    f1[idx] = packed[n + idx];
    f2[idx] = packed[2 * n + idx];
    f3[idx] = packed[3 * n + idx];
    f4[idx] = packed[4 * n + idx];
    f5[idx] = packed[5 * n + idx];
}

__global__ void pack3_kernel(const float_type* f0, const float_type* f1,
    const float_type* f2, float_type* packed, size_t n)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;
    packed[idx] = f0[idx];
    packed[n + idx] = f1[idx];
    packed[2 * n + idx] = f2[idx];
}

__global__ void pack8_kernel(const float_type* f0, const float_type* f1,
    const float_type* f2, const float_type* f3, const float_type* f4,
    const float_type* f5, const float_type* f6, const float_type* f7,
    float_type* packed, size_t n)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;
    packed[idx] = f0[idx];
    packed[n + idx] = f1[idx];
    packed[2 * n + idx] = f2[idx];
    packed[3 * n + idx] = f3[idx];
    packed[4 * n + idx] = f4[idx];
    packed[5 * n + idx] = f5[idx];
    packed[6 * n + idx] = f6[idx];
    packed[7 * n + idx] = f7[idx];
}

// field-major slab layout:
// slab[field * (n_blocks * n_per_block) + block * n_per_block + idx]
__global__ void scatter8_from_slab(const float_type* slab,
    const float_type* const* f0, const float_type* const* f1,
    const float_type* const* f2, const float_type* const* f3,
    const float_type* const* f4, const float_type* const* f5,
    const float_type* const* f6, const float_type* const* f7, int n_blocks,
    size_t n_per_block)
{
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    const size_t total = static_cast<size_t>(n_blocks) * n_per_block;
    if (gid >= total) return;
    const int b = static_cast<int>(gid / n_per_block);
    const size_t i = gid - static_cast<size_t>(b) * n_per_block;
    const_cast<float_type*>(f0[b])[i] = slab[0 * total + gid];
    const_cast<float_type*>(f1[b])[i] = slab[1 * total + gid];
    const_cast<float_type*>(f2[b])[i] = slab[2 * total + gid];
    const_cast<float_type*>(f3[b])[i] = slab[3 * total + gid];
    const_cast<float_type*>(f4[b])[i] = slab[4 * total + gid];
    const_cast<float_type*>(f5[b])[i] = slab[5 * total + gid];
    const_cast<float_type*>(f6[b])[i] = slab[6 * total + gid];
    const_cast<float_type*>(f7[b])[i] = slab[7 * total + gid];
}

__global__ void scatter6_from_slab(const float_type* slab,
    const float_type* const* f0, const float_type* const* f1,
    const float_type* const* f2, const float_type* const* f3,
    const float_type* const* f4, const float_type* const* f5, int n_blocks,
    size_t n_per_block)
{
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    const size_t total = static_cast<size_t>(n_blocks) * n_per_block;
    if (gid >= total) return;
    const int b = static_cast<int>(gid / n_per_block);
    const size_t i = gid - static_cast<size_t>(b) * n_per_block;
    const_cast<float_type*>(f0[b])[i] = slab[0 * total + gid];
    const_cast<float_type*>(f1[b])[i] = slab[1 * total + gid];
    const_cast<float_type*>(f2[b])[i] = slab[2 * total + gid];
    const_cast<float_type*>(f3[b])[i] = slab[3 * total + gid];
    const_cast<float_type*>(f4[b])[i] = slab[4 * total + gid];
    const_cast<float_type*>(f5[b])[i] = slab[5 * total + gid];
}

__global__ void scatter3_from_slab(const float_type* slab,
    const float_type* const* f0, const float_type* const* f1,
    const float_type* const* f2, int n_blocks, size_t n_per_block)
{
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    const size_t total = static_cast<size_t>(n_blocks) * n_per_block;
    if (gid >= total) return;
    const int b = static_cast<int>(gid / n_per_block);
    const size_t i = gid - static_cast<size_t>(b) * n_per_block;
    const_cast<float_type*>(f0[b])[i] = slab[0 * total + gid];
    const_cast<float_type*>(f1[b])[i] = slab[1 * total + gid];
    const_cast<float_type*>(f2[b])[i] = slab[2 * total + gid];
}

__global__ void gather3_to_slab(const float_type* const* f0,
    const float_type* const* f1, const float_type* const* f2, float_type* slab,
    int n_blocks, size_t n_per_block)
{
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    const size_t total = static_cast<size_t>(n_blocks) * n_per_block;
    if (gid >= total) return;
    const int b = static_cast<int>(gid / n_per_block);
    const size_t i = gid - static_cast<size_t>(b) * n_per_block;
    slab[0 * total + gid] = f0[b][i];
    slab[1 * total + gid] = f1[b][i];
    slab[2 * total + gid] = f2[b][i];
}

__global__ void gather1_to_slab(const float_type* const* f0, float_type* slab,
    int n_blocks, size_t n_per_block)
{
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    const size_t total = static_cast<size_t>(n_blocks) * n_per_block;
    if (gid >= total) return;
    const int b = static_cast<int>(gid / n_per_block);
    const size_t i = gid - static_cast<size_t>(b) * n_per_block;
    slab[gid] = f0[b][i];
}

__device__ inline bool halo_index_3d(int i, int j, int k, int ex, int ey, int ez,
    int wx, int wy, int wz, size_t& idx_out, size_t& n_halo_out)
{
    const int mx = ex - 2 * wx;
    const int my = ey - 2 * wy;
    const int mz = ez - 2 * wz;
    if (mx <= 0 || my <= 0 || mz <= 0)
    {
        idx_out = static_cast<size_t>(i) + static_cast<size_t>(ex) *
                                           (static_cast<size_t>(j) +
                                               static_cast<size_t>(ey) *
                                                   static_cast<size_t>(k));
        n_halo_out = static_cast<size_t>(ex) * ey * ez;
        return true;
    }

    const size_t nA = static_cast<size_t>(2 * wz) * ex * ey;
    const size_t nB = static_cast<size_t>(ez - 2 * wz) * (2 * wy) * ex;
    const size_t nC =
        static_cast<size_t>(ez - 2 * wz) * (ey - 2 * wy) * (2 * wx);
    n_halo_out = nA + nB + nC;

    if (k < wz)
    {
        idx_out = static_cast<size_t>(k) * ex * ey +
                  static_cast<size_t>(j) * ex + i;
        return true;
    }
    if (k >= ez - wz)
    {
        idx_out = static_cast<size_t>(wz) * ex * ey +
                  static_cast<size_t>(k - (ez - wz)) * ex * ey +
                  static_cast<size_t>(j) * ex + i;
        return true;
    }
    if (j < wy)
    {
        idx_out = nA + static_cast<size_t>(k - wz) * (2 * wy) * ex +
                  static_cast<size_t>(j) * ex + i;
        return true;
    }
    if (j >= ey - wy)
    {
        idx_out = nA + static_cast<size_t>(k - wz) * (2 * wy) * ex +
                  static_cast<size_t>(wy) * ex +
                  static_cast<size_t>(j - (ey - wy)) * ex + i;
        return true;
    }
    if (i < wx)
    {
        idx_out = nA + nB +
                  static_cast<size_t>(k - wz) * (ey - 2 * wy) * (2 * wx) +
                  static_cast<size_t>(j - wy) * (2 * wx) + i;
        return true;
    }
    if (i >= ex - wx)
    {
        idx_out = nA + nB +
                  static_cast<size_t>(k - wz) * (ey - 2 * wy) * (2 * wx) +
                  static_cast<size_t>(j - wy) * (2 * wx) + wx +
                  static_cast<size_t>(i - (ex - wx));
        return true;
    }
    return false;
}

__global__ void gather3_halo_to_slab(const float_type* const* f0,
    const float_type* const* f1, const float_type* const* f2, float_type* slab,
    int n_blocks, int ex, int ey, int ez, int wx, int wy, int wz)
{
    const size_t n_total_points = static_cast<size_t>(n_blocks) * ex * ey * ez;
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    if (gid >= n_total_points) return;

    const size_t points_per_block = static_cast<size_t>(ex) * ey * ez;
    const int b = static_cast<int>(gid / points_per_block);
    size_t p = gid - static_cast<size_t>(b) * points_per_block;
    const int i = static_cast<int>(p % ex);
    p /= ex;
    const int j = static_cast<int>(p % ey);
    const int k = static_cast<int>(p / ey);

    size_t hidx = 0, n_halo = 0;
    if (!halo_index_3d(i, j, k, ex, ey, ez, wx, wy, wz, hidx, n_halo)) return;
    const size_t block_off = static_cast<size_t>(b) * n_halo;
    const size_t out_idx = block_off + hidx;
    const size_t lin = static_cast<size_t>(i) + static_cast<size_t>(ex) *
                                              (static_cast<size_t>(j) +
                                                  static_cast<size_t>(ey) *
                                                      static_cast<size_t>(k));
    const size_t total_halo = static_cast<size_t>(n_blocks) * n_halo;
    slab[0 * total_halo + out_idx] = f0[b][lin];
    slab[1 * total_halo + out_idx] = f1[b][lin];
    slab[2 * total_halo + out_idx] = f2[b][lin];
}

__global__ void scatter3_halo_from_slab(const float_type* slab,
    const float_type* const* f0, const float_type* const* f1,
    const float_type* const* f2, int n_blocks, int ex, int ey, int ez, int wx,
    int wy, int wz)
{
    const size_t n_total_points = static_cast<size_t>(n_blocks) * ex * ey * ez;
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    if (gid >= n_total_points) return;

    const size_t points_per_block = static_cast<size_t>(ex) * ey * ez;
    const int b = static_cast<int>(gid / points_per_block);
    size_t p = gid - static_cast<size_t>(b) * points_per_block;
    const int i = static_cast<int>(p % ex);
    p /= ex;
    const int j = static_cast<int>(p % ey);
    const int k = static_cast<int>(p / ey);

    size_t hidx = 0, n_halo = 0;
    if (!halo_index_3d(i, j, k, ex, ey, ez, wx, wy, wz, hidx, n_halo)) return;
    const size_t block_off = static_cast<size_t>(b) * n_halo;
    const size_t in_idx = block_off + hidx;
    const size_t lin = static_cast<size_t>(i) + static_cast<size_t>(ex) *
                                              (static_cast<size_t>(j) +
                                                  static_cast<size_t>(ey) *
                                                      static_cast<size_t>(k));
    const size_t total_halo = static_cast<size_t>(n_blocks) * n_halo;
    const_cast<float_type*>(f0[b])[lin] = slab[0 * total_halo + in_idx];
    const_cast<float_type*>(f1[b])[lin] = slab[1 * total_halo + in_idx];
    const_cast<float_type*>(f2[b])[lin] = slab[2 * total_halo + in_idx];
}

__global__ void gather8_to_slab(const float_type* const* f0,
    const float_type* const* f1, const float_type* const* f2,
    const float_type* const* f3, const float_type* const* f4,
    const float_type* const* f5, const float_type* const* f6,
    const float_type* const* f7, float_type* slab, int n_blocks,
    size_t n_per_block)
{
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    const size_t total = static_cast<size_t>(n_blocks) * n_per_block;
    if (gid >= total) return;
    const int b = static_cast<int>(gid / n_per_block);
    const size_t i = gid - static_cast<size_t>(b) * n_per_block;
    slab[0 * total + gid] = f0[b][i];
    slab[1 * total + gid] = f1[b][i];
    slab[2 * total + gid] = f2[b][i];
    slab[3 * total + gid] = f3[b][i];
    slab[4 * total + gid] = f4[b][i];
    slab[5 * total + gid] = f5[b][i];
    slab[6 * total + gid] = f6[b][i];
    slab[7 * total + gid] = f7[b][i];
}

__global__ void laplace_kernel_batched(const float_type* const* src,
    float_type* const* dst, int n_blocks, int ex, int ey, int ox, int oy,
    int oz, int nx, int ny, int nz, float_type fac)
{
    const size_t interior_size =
        static_cast<size_t>(nx) * ny * static_cast<size_t>(nz);
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    const size_t total = interior_size * static_cast<size_t>(n_blocks);
    if (gid >= total) return;

    const int b = static_cast<int>(gid / interior_size);
    size_t    local = gid - static_cast<size_t>(b) * interior_size;
    const int i = static_cast<int>(local % nx);
    local /= nx;
    const int j = static_cast<int>(local % ny);
    const int k = static_cast<int>(local / ny);

    const int plane = ex * ey;
    const int idx = (i + ox) + ex * ((j + oy) + ey * (k + oz));

    const float_type* s = src[b];
    float_type*       d = dst[b];
    d[idx] = fac * (-6.0 * s[idx] + s[idx - 1] + s[idx + 1] + s[idx - ex] +
                    s[idx + ex] + s[idx - plane] + s[idx + plane]);
}

__global__ void gradient_kernel_batched(const float_type* const* src,
    float_type* const* dst0, float_type* const* dst1, float_type* const* dst2,
    int n_blocks, int ex, int ey, int ox, int oy, int oz, int nx, int ny,
    int nz, float_type fac)
{
    const size_t interior_size =
        static_cast<size_t>(nx) * ny * static_cast<size_t>(nz);
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    const size_t total = interior_size * static_cast<size_t>(n_blocks);
    if (gid >= total) return;

    const int b = static_cast<int>(gid / interior_size);
    size_t    local = gid - static_cast<size_t>(b) * interior_size;
    const int i = static_cast<int>(local % nx);
    local /= nx;
    const int j = static_cast<int>(local % ny);
    const int k = static_cast<int>(local / ny);

    const int plane = ex * ey;
    const int idx = (i + ox) + ex * ((j + oy) + ey * (k + oz));

    const float_type* s = src[b];
    dst0[b][idx] = fac * (s[idx] - s[idx - 1]);
    dst1[b][idx] = fac * (s[idx] - s[idx - ex]);
    dst2[b][idx] = fac * (s[idx] - s[idx - plane]);
}

__global__ void divergence_kernel_batched(const float_type* const* src0,
    const float_type* const* src1, const float_type* const* src2,
    float_type* const* dst, int n_blocks, int ex, int ey, int ox, int oy,
    int oz, int nx, int ny, int nz, float_type fac)
{
    const size_t interior_size =
        static_cast<size_t>(nx) * ny * static_cast<size_t>(nz);
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    const size_t total = interior_size * static_cast<size_t>(n_blocks);
    if (gid >= total) return;

    const int b = static_cast<int>(gid / interior_size);
    size_t    local = gid - static_cast<size_t>(b) * interior_size;
    const int i = static_cast<int>(local % nx);
    local /= nx;
    const int j = static_cast<int>(local % ny);
    const int k = static_cast<int>(local / ny);

    const int plane = ex * ey;
    const int idx = (i + ox) + ex * ((j + oy) + ey * (k + oz));

    dst[b][idx] =
        fac * (-src0[b][idx] - src1[b][idx] - src2[b][idx] + src0[b][idx + 1] +
               src1[b][idx + ex] + src2[b][idx + plane]);
}

__global__ void curl_kernel_batched(const float_type* const* src0,
    const float_type* const* src1, const float_type* const* src2,
    float_type* const* dst0, float_type* const* dst1, float_type* const* dst2,
    int n_blocks, int ex, int ey, int ox, int oy, int oz, int nx, int ny,
    int nz, float_type fac)
{
    const size_t interior_size =
        static_cast<size_t>(nx) * ny * static_cast<size_t>(nz);
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    const size_t total = interior_size * static_cast<size_t>(n_blocks);
    if (gid >= total) return;

    const int b = static_cast<int>(gid / interior_size);
    size_t    local = gid - static_cast<size_t>(b) * interior_size;
    const int i = static_cast<int>(local % nx);
    local /= nx;
    const int j = static_cast<int>(local % ny);
    const int k = static_cast<int>(local / ny);

    const int plane = ex * ey;
    const int idx = (i + ox) + ex * ((j + oy) + ey * (k + oz));

    dst0[b][idx] = fac * ((src2[b][idx] - src2[b][idx - ex]) -
                          (src1[b][idx] - src1[b][idx - plane]));
    dst1[b][idx] = fac * ((src0[b][idx] - src0[b][idx - plane]) -
                          (src2[b][idx] - src2[b][idx - 1]));
    dst2[b][idx] = fac * ((src1[b][idx] - src1[b][idx - 1]) -
                          (src0[b][idx] - src0[b][idx - ex]));
}

__global__ void nonlinear_kernel_batched(const float_type* const* face0,
    const float_type* const* face1, const float_type* const* face2,
    const float_type* const* edge0, const float_type* const* edge1,
    const float_type* const* edge2, float_type* const* dst0,
    float_type* const* dst1, float_type* const* dst2, int n_blocks, int ex,
    int ey, int ox, int oy, int oz, int nx, int ny, int nz)
{
    const size_t interior_size =
        static_cast<size_t>(nx) * ny * static_cast<size_t>(nz);
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       static_cast<size_t>(threadIdx.x);
    const size_t total = interior_size * static_cast<size_t>(n_blocks);
    if (gid >= total) return;

    const int b = static_cast<int>(gid / interior_size);
    size_t    local = gid - static_cast<size_t>(b) * interior_size;
    const int i = static_cast<int>(local % nx);
    local /= nx;
    const int j = static_cast<int>(local % ny);
    const int k = static_cast<int>(local / ny);

    const int plane = ex * ey;
    const int idx = (i + ox) + ex * ((j + oy) + ey * (k + oz));

    dst0[b][idx] =
        0.25 * (+edge1[b][idx] * (+face2[b][idx] + face2[b][idx - 1]) +
                edge1[b][idx + plane] *
                    (+face2[b][idx + plane] + face2[b][idx + plane - 1]) -
                edge2[b][idx] * (+face1[b][idx] + face1[b][idx - 1]) -
                edge2[b][idx + ex] *
                    (+face1[b][idx + ex] + face1[b][idx + ex - 1]));

    dst1[b][idx] =
        0.25 * (+edge2[b][idx] * (+face0[b][idx] + face0[b][idx - ex]) +
                edge2[b][idx + 1] *
                    (+face0[b][idx + 1] + face0[b][idx + 1 - ex]) -
                edge0[b][idx] * (+face2[b][idx] + face2[b][idx - ex]) -
                edge0[b][idx + plane] *
                    (+face2[b][idx + plane] + face2[b][idx + plane - ex]));

    dst2[b][idx] = 0.25 * (+edge0[b][idx] * (+face1[b][idx] + face1[b][idx - plane]) +
                           edge0[b][idx + ex] *
                               (+face1[b][idx + ex] + face1[b][idx + ex - plane]) -
                           edge1[b][idx] * (+face0[b][idx] + face0[b][idx - plane]) -
                           edge1[b][idx + 1] *
                               (+face0[b][idx + 1] + face0[b][idx + 1 - plane]));
}
} // namespace operators_gpu_cuda

} // namespace iblgf

#endif // IBLGF_COMPILE_CUDA

#endif // IBLGF_INCLUDED_OPERATORS_GPU_KERNELS_HPP
