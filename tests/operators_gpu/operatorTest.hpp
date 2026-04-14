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

#ifndef IBLGF_INCLUDED_OPERATORTEST_HPP
#define IBLGF_INCLUDED_OPERATORTEST_HPP

#include <iostream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <iblgf/dictionary/dictionary.hpp>

#include "../../setups/setup_base.hpp"
#include <iblgf/operators/operators.hpp>
#ifdef IBLGF_COMPILE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#endif

namespace iblgf
{
using namespace domain;
using namespace types;
using namespace dictionary;

const int Dim = 3;

#ifdef IBLGF_COMPILE_CUDA
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
#endif

struct parameters
{
    static constexpr std::size_t Dim = 3;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
        (
            //name, type, nFields, l/h-buf,mesh_obj, output(optional)
            (grad_source,       float_type, 1, 1, 1, cell, true),
            (grad_target,       float_type, 3, 1, 1, face, true),
            (grad_exact,        float_type, 3, 1, 1, face, true),
            (grad_error,        float_type, 3, 1, 1, face, true),

            (lap_source,        float_type, 1, 1, 1, cell, true),
            (lap_target,        float_type, 1, 1, 1, cell, true),
            (lap_exact,         float_type, 1, 1, 1, cell, true),
            (lap_error,         float_type, 1, 1, 1, cell, true),

            (div_source,        float_type, 3, 1, 1, face, true),
            (div_target,        float_type, 1, 1, 1, cell, true),
            (div_exact,         float_type, 1, 1, 1, cell, true),
            (div_error,         float_type, 1, 1, 1, cell, true),

            (curl_source,       float_type, 3, 1, 1, face, true),
            (curl_target,       float_type, 3, 1, 1, edge, true),
            (curl_exact,        float_type, 3, 1, 1, edge, true),
            (curl_error,        float_type, 3, 1, 1, edge, true),

            (nonlinear_source,  float_type, 3, 1, 1, face, true),
            (nonlinear_target,  float_type, 3, 1, 1, face, true),
            (nonlinear_exact,   float_type, 3, 1, 1, face, true),
            (nonlinear_error,   float_type, 3, 1, 1, face, true)
        )
    )
    // clang-format on
};

struct OperatorTest : public SetupBase<OperatorTest, parameters>
{
    using super_type = SetupBase<OperatorTest, parameters>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

    OperatorTest(Dictionary* _d)
    : super_type(_d, [this](auto _d, auto _domain) {
        return this->initialize_domain(_d, _domain);
    })
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);

        global_refinement_ = simulation_.dictionary_->template get_or<int>(
            "global_refinement", 0);

        pcout << "\n Setup:  Test - Vortex ring \n" << std::endl;
        pcout << "Number of refinement levels: " << nLevels_ << std::endl;

        domain_->register_refinement_condition() = [this](auto octant,
                                                       int     diff_level) {
            return this->refinement(octant, diff_level);
        };
        domain_->init_refine(_d->get_dictionary("simulation_parameters")
                                 ->template get_or<int>("nLevels", 0),
            global_refinement_,0);
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        this->initialize();

        boost::mpi::communicator world;
        if (world.rank() == 0)
            std::cout << "on Simulation: \n" << simulation_ << std::endl;
    }

    void run()
    {
        boost::mpi::communicator world;
        if (domain_->is_client())
        {
            const float_type dx_base = domain_->dx_base();
            const float_type base_level = domain_->tree()->base_level();
            const int max_refinement_level =
                nLevels_ + global_refinement_ + static_cast<int>(base_level);
            using block_ptr_t = super_type::datablock_t*;
            std::vector<std::vector<block_ptr_t>> level_blocks(
                max_refinement_level + 1);
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                const int level = it->refinement_level();
                if (level < 0 || level > max_refinement_level) continue;
                level_blocks[level].push_back(&it->data());
            }

            //Bufffer exchange of some fields
            auto client = domain_->decomposition().client();
            client->buffer_exchange<lap_source_type>(base_level);
            client->buffer_exchange<div_source_type>(base_level);
            client->buffer_exchange<curl_source_type>(base_level);
            client->buffer_exchange<grad_source_type>(base_level);
            client->buffer_exchange<curl_exact_type>(base_level);
            client->buffer_exchange<nonlinear_source_type>(base_level);

            for (int level = 0; level <= max_refinement_level; ++level)
            {
                if (level_blocks[level].empty()) continue;
                auto dx_level = dx_base / std::pow(2, level);
                this->apply_derivatives_cuda_batched(
                    level_blocks[level], dx_level, level);
            }

            for (int level = 0; level <= max_refinement_level; ++level)
            {
                if (level_blocks[level].empty()) continue;
                this->sync_curl_target_to_host_batched(level_blocks[level], level);
            }

            client->buffer_exchange<curl_target_type>(base_level);

            for (int level = 0; level <= max_refinement_level; ++level)
            {
                if (level_blocks[level].empty()) continue;
                this->sync_curl_target_to_device_batched(level_blocks[level], level);
            }

            for (int level = 0; level <= max_refinement_level; ++level)
            {
                if (level_blocks[level].empty()) continue;
                this->apply_nonlinear_cuda_batched(level_blocks[level], level);
            }

            for (int level = 0; level <= max_refinement_level; ++level)
            {
                if (level_blocks[level].empty()) continue;
                this->sync_targets_to_host_batched(level_blocks[level], level);
            }
            for (int level = 0; level <= max_refinement_level; ++level)
            {
                if (level_blocks[level].empty()) continue;
                this->sync_curl_target_full_to_host_batched(
                    level_blocks[level], level);
            }
        }

        this->compute_errors<lap_target_type, lap_exact_type, lap_error_type>(
            "Lap_");
        this->compute_errors<grad_target_type, grad_exact_type,
            grad_error_type>("Grad_");
        this->compute_errors<div_target_type, div_exact_type, div_error_type>(
            "Div_");
        this->compute_errors<curl_target_type, curl_exact_type,
            curl_error_type>("Curl_");
        this->compute_errors<nonlinear_target_type, nonlinear_exact_type,
            nonlinear_error_type>("Nonlin_");
        simulation_.write("mesh.hdf5");
    }

    /** @brief Initialization of poisson problem.
     *  @detail Testing poisson with manufactured solutions: exp
     */
    void initialize()
    {
        boost::mpi::communicator world;
        if (domain_->is_server()) return;
        auto center =
            (domain_->bounding_box().max() - domain_->bounding_box().min()) /
                2.0 +
            domain_->bounding_box().min();

        // Adapt center to always have peak value in a cell-center
        //center+=0.5/std::pow(2,nRef);
        const float_type dx_base = domain_->dx_base();

        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
             ++it)
        {
            if (!it->locally_owned()) continue;
            if (!(*it && it->has_data())) continue;
            auto dx_level = dx_base / std::pow(2, it->refinement_level());
            auto scaling = std::pow(2, it->refinement_level());

            for (auto& node : it->data())
            {
                const auto& coord = node.level_coordinate();

                //Cell centered coordinates
                //This can obviously be made much less verbose
                float_type xc = static_cast<float_type>(
                                    coord[0] - center[0] * scaling + 0.5) *
                                dx_level;
                float_type yc = static_cast<float_type>(
                                    coord[1] - center[1] * scaling + 0.5) *
                                dx_level;
                float_type zc = static_cast<float_type>(
                                    coord[2] - center[2] * scaling + 0.5) *
                                dx_level;

                //Face centered coordinates
                float_type xf0 =
                    static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                float_type yf0 = yc;
                float_type zf0 = zc;

                float_type xf1 = xc;
                float_type yf1 =
                    static_cast<float_type>(coord[1] - center[1] * scaling) *
                    dx_level;
                float_type zf1 = zc;

                float_type xf2 = xc;
                float_type yf2 = yc;
                float_type zf2 =
                    static_cast<float_type>(coord[2] - center[2] * scaling) *
                    dx_level;

                //Edge centered coordinates
                float_type xe0 = static_cast<float_type>(
                                     coord[0] - center[0] * scaling + 0.5) *
                                 dx_level;
                float_type ye0 =
                    static_cast<float_type>(coord[1] - center[1] * scaling) *
                    dx_level;
                float_type ze0 =
                    static_cast<float_type>(coord[2] - center[2] * scaling) *
                    dx_level;
                float_type xe1 =
                    static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                float_type ye1 = static_cast<float_type>(
                                     coord[1] - center[1] * scaling + 0.5) *
                                 dx_level;
                float_type ze1 =
                    static_cast<float_type>(coord[2] - center[2] * scaling) *
                    dx_level;
                float_type xe2 =
                    static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                float_type ye2 =
                    static_cast<float_type>(coord[1] - center[1] * scaling) *
                    dx_level;
                float_type ze2 = static_cast<float_type>(
                                     coord[2] - center[2] * scaling + 0.5) *
                                 dx_level;

                const float_type r = std::sqrt(xc * xc + yc * yc + zc * zc);
                const float_type rf0 =
                    std::sqrt(xf0 * xf0 + yf0 * yf0 + zf0 * zf0);
                const float_type rf1 =
                    std::sqrt(xf1 * xf1 + yf1 * yf1 + zf1 * zf1);
                const float_type rf2 =
                    std::sqrt(xf2 * xf2 + yf2 * yf2 + zf2 * zf2);
                const float_type re0 =
                    std::sqrt(xe0 * xe0 + ye0 * ye0 + ze0 * ze0);
                const float_type re1 =
                    std::sqrt(xe1 * xe1 + ye1 * ye1 + ze1 * ze1);
                const float_type re2 =
                    std::sqrt(xe2 * xe2 + ye2 * ye2 + ze2 * ze2);
                const float_type a2 = a_ * a_;
                const float_type xc2 = xc * xc;
                const float_type yc2 = yc * yc;
                const float_type zc2 = zc * zc;
                /***********************************************************/

                float_type r_2 = r * r;
                const auto fct = std::exp(-a_ * r_2);
                const auto tmpc = std::exp(-a_ * r_2);

                const auto tmpf0 = std::exp(-a_ * rf0 * rf0);
                const auto tmpf1 = std::exp(-a_ * rf1 * rf1);
                const auto tmpf2 = std::exp(-a_ * rf2 * rf2);

                const auto tmpe0 = std::exp(-a_ * re0 * re0);
                const auto tmpe1 = std::exp(-a_ * re1 * re1);
                const auto tmpe2 = std::exp(-a_ * re2 * re2);

                //Gradient
                node(grad_source) = fct;
                node(grad_exact, 0) = -2 * a_ * xf0 * tmpf0;
                node(grad_exact, 1) = -2 * a_ * yf1 * tmpf1;
                node(grad_exact, 2) = -2 * a_ * zf2 * tmpf2;

                //Laplace
                node(lap_source, 0) = tmpc;
                node(lap_exact) = -6 * a_ * tmpc + 4 * a2 * xc2 * tmpc +
                                  4 * a2 * yc2 * tmpc + 4 * a2 * zc2 * tmpc;

                //Divergence
                node(div_source, 0) = tmpf0;
                node(div_source, 1) = tmpf1;
                node(div_source, 2) = tmpf2;
                node(div_exact, 0) = -2 * a_ * xc * tmpc - 2 * a_ * yc * tmpc -
                                     2 * a_ * zc * tmpc;

                //Curl
                node(curl_source, 0) = tmpf0;
                node(curl_source, 1) = tmpf1;
                node(curl_source, 2) = tmpf2;

                node(curl_exact, 0) =
                    2 * a_ * ze0 * tmpe0 - 2 * a_ * ye0 * tmpe0;
                node(curl_exact, 1) =
                    2 * a_ * xe1 * tmpe1 - 2 * a_ * ze1 * tmpe1;
                node(curl_exact, 2) =
                    2 * a_ * ye2 * tmpe2 - 2 * a_ * xe2 * tmpe2;

                //non_linear
                node(nonlinear_source, 0) = tmpf0;
                node(nonlinear_source, 1) = tmpf1;
                node(nonlinear_source, 2) = tmpf2;

                node(nonlinear_exact, 0) =
                    tmpf0 * (2 * a_ * xf0 * tmpf0 - 2 * a_ * yf0 * tmpf0) +
                    tmpf0 * (2 * a_ * xf0 * tmpf0 - 2 * a_ * zf0 * tmpf0);

                node(nonlinear_exact, 1) =
                    tmpf1 * (2 * a_ * yf1 * tmpf1 - 2 * a_ * zf1 * tmpf1) -
                    tmpf1 * (2 * a_ * xf1 * tmpf1 - 2 * a_ * yf1 * tmpf1);

                node(nonlinear_exact, 2) =
                    -tmpf2 * (2 * a_ * xf2 * tmpf2 - 2 * a_ * zf2 * tmpf2) -
                    tmpf2 * (2 * a_ * yf2 * tmpf2 - 2 * a_ * zf2 * tmpf2);
            }
        }
    }

    /** @brief  Refienment conditon for octants.  */
    template<class OctantType>
    bool refinement(
        OctantType* it, int diff_level, bool use_all = false) const noexcept
    {
        return false;
    }

    /** @brief  Initialization of the domain blocks. This is registered in the
     *          domain through the base setup class, passing it to the domain ctor.
     */
    std::vector<coordinate_t> initialize_domain(
        Dictionary* _d, domain_t* _domain)
    {
        return _domain->construct_basemesh_blocks(_d, _domain->block_extent());
    }

    template<class BlockPtr>
    bool level_layout_is_uniform(const std::vector<BlockPtr>& blocks, int& ex,
        int& ey, int& ez, int& ox, int& oy, int& oz, int& nx, int& ny,
        int& nz) const
    {
        if (blocks.empty()) return false;

        const auto& first_src = (*blocks.front())(lap_source, 0);
        const auto  ext = first_src.real_block().extent();
        const auto  lb = first_src.lbuffer();
        const auto  interior = blocks.front()->descriptor().extent();

        ex = ext[0];
        ey = ext[1];
        ez = ext[2];
        ox = lb[0];
        oy = lb[1];
        oz = lb[2];
        nx = interior[0];
        ny = interior[1];
        nz = interior[2];

        for (auto* block : blocks)
        {
            const auto& src_i = (*block)(lap_source, 0);
            const auto  ext_i = src_i.real_block().extent();
            const auto  lb_i = src_i.lbuffer();
            const auto  interior_i = block->descriptor().extent();
            if (ext_i[0] != ex || ext_i[1] != ey || ext_i[2] != ez ||
                lb_i[0] != ox || lb_i[1] != oy || lb_i[2] != oz ||
                interior_i[0] != nx || interior_i[1] != ny ||
                interior_i[2] != nz)
                return false;
        }
        return true;
    }

    bool halo_index_host(int i, int j, int k, int ex, int ey, int ez, int wx,
        int wy, int wz, size_t& idx_out, size_t& n_halo_out) const
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

#ifdef IBLGF_COMPILE_CUDA
    void ensure_cuda_stream()
    {
        if (compute_stream_) return;
        operators_gpu_cuda::cuda_check(
            cudaStreamCreate(&compute_stream_), "Create compute stream");
    }

    template<typename PtrType>
    PtrType* upload_pointer_table(
        const std::vector<PtrType>& host_ptrs, const char* context)
    {
        PtrType* d_ptrs = nullptr;
        operators_gpu_cuda::cuda_check(
            cudaMalloc(reinterpret_cast<void**>(&d_ptrs),
                host_ptrs.size() * sizeof(PtrType)),
            context);
        operators_gpu_cuda::cuda_check(cudaMemcpyAsync(d_ptrs, host_ptrs.data(),
                                             host_ptrs.size() * sizeof(PtrType),
                                             cudaMemcpyHostToDevice,
                                             compute_stream_),
            context);
        return d_ptrs;
    }

    struct LevelPointerTables
    {
        bool initialized = false;
        std::size_t n_blocks = 0;

        const float_type** d_lap_src = nullptr;
        const float_type** d_grad_src = nullptr;
        const float_type** d_div_src0 = nullptr;
        const float_type** d_div_src1 = nullptr;
        const float_type** d_div_src2 = nullptr;
        const float_type** d_curl_src0 = nullptr;
        const float_type** d_curl_src1 = nullptr;
        const float_type** d_curl_src2 = nullptr;

        float_type** d_lap_dst = nullptr;
        float_type** d_grad_dst0 = nullptr;
        float_type** d_grad_dst1 = nullptr;
        float_type** d_grad_dst2 = nullptr;
        float_type** d_div_dst = nullptr;
        float_type** d_curl_dst0 = nullptr;
        float_type** d_curl_dst1 = nullptr;
        float_type** d_curl_dst2 = nullptr;

        const float_type** d_nonlin_src0 = nullptr;
        const float_type** d_nonlin_src1 = nullptr;
        const float_type** d_nonlin_src2 = nullptr;
        float_type** d_nonlin_dst0 = nullptr;
        float_type** d_nonlin_dst1 = nullptr;
        float_type** d_nonlin_dst2 = nullptr;
    };

    void free_level_pointer_tables(LevelPointerTables& t)
    {
        if (t.d_lap_src) cudaFree((void*)t.d_lap_src);
        if (t.d_grad_src) cudaFree((void*)t.d_grad_src);
        if (t.d_div_src0) cudaFree((void*)t.d_div_src0);
        if (t.d_div_src1) cudaFree((void*)t.d_div_src1);
        if (t.d_div_src2) cudaFree((void*)t.d_div_src2);
        if (t.d_curl_src0) cudaFree((void*)t.d_curl_src0);
        if (t.d_curl_src1) cudaFree((void*)t.d_curl_src1);
        if (t.d_curl_src2) cudaFree((void*)t.d_curl_src2);
        if (t.d_lap_dst) cudaFree((void*)t.d_lap_dst);
        if (t.d_grad_dst0) cudaFree((void*)t.d_grad_dst0);
        if (t.d_grad_dst1) cudaFree((void*)t.d_grad_dst1);
        if (t.d_grad_dst2) cudaFree((void*)t.d_grad_dst2);
        if (t.d_div_dst) cudaFree((void*)t.d_div_dst);
        if (t.d_curl_dst0) cudaFree((void*)t.d_curl_dst0);
        if (t.d_curl_dst1) cudaFree((void*)t.d_curl_dst1);
        if (t.d_curl_dst2) cudaFree((void*)t.d_curl_dst2);
        if (t.d_nonlin_src0) cudaFree((void*)t.d_nonlin_src0);
        if (t.d_nonlin_src1) cudaFree((void*)t.d_nonlin_src1);
        if (t.d_nonlin_src2) cudaFree((void*)t.d_nonlin_src2);
        if (t.d_nonlin_dst0) cudaFree((void*)t.d_nonlin_dst0);
        if (t.d_nonlin_dst1) cudaFree((void*)t.d_nonlin_dst1);
        if (t.d_nonlin_dst2) cudaFree((void*)t.d_nonlin_dst2);
        t = LevelPointerTables{};
    }

    template<class BlockPtr>
    LevelPointerTables& get_or_build_level_pointer_tables(
        int level_id, const std::vector<BlockPtr>& blocks)
    {
        auto& tbl = level_pointer_tables_[level_id];
        if (tbl.initialized && tbl.n_blocks == blocks.size()) return tbl;
        if (tbl.initialized) free_level_pointer_tables(tbl);

        std::vector<const float_type*> h_lap_src, h_grad_src, h_div_src0,
            h_div_src1, h_div_src2, h_curl_src0, h_curl_src1, h_curl_src2;
        std::vector<float_type*> h_lap_dst, h_grad_dst0, h_grad_dst1, h_grad_dst2,
            h_div_dst, h_curl_dst0, h_curl_dst1, h_curl_dst2;
        std::vector<const float_type*> h_nonlin_src0, h_nonlin_src1, h_nonlin_src2;
        std::vector<float_type*> h_nonlin_dst0, h_nonlin_dst1, h_nonlin_dst2;

        h_lap_src.reserve(blocks.size());
        h_grad_src.reserve(blocks.size());
        h_div_src0.reserve(blocks.size());
        h_div_src1.reserve(blocks.size());
        h_div_src2.reserve(blocks.size());
        h_curl_src0.reserve(blocks.size());
        h_curl_src1.reserve(blocks.size());
        h_curl_src2.reserve(blocks.size());
        h_lap_dst.reserve(blocks.size());
        h_grad_dst0.reserve(blocks.size());
        h_grad_dst1.reserve(blocks.size());
        h_grad_dst2.reserve(blocks.size());
        h_div_dst.reserve(blocks.size());
        h_curl_dst0.reserve(blocks.size());
        h_curl_dst1.reserve(blocks.size());
        h_curl_dst2.reserve(blocks.size());
        h_nonlin_src0.reserve(blocks.size());
        h_nonlin_src1.reserve(blocks.size());
        h_nonlin_src2.reserve(blocks.size());
        h_nonlin_dst0.reserve(blocks.size());
        h_nonlin_dst1.reserve(blocks.size());
        h_nonlin_dst2.reserve(blocks.size());

        for (auto* block : blocks)
        {
            h_lap_src.push_back((*block)(lap_source, 0).device_ptr());
            h_grad_src.push_back((*block)(grad_source, 0).device_ptr());
            h_div_src0.push_back((*block)(div_source, 0).device_ptr());
            h_div_src1.push_back((*block)(div_source, 1).device_ptr());
            h_div_src2.push_back((*block)(div_source, 2).device_ptr());
            h_curl_src0.push_back((*block)(curl_source, 0).device_ptr());
            h_curl_src1.push_back((*block)(curl_source, 1).device_ptr());
            h_curl_src2.push_back((*block)(curl_source, 2).device_ptr());

            h_lap_dst.push_back((*block)(lap_target, 0).device_ptr());
            h_grad_dst0.push_back((*block)(grad_target, 0).device_ptr());
            h_grad_dst1.push_back((*block)(grad_target, 1).device_ptr());
            h_grad_dst2.push_back((*block)(grad_target, 2).device_ptr());
            h_div_dst.push_back((*block)(div_target, 0).device_ptr());
            h_curl_dst0.push_back((*block)(curl_target, 0).device_ptr());
            h_curl_dst1.push_back((*block)(curl_target, 1).device_ptr());
            h_curl_dst2.push_back((*block)(curl_target, 2).device_ptr());

            h_nonlin_src0.push_back((*block)(nonlinear_source, 0).device_ptr());
            h_nonlin_src1.push_back((*block)(nonlinear_source, 1).device_ptr());
            h_nonlin_src2.push_back((*block)(nonlinear_source, 2).device_ptr());
            h_nonlin_dst0.push_back((*block)(nonlinear_target, 0).device_ptr());
            h_nonlin_dst1.push_back((*block)(nonlinear_target, 1).device_ptr());
            h_nonlin_dst2.push_back((*block)(nonlinear_target, 2).device_ptr());
        }

        tbl.d_lap_src =
            upload_pointer_table(h_lap_src, "Upload cached lap src pointer table");
        tbl.d_grad_src = upload_pointer_table(
            h_grad_src, "Upload cached grad src pointer table");
        tbl.d_div_src0 = upload_pointer_table(
            h_div_src0, "Upload cached div src0 pointer table");
        tbl.d_div_src1 = upload_pointer_table(
            h_div_src1, "Upload cached div src1 pointer table");
        tbl.d_div_src2 = upload_pointer_table(
            h_div_src2, "Upload cached div src2 pointer table");
        tbl.d_curl_src0 = upload_pointer_table(
            h_curl_src0, "Upload cached curl src0 pointer table");
        tbl.d_curl_src1 = upload_pointer_table(
            h_curl_src1, "Upload cached curl src1 pointer table");
        tbl.d_curl_src2 = upload_pointer_table(
            h_curl_src2, "Upload cached curl src2 pointer table");

        tbl.d_lap_dst =
            upload_pointer_table(h_lap_dst, "Upload cached lap dst pointer table");
        tbl.d_grad_dst0 = upload_pointer_table(
            h_grad_dst0, "Upload cached grad dst0 pointer table");
        tbl.d_grad_dst1 = upload_pointer_table(
            h_grad_dst1, "Upload cached grad dst1 pointer table");
        tbl.d_grad_dst2 = upload_pointer_table(
            h_grad_dst2, "Upload cached grad dst2 pointer table");
        tbl.d_div_dst =
            upload_pointer_table(h_div_dst, "Upload cached div dst pointer table");
        tbl.d_curl_dst0 = upload_pointer_table(
            h_curl_dst0, "Upload cached curl dst0 pointer table");
        tbl.d_curl_dst1 = upload_pointer_table(
            h_curl_dst1, "Upload cached curl dst1 pointer table");
        tbl.d_curl_dst2 = upload_pointer_table(
            h_curl_dst2, "Upload cached curl dst2 pointer table");

        tbl.d_nonlin_src0 = upload_pointer_table(
            h_nonlin_src0, "Upload cached nonlinear src0 pointer table");
        tbl.d_nonlin_src1 = upload_pointer_table(
            h_nonlin_src1, "Upload cached nonlinear src1 pointer table");
        tbl.d_nonlin_src2 = upload_pointer_table(
            h_nonlin_src2, "Upload cached nonlinear src2 pointer table");
        tbl.d_nonlin_dst0 = upload_pointer_table(
            h_nonlin_dst0, "Upload cached nonlinear dst0 pointer table");
        tbl.d_nonlin_dst1 = upload_pointer_table(
            h_nonlin_dst1, "Upload cached nonlinear dst1 pointer table");
        tbl.d_nonlin_dst2 = upload_pointer_table(
            h_nonlin_dst2, "Upload cached nonlinear dst2 pointer table");

        tbl.n_blocks = blocks.size();
        tbl.initialized = true;
        return tbl;
    }
#endif

    template<class BlockPtr>
    void apply_derivatives_cuda_batched(
        const std::vector<BlockPtr>& blocks, float_type dx_level,
        int level_id)
    {
#ifdef IBLGF_COMPILE_CUDA
        if (blocks.empty()) return;
        (void)level_id;
        ensure_cuda_stream();

        int ex = 0, ey = 0, ez = 0, ox = 0, oy = 0, oz = 0;
        int nx = 0, ny = 0, nz = 0;
        if (!level_layout_is_uniform(blocks, ex, ey, ez, ox, oy, oz, nx, ny, nz))
        {
            for (auto* b : blocks) apply_derivatives_cuda(*b, dx_level);
            return;
        }

        static bool printed_level_deriv = false;
        if (!printed_level_deriv)
        {
            boost::mpi::communicator world;
            std::cout << "[operators_gpu][rank " << world.rank()
                      << "] Running level-batched derivative kernels"
                      << std::endl;
            printed_level_deriv = true;
        }

        for (auto* block : blocks)
        {
            // ensure device allocations exist
            (void)(*block)(lap_source, 0).device_ptr();
            (void)(*block)(grad_source, 0).device_ptr();
            (void)(*block)(div_source, 0).device_ptr();
            (void)(*block)(div_source, 1).device_ptr();
            (void)(*block)(div_source, 2).device_ptr();
            (void)(*block)(curl_source, 0).device_ptr();
            (void)(*block)(curl_source, 1).device_ptr();
            (void)(*block)(curl_source, 2).device_ptr();
            (void)(*block)(lap_target, 0).device_ptr();
            (void)(*block)(grad_target, 0).device_ptr();
            (void)(*block)(grad_target, 1).device_ptr();
            (void)(*block)(grad_target, 2).device_ptr();
            (void)(*block)(div_target, 0).device_ptr();
            (void)(*block)(curl_target, 0).device_ptr();
            (void)(*block)(curl_target, 1).device_ptr();
            (void)(*block)(curl_target, 2).device_ptr();
        }
        auto& tbl = get_or_build_level_pointer_tables(level_id, blocks);

        const size_t n_per_block =
            blocks.empty() ? 0 : (*blocks.front())(lap_source, 0).data().size();
        const size_t slab_items = n_per_block * blocks.size();
        ensure_pack_capacity(h2d_pack_, d_h2d_pack_, h2d_pack_capacity_,
            8 * slab_items);

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            auto& lap_src = (*blocks[b])(lap_source, 0);
            auto& grad_src = (*blocks[b])(grad_source, 0);
            auto& div_src0 = (*blocks[b])(div_source, 0);
            auto& div_src1 = (*blocks[b])(div_source, 1);
            auto& div_src2 = (*blocks[b])(div_source, 2);
            auto& curl_src0 = (*blocks[b])(curl_source, 0);
            auto& curl_src1 = (*blocks[b])(curl_source, 1);
            auto& curl_src2 = (*blocks[b])(curl_source, 2);

            auto copy_field = [&](size_t field_idx, auto& field) {
                std::copy(field.data().begin(), field.data().end(),
                    h2d_pack_.begin() + field_idx * slab_items +
                        b * n_per_block);
            };
            copy_field(0, lap_src);
            copy_field(1, grad_src);
            copy_field(2, div_src0);
            copy_field(3, div_src1);
            copy_field(4, div_src2);
            copy_field(5, curl_src0);
            copy_field(6, curl_src1);
            copy_field(7, curl_src2);
        }

        operators_gpu_cuda::cuda_check(cudaMemcpyAsync(d_h2d_pack_,
                                             h2d_pack_.data(),
                                             8 * slab_items * sizeof(float_type),
                                             cudaMemcpyHostToDevice,
                                             compute_stream_),
            "Batched derivative slab H2D");

        const int threads = 256;
        const int grid_scatter = static_cast<int>(
            (slab_items + threads - 1) / threads);
        operators_gpu_cuda::scatter8_from_slab<<<grid_scatter, threads, 0,
            compute_stream_>>>(d_h2d_pack_, tbl.d_lap_src, tbl.d_grad_src,
            tbl.d_div_src0, tbl.d_div_src1, tbl.d_div_src2, tbl.d_curl_src0,
            tbl.d_curl_src1, tbl.d_curl_src2,
            static_cast<int>(blocks.size()), n_per_block);
        operators_gpu_cuda::cuda_check(
            cudaGetLastError(), "Batched derivative slab scatter");

        const size_t interior_size = static_cast<size_t>(nx) * ny *
                                     static_cast<size_t>(nz);
        const size_t total_threads = interior_size * blocks.size();
        const int grid = static_cast<int>((total_threads + threads - 1) / threads);

        operators_gpu_cuda::laplace_kernel_batched<<<grid, threads, 0,
            compute_stream_>>>(tbl.d_lap_src, tbl.d_lap_dst, static_cast<int>(blocks.size()),
            ex, ey, ox, oy, oz, nx, ny, nz, 1.0 / (dx_level * dx_level));

        operators_gpu_cuda::gradient_kernel_batched<<<grid, threads, 0,
            compute_stream_>>>(tbl.d_grad_src, tbl.d_grad_dst0, tbl.d_grad_dst1, tbl.d_grad_dst2,
            static_cast<int>(blocks.size()), ex, ey, ox, oy, oz, nx, ny, nz,
            1.0 / dx_level);

        operators_gpu_cuda::divergence_kernel_batched<<<grid, threads, 0,
            compute_stream_>>>(tbl.d_div_src0, tbl.d_div_src1, tbl.d_div_src2,
            tbl.d_div_dst,
            static_cast<int>(blocks.size()), ex, ey, ox, oy, oz, nx, ny, nz,
            1.0 / dx_level);

        operators_gpu_cuda::curl_kernel_batched<<<grid, threads, 0,
            compute_stream_>>>(tbl.d_curl_src0, tbl.d_curl_src1, tbl.d_curl_src2,
            tbl.d_curl_dst0, tbl.d_curl_dst1, tbl.d_curl_dst2,
            static_cast<int>(blocks.size()), ex, ey, ox, oy, oz, nx, ny, nz,
            1.0 / dx_level);

        operators_gpu_cuda::cuda_check(
            cudaGetLastError(), "Batched derivative kernel launch");
        operators_gpu_cuda::cuda_check(
            cudaStreamSynchronize(compute_stream_),
            "Batched derivative kernel synchronization");

#else
        (void)level_id;
        for (auto* b : blocks) apply_derivatives_cuda(*b, dx_level);
#endif
    }

    template<class BlockPtr>
    void sync_curl_target_to_host_batched(const std::vector<BlockPtr>& blocks,
        int level_id)
    {
#ifdef IBLGF_COMPILE_CUDA
        if (blocks.empty()) return;
        ensure_cuda_stream();
        auto& tbl = get_or_build_level_pointer_tables(level_id, blocks);

        auto& c0 = (*blocks.front())(curl_target, 0);
        const auto ext = c0.real_block().extent();
        const int ex = ext[0], ey = ext[1], ez = ext[2];
        const auto lb = c0.lbuffer();
        const auto hb = c0.hbuffer();
        const int wx = static_cast<int>(lb[0] + hb[0]);
        const int wy = static_cast<int>(lb[1] + hb[1]);
        const int wz = static_cast<int>(lb[2] + hb[2]);
        const int mx = ex - 2 * wx;
        const int my = ey - 2 * wy;
        const int mz = ez - 2 * wz;
        const size_t n_halo = (mx <= 0 || my <= 0 || mz <= 0)
                                  ? static_cast<size_t>(ex) * ey * ez
                                  : static_cast<size_t>(2 * wz) * ex * ey +
                                        static_cast<size_t>(ez - 2 * wz) *
                                            (2 * wy) * ex +
                                        static_cast<size_t>(ez - 2 * wz) *
                                            (ey - 2 * wy) * (2 * wx);
        const size_t slab_items = n_halo * blocks.size();

        ensure_pack_capacity(d2h_pack_, d_d2h_pack_, d2h_pack_capacity_,
            3 * slab_items);
        const size_t total_points =
            static_cast<size_t>(blocks.size()) * ex * ey * ez;
        const int threads = 256;
        const int grid = static_cast<int>((total_points + threads - 1) / threads);
        operators_gpu_cuda::gather3_halo_to_slab<<<grid, threads, 0,
            compute_stream_>>>(tbl.d_curl_dst0, tbl.d_curl_dst1, tbl.d_curl_dst2,
            d_d2h_pack_, static_cast<int>(blocks.size()), ex, ey, ez, wx, wy,
            wz);
        operators_gpu_cuda::cuda_check(
            cudaGetLastError(), "Gather curl halo to slab");
        operators_gpu_cuda::cuda_check(cudaMemcpyAsync(d2h_pack_.data(),
                                             d_d2h_pack_,
                                             3 * slab_items * sizeof(float_type),
                                             cudaMemcpyDeviceToHost,
                                             compute_stream_),
            "Curl halo slab D2H");
        operators_gpu_cuda::cuda_check(
            cudaStreamSynchronize(compute_stream_),
            "Synchronize curl halo D2H");

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            auto& f0 = (*blocks[b])(curl_target, 0);
            auto& f1 = (*blocks[b])(curl_target, 1);
            auto& f2 = (*blocks[b])(curl_target, 2);
            for (int k = 0; k < ez; ++k)
            {
                for (int j = 0; j < ey; ++j)
                {
                    for (int i = 0; i < ex; ++i)
                    {
                        size_t hidx = 0, nh = 0;
                        if (!halo_index_host(
                                i, j, k, ex, ey, ez, wx, wy, wz, hidx, nh))
                            continue;
                        const size_t lin = static_cast<size_t>(i) +
                                           static_cast<size_t>(ex) *
                                               (static_cast<size_t>(j) +
                                                   static_cast<size_t>(ey) *
                                                       static_cast<size_t>(k));
                        f0.data()[lin] = d2h_pack_[0 * slab_items + b * n_halo + hidx];
                        f1.data()[lin] = d2h_pack_[1 * slab_items + b * n_halo + hidx];
                        f2.data()[lin] = d2h_pack_[2 * slab_items + b * n_halo + hidx];
                    }
                }
            }
        }
#else
        (void)level_id;
        for (auto* b : blocks) sync_curl_target_to_host(*b);
#endif
    }

    template<class BlockPtr>
    void sync_curl_target_to_device_batched(const std::vector<BlockPtr>& blocks,
        int level_id)
    {
#ifdef IBLGF_COMPILE_CUDA
        if (blocks.empty()) return;
        ensure_cuda_stream();
        auto& tbl = get_or_build_level_pointer_tables(level_id, blocks);

        auto& c0 = (*blocks.front())(curl_target, 0);
        const auto ext = c0.real_block().extent();
        const int ex = ext[0], ey = ext[1], ez = ext[2];
        const auto lb = c0.lbuffer();
        const auto hb = c0.hbuffer();
        const int wx = static_cast<int>(lb[0] + hb[0]);
        const int wy = static_cast<int>(lb[1] + hb[1]);
        const int wz = static_cast<int>(lb[2] + hb[2]);
        const int mx = ex - 2 * wx;
        const int my = ey - 2 * wy;
        const int mz = ez - 2 * wz;
        const size_t n_halo = (mx <= 0 || my <= 0 || mz <= 0)
                                  ? static_cast<size_t>(ex) * ey * ez
                                  : static_cast<size_t>(2 * wz) * ex * ey +
                                        static_cast<size_t>(ez - 2 * wz) *
                                            (2 * wy) * ex +
                                        static_cast<size_t>(ez - 2 * wz) *
                                            (ey - 2 * wy) * (2 * wx);
        const size_t slab_items = n_halo * blocks.size();
        ensure_pack_capacity(h2d_pack_, d_h2d_pack_, h2d_pack_capacity_,
            3 * slab_items);

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            auto& f0 = (*blocks[b])(curl_target, 0);
            auto& f1 = (*blocks[b])(curl_target, 1);
            auto& f2 = (*blocks[b])(curl_target, 2);
            for (int k = 0; k < ez; ++k)
            {
                for (int j = 0; j < ey; ++j)
                {
                    for (int i = 0; i < ex; ++i)
                    {
                        size_t hidx = 0, nh = 0;
                        if (!halo_index_host(
                                i, j, k, ex, ey, ez, wx, wy, wz, hidx, nh))
                            continue;
                        const size_t lin = static_cast<size_t>(i) +
                                           static_cast<size_t>(ex) *
                                               (static_cast<size_t>(j) +
                                                   static_cast<size_t>(ey) *
                                                       static_cast<size_t>(k));
                        h2d_pack_[0 * slab_items + b * n_halo + hidx] = f0.data()[lin];
                        h2d_pack_[1 * slab_items + b * n_halo + hidx] = f1.data()[lin];
                        h2d_pack_[2 * slab_items + b * n_halo + hidx] = f2.data()[lin];
                    }
                }
            }
        }

        operators_gpu_cuda::cuda_check(cudaMemcpyAsync(d_h2d_pack_,
                                             h2d_pack_.data(),
                                             3 * slab_items * sizeof(float_type),
                                             cudaMemcpyHostToDevice,
                                             compute_stream_),
            "Curl halo slab H2D");
        const size_t total_points =
            static_cast<size_t>(blocks.size()) * ex * ey * ez;
        const int threads = 256;
        const int grid = static_cast<int>((total_points + threads - 1) / threads);
        operators_gpu_cuda::scatter3_halo_from_slab<<<grid, threads, 0,
            compute_stream_>>>(d_h2d_pack_, tbl.d_curl_dst0, tbl.d_curl_dst1,
            tbl.d_curl_dst2, static_cast<int>(blocks.size()), ex, ey, ez, wx,
            wy, wz);
        operators_gpu_cuda::cuda_check(
            cudaGetLastError(), "Scatter curl halo from slab");
        operators_gpu_cuda::cuda_check(
            cudaStreamSynchronize(compute_stream_),
            "Synchronize curl halo H2D");
#else
        (void)level_id;
        (void)blocks;
#endif
    }

    template<class BlockPtr>
    void apply_nonlinear_cuda_batched(const std::vector<BlockPtr>& blocks,
        int level_id)
    {
#ifdef IBLGF_COMPILE_CUDA
        if (blocks.empty()) return;
        ensure_cuda_stream();

        int ex = 0, ey = 0, ez = 0, ox = 0, oy = 0, oz = 0;
        int nx = 0, ny = 0, nz = 0;
        if (!level_layout_is_uniform(blocks, ex, ey, ez, ox, oy, oz, nx, ny, nz))
        {
            for (auto* b : blocks) apply_nonlinear_cuda(*b);
            return;
        }

        static bool printed_level_nonlin = false;
        if (!printed_level_nonlin)
        {
            boost::mpi::communicator world;
            std::cout << "[operators_gpu][rank " << world.rank()
                      << "] Running level-batched nonlinear kernel"
                      << std::endl;
            printed_level_nonlin = true;
        }

        for (auto* block : blocks)
        {
            (void)(*block)(nonlinear_source, 0).device_ptr();
            (void)(*block)(nonlinear_source, 1).device_ptr();
            (void)(*block)(nonlinear_source, 2).device_ptr();
            (void)(*block)(nonlinear_target, 0).device_ptr();
            (void)(*block)(nonlinear_target, 1).device_ptr();
            (void)(*block)(nonlinear_target, 2).device_ptr();
            (void)(*block)(curl_target, 0).device_ptr();
            (void)(*block)(curl_target, 1).device_ptr();
            (void)(*block)(curl_target, 2).device_ptr();
        }
        auto& tbl = get_or_build_level_pointer_tables(level_id, blocks);

        const size_t n_per_block =
            (*blocks.front())(nonlinear_source, 0).data().size();
        const size_t slab_items = n_per_block * blocks.size();
        ensure_pack_capacity(h2d_pack_, d_h2d_pack_, h2d_pack_capacity_,
            3 * slab_items);

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            auto& face0 = (*blocks[b])(nonlinear_source, 0);
            auto& face1 = (*blocks[b])(nonlinear_source, 1);
            auto& face2 = (*blocks[b])(nonlinear_source, 2);
            auto copy_field = [&](size_t field_idx, auto& field) {
                std::copy(field.data().begin(), field.data().end(),
                    h2d_pack_.begin() + field_idx * slab_items +
                        b * n_per_block);
            };
            copy_field(0, face0);
            copy_field(1, face1);
            copy_field(2, face2);
        }
        operators_gpu_cuda::cuda_check(cudaMemcpyAsync(d_h2d_pack_,
                                             h2d_pack_.data(),
                                             3 * slab_items * sizeof(float_type),
                                             cudaMemcpyHostToDevice,
                                             compute_stream_),
            "Batched nonlinear slab H2D");

        const int threads = 256;
        const int grid_scatter = static_cast<int>(
            (slab_items + threads - 1) / threads);
        operators_gpu_cuda::scatter3_from_slab<<<grid_scatter, threads, 0,
            compute_stream_>>>(d_h2d_pack_, tbl.d_nonlin_src0,
            tbl.d_nonlin_src1, tbl.d_nonlin_src2, static_cast<int>(blocks.size()),
            n_per_block);
        operators_gpu_cuda::cuda_check(
            cudaGetLastError(), "Batched nonlinear slab scatter");

        const size_t interior_size = static_cast<size_t>(nx) * ny *
                                     static_cast<size_t>(nz);
        const size_t total_threads = interior_size * blocks.size();
        const int grid = static_cast<int>((total_threads + threads - 1) / threads);

        operators_gpu_cuda::nonlinear_kernel_batched<<<grid, threads, 0,
            compute_stream_>>>(tbl.d_nonlin_src0, tbl.d_nonlin_src1,
            tbl.d_nonlin_src2, tbl.d_curl_dst0, tbl.d_curl_dst1,
            tbl.d_curl_dst2, tbl.d_nonlin_dst0, tbl.d_nonlin_dst1,
            tbl.d_nonlin_dst2, static_cast<int>(blocks.size()), ex, ey, ox, oy,
            oz, nx, ny, nz);
        operators_gpu_cuda::cuda_check(
            cudaGetLastError(), "Batched nonlinear kernel launch");
        operators_gpu_cuda::cuda_check(
            cudaStreamSynchronize(compute_stream_),
            "Batched nonlinear kernel synchronization");

#else
        (void)level_id;
        for (auto* b : blocks) apply_nonlinear_cuda(*b);
#endif
    }

    template<class BlockPtr>
    void sync_curl_target_full_to_host_batched(const std::vector<BlockPtr>& blocks,
        int level_id)
    {
#ifdef IBLGF_COMPILE_CUDA
        if (blocks.empty()) return;
        ensure_cuda_stream();
        auto& tbl = get_or_build_level_pointer_tables(level_id, blocks);

        const size_t n_per_block = (*blocks.front())(curl_target, 0).data().size();
        const size_t slab_items = n_per_block * blocks.size();
        ensure_pack_capacity(
            d2h_pack_, d_d2h_pack_, d2h_pack_capacity_, 3 * slab_items);

        const int threads = 256;
        const int grid = static_cast<int>((slab_items + threads - 1) / threads);
        operators_gpu_cuda::gather3_to_slab<<<grid, threads, 0, compute_stream_>>>(
            tbl.d_curl_dst0, tbl.d_curl_dst1, tbl.d_curl_dst2, d_d2h_pack_,
            static_cast<int>(blocks.size()), n_per_block);
        operators_gpu_cuda::cuda_check(
            cudaGetLastError(), "Gather full curl target to slab");
        operators_gpu_cuda::cuda_check(cudaMemcpyAsync(d2h_pack_.data(),
                                             d_d2h_pack_,
                                             3 * slab_items * sizeof(float_type),
                                             cudaMemcpyDeviceToHost,
                                             compute_stream_),
            "Batched full curl slab D2H");
        operators_gpu_cuda::cuda_check(
            cudaStreamSynchronize(compute_stream_),
            "Synchronize batched full curl D2H");

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            auto& c0 = (*blocks[b])(curl_target, 0);
            auto& c1 = (*blocks[b])(curl_target, 1);
            auto& c2 = (*blocks[b])(curl_target, 2);
            auto copy_field = [&](size_t field_idx, auto& field) {
                std::copy(d2h_pack_.begin() + field_idx * slab_items +
                              b * n_per_block,
                    d2h_pack_.begin() + field_idx * slab_items +
                        (b + 1) * n_per_block,
                    field.data().begin());
            };
            copy_field(0, c0);
            copy_field(1, c1);
            copy_field(2, c2);
        }
#else
        (void)level_id;
        for (auto* b : blocks) sync_curl_target_to_host(*b);
#endif
    }

    template<class BlockPtr>
    void sync_targets_to_host_batched(const std::vector<BlockPtr>& blocks,
        int level_id)
    {
#ifdef IBLGF_COMPILE_CUDA
        if (blocks.empty()) return;
        ensure_cuda_stream();
        auto& tbl = get_or_build_level_pointer_tables(level_id, blocks);

        const size_t n_per_block = (*blocks.front())(lap_target, 0).data().size();
        const size_t slab_items = n_per_block * blocks.size();

        ensure_pack_capacity(d2h_pack_, d_d2h_pack_, d2h_pack_capacity_,
            8 * slab_items);
        const int threads = 256;
        const int grid = static_cast<int>((slab_items + threads - 1) / threads);
        operators_gpu_cuda::gather8_to_slab<<<grid, threads, 0, compute_stream_>>>(
            tbl.d_lap_dst, tbl.d_grad_dst0, tbl.d_grad_dst1, tbl.d_grad_dst2,
            tbl.d_div_dst, tbl.d_nonlin_dst0, tbl.d_nonlin_dst1,
            tbl.d_nonlin_dst2, d_d2h_pack_,
            static_cast<int>(blocks.size()), n_per_block);
        operators_gpu_cuda::cuda_check(
            cudaGetLastError(), "Gather final targets to slab");
        operators_gpu_cuda::cuda_check(cudaMemcpyAsync(d2h_pack_.data(),
                                             d_d2h_pack_,
                                             8 * slab_items * sizeof(float_type),
                                             cudaMemcpyDeviceToHost,
                                             compute_stream_),
            "Batched final slab D2H");
        operators_gpu_cuda::cuda_check(
            cudaStreamSynchronize(compute_stream_),
            "Synchronize batched final targets D2H");

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            auto& lap = (*blocks[b])(lap_target, 0);
            auto& g0 = (*blocks[b])(grad_target, 0);
            auto& g1 = (*blocks[b])(grad_target, 1);
            auto& g2 = (*blocks[b])(grad_target, 2);
            auto& div = (*blocks[b])(div_target, 0);
            auto& n0 = (*blocks[b])(nonlinear_target, 0);
            auto& n1 = (*blocks[b])(nonlinear_target, 1);
            auto& n2 = (*blocks[b])(nonlinear_target, 2);
            auto copy_field = [&](size_t field_idx, auto& field) {
                std::copy(d2h_pack_.begin() + field_idx * slab_items +
                              b * n_per_block,
                    d2h_pack_.begin() + field_idx * slab_items +
                        (b + 1) * n_per_block,
                    field.data().begin());
            };
            copy_field(0, lap);
            copy_field(1, g0);
            copy_field(2, g1);
            copy_field(3, g2);
            copy_field(4, div);
            copy_field(5, n0);
            copy_field(6, n1);
            copy_field(7, n2);
        }

#else
        (void)level_id;
        for (auto* b : blocks) sync_targets_to_host(*b);
#endif
    }

    template<class Block>
    void apply_derivatives_cuda(Block& block, float_type dx_level)
    {
#ifdef IBLGF_COMPILE_CUDA
        auto& lap_src = block(lap_source, 0);
        auto& lap_dst = block(lap_target, 0);

        auto& grad_src = block(grad_source, 0);
        auto& grad_dst0 = block(grad_target, 0);
        auto& grad_dst1 = block(grad_target, 1);
        auto& grad_dst2 = block(grad_target, 2);

        auto& div_src0 = block(div_source, 0);
        auto& div_src1 = block(div_source, 1);
        auto& div_src2 = block(div_source, 2);
        auto& div_dst = block(div_target, 0);

        auto& curl_src0 = block(curl_source, 0);
        auto& curl_src1 = block(curl_source, 1);
        auto& curl_src2 = block(curl_source, 2);
        auto& curl_dst0 = block(curl_target, 0);
        auto& curl_dst1 = block(curl_target, 1);
        auto& curl_dst2 = block(curl_target, 2);

        static bool printed_h2d = false;
        if (!printed_h2d)
        {
            boost::mpi::communicator world;
            std::cout << "[operators_gpu][rank " << world.rank()
                      << "] Copying derivative inputs Host->Device"
                      << std::endl;
            printed_h2d = true;
        }

        const auto n = lap_src.data().size();
        if (all_equal_sizes({n, grad_src.data().size(), div_src0.data().size(),
                div_src1.data().size(), div_src2.data().size(),
                curl_src0.data().size(), curl_src1.data().size(),
                curl_src2.data().size()}))
        {
            ensure_pack_capacity(h2d_pack_, d_h2d_pack_, h2d_pack_capacity_,
                8 * n);
            std::copy(lap_src.data().begin(), lap_src.data().end(),
                h2d_pack_.begin() + 0 * n);
            std::copy(grad_src.data().begin(), grad_src.data().end(),
                h2d_pack_.begin() + 1 * n);
            std::copy(div_src0.data().begin(), div_src0.data().end(),
                h2d_pack_.begin() + 2 * n);
            std::copy(div_src1.data().begin(), div_src1.data().end(),
                h2d_pack_.begin() + 3 * n);
            std::copy(div_src2.data().begin(), div_src2.data().end(),
                h2d_pack_.begin() + 4 * n);
            std::copy(curl_src0.data().begin(), curl_src0.data().end(),
                h2d_pack_.begin() + 5 * n);
            std::copy(curl_src1.data().begin(), curl_src1.data().end(),
                h2d_pack_.begin() + 6 * n);
            std::copy(curl_src2.data().begin(), curl_src2.data().end(),
                h2d_pack_.begin() + 7 * n);

            operators_gpu_cuda::cuda_check(cudaMemcpy(d_h2d_pack_,
                                             h2d_pack_.data(),
                                             8 * n * sizeof(float_type),
                                             cudaMemcpyHostToDevice),
                "Packed H2D derivative inputs");

            const int threads_1d = 256;
            const int blocks_1d = static_cast<int>((n + threads_1d - 1) /
                                                   threads_1d);
            operators_gpu_cuda::unpack8_kernel<<<blocks_1d, threads_1d>>>(
                d_h2d_pack_, lap_src.device_ptr(), grad_src.device_ptr(),
                div_src0.device_ptr(), div_src1.device_ptr(), div_src2.device_ptr(),
                curl_src0.device_ptr(), curl_src1.device_ptr(),
                curl_src2.device_ptr(), n);
            operators_gpu_cuda::cuda_check(
                cudaGetLastError(), "Unpack derivative inputs");
            operators_gpu_cuda::cuda_check(
                cudaDeviceSynchronize(), "Sync unpack derivative inputs");
        }
        else
        {
            lap_src.update_device(nullptr, true);
            grad_src.update_device(nullptr, true);
            div_src0.update_device(nullptr, true);
            div_src1.update_device(nullptr, true);
            div_src2.update_device(nullptr, true);
            curl_src0.update_device(nullptr, true);
            curl_src1.update_device(nullptr, true);
            curl_src2.update_device(nullptr, true);
        }

        const auto ext = lap_src.real_block().extent();
        const int  ex = ext[0];
        const int  ey = ext[1];
        const int  ez = ext[2];

        const auto interior = block.descriptor().extent();
        const int  nx = interior[0];
        const int  ny = interior[1];
        const int  nz = interior[2];

        const auto lb = lap_src.lbuffer();
        const int  ox = lb[0];
        const int  oy = lb[1];
        const int  oz = lb[2];

        const dim3 threads(8, 8, 8);
        const dim3 blocks((nx + threads.x - 1) / threads.x,
            (ny + threads.y - 1) / threads.y,
            (nz + threads.z - 1) / threads.z);

        operators_gpu_cuda::laplace_kernel<<<blocks, threads>>>(
            lap_src.device_ptr(),
            lap_dst.device_ptr(), ex, ey, ez, ox, oy, oz, nx, ny, nz,
            1.0 / (dx_level * dx_level));

        operators_gpu_cuda::gradient_kernel<<<blocks, threads>>>(
            grad_src.device_ptr(),
            grad_dst0.device_ptr(), grad_dst1.device_ptr(), grad_dst2.device_ptr(),
            ex, ey, ez, ox, oy, oz, nx, ny, nz, 1.0 / dx_level);

        operators_gpu_cuda::divergence_kernel<<<blocks, threads>>>(
            div_src0.device_ptr(),
            div_src1.device_ptr(), div_src2.device_ptr(), div_dst.device_ptr(), ex, ey,
            ez, ox, oy, oz, nx, ny, nz, 1.0 / dx_level);

        operators_gpu_cuda::curl_kernel<<<blocks, threads>>>(
            curl_src0.device_ptr(),
            curl_src1.device_ptr(), curl_src2.device_ptr(), curl_dst0.device_ptr(),
            curl_dst1.device_ptr(), curl_dst2.device_ptr(), ex, ey, ez, ox, oy, oz, nx,
            ny, nz, 1.0 / dx_level);

        operators_gpu_cuda::cuda_check(
            cudaGetLastError(), "Derivative kernel launch");
        operators_gpu_cuda::cuda_check(
            cudaDeviceSynchronize(), "Derivative kernel synchronization");
#else
        domain::Operator::laplace<lap_source_type, lap_target_type>(
            block, dx_level);
        domain::Operator::divergence<div_source_type, div_target_type>(
            block, dx_level);
        domain::Operator::curl<curl_source_type, curl_target_type>(
            block, dx_level);
        domain::Operator::gradient<grad_source_type, grad_target_type>(
            block, dx_level);
#endif
    }

    template<class Block>
    void sync_curl_target_to_host(Block& block)
    {
#ifdef IBLGF_COMPILE_CUDA
        auto& curl_dst0 = block(curl_target, 0);
        auto& curl_dst1 = block(curl_target, 1);
        auto& curl_dst2 = block(curl_target, 2);

        static bool printed_d2h_curl = false;
        if (!printed_d2h_curl)
        {
            boost::mpi::communicator world;
            std::cout << "[operators_gpu][rank " << world.rank()
                      << "] Copying curl_target Device->Host before MPI exchange"
                      << std::endl;
            printed_d2h_curl = true;
        }

        const auto n = curl_dst0.data().size();
        if (all_equal_sizes(
                {n, curl_dst1.data().size(), curl_dst2.data().size()}))
        {
            ensure_pack_capacity(d2h_pack_, d_d2h_pack_, d2h_pack_capacity_,
                3 * n);
            const int threads_1d = 256;
            const int blocks_1d = static_cast<int>((n + threads_1d - 1) /
                                                   threads_1d);
            operators_gpu_cuda::pack3_kernel<<<blocks_1d, threads_1d>>>(
                curl_dst0.device_ptr(), curl_dst1.device_ptr(),
                curl_dst2.device_ptr(), d_d2h_pack_, n);
            operators_gpu_cuda::cuda_check(
                cudaGetLastError(), "Pack curl targets");
            operators_gpu_cuda::cuda_check(
                cudaDeviceSynchronize(), "Sync pack curl targets");

            operators_gpu_cuda::cuda_check(cudaMemcpy(d2h_pack_.data(),
                                             d_d2h_pack_,
                                             3 * n * sizeof(float_type),
                                             cudaMemcpyDeviceToHost),
                "Packed D2H curl targets");

            std::copy(d2h_pack_.begin() + 0 * n, d2h_pack_.begin() + 1 * n,
                curl_dst0.data().begin());
            std::copy(d2h_pack_.begin() + 1 * n, d2h_pack_.begin() + 2 * n,
                curl_dst1.data().begin());
            std::copy(d2h_pack_.begin() + 2 * n, d2h_pack_.begin() + 3 * n,
                curl_dst2.data().begin());
        }
        else
        {
            auto copy_back = [](auto& f, const char* context) {
                operators_gpu_cuda::cuda_check(cudaMemcpy(f.data().data(),
                                                 f.device_ptr(),
                                                 f.data().size() * sizeof(float_type),
                                                 cudaMemcpyDeviceToHost),
                    context);
            };
            copy_back(curl_dst0, "Copy curl target x back to host");
            copy_back(curl_dst1, "Copy curl target y back to host");
            copy_back(curl_dst2, "Copy curl target z back to host");
        }
#else
        (void)block;
#endif
    }

    template<class Block>
    void apply_nonlinear_cuda(Block& block)
    {
#ifdef IBLGF_COMPILE_CUDA
        auto& face0 = block(nonlinear_source, 0);
        auto& face1 = block(nonlinear_source, 1);
        auto& face2 = block(nonlinear_source, 2);

        auto& edge0 = block(curl_target, 0);
        auto& edge1 = block(curl_target, 1);
        auto& edge2 = block(curl_target, 2);

        auto& dst0 = block(nonlinear_target, 0);
        auto& dst1 = block(nonlinear_target, 1);
        auto& dst2 = block(nonlinear_target, 2);

        static bool printed_h2d_nonlin = false;
        if (!printed_h2d_nonlin)
        {
            boost::mpi::communicator world;
            std::cout << "[operators_gpu][rank " << world.rank()
                      << "] Copying nonlinear inputs Host->Device" << std::endl;
            printed_h2d_nonlin = true;
        }

        const auto n = face0.data().size();
        if (all_equal_sizes({n, face1.data().size(), face2.data().size(),
                edge0.data().size(), edge1.data().size(),
                edge2.data().size()}))
        {
            ensure_pack_capacity(h2d_pack_, d_h2d_pack_, h2d_pack_capacity_,
                6 * n);
            std::copy(face0.data().begin(), face0.data().end(),
                h2d_pack_.begin() + 0 * n);
            std::copy(face1.data().begin(), face1.data().end(),
                h2d_pack_.begin() + 1 * n);
            std::copy(face2.data().begin(), face2.data().end(),
                h2d_pack_.begin() + 2 * n);
            std::copy(edge0.data().begin(), edge0.data().end(),
                h2d_pack_.begin() + 3 * n);
            std::copy(edge1.data().begin(), edge1.data().end(),
                h2d_pack_.begin() + 4 * n);
            std::copy(edge2.data().begin(), edge2.data().end(),
                h2d_pack_.begin() + 5 * n);

            operators_gpu_cuda::cuda_check(cudaMemcpy(d_h2d_pack_,
                                             h2d_pack_.data(),
                                             6 * n * sizeof(float_type),
                                             cudaMemcpyHostToDevice),
                "Packed H2D nonlinear inputs");
            const int threads_1d = 256;
            const int blocks_1d = static_cast<int>((n + threads_1d - 1) /
                                                   threads_1d);
            operators_gpu_cuda::unpack6_kernel<<<blocks_1d, threads_1d>>>(
                d_h2d_pack_, face0.device_ptr(), face1.device_ptr(),
                face2.device_ptr(), edge0.device_ptr(), edge1.device_ptr(),
                edge2.device_ptr(), n);
            operators_gpu_cuda::cuda_check(
                cudaGetLastError(), "Unpack nonlinear inputs");
            operators_gpu_cuda::cuda_check(
                cudaDeviceSynchronize(), "Sync unpack nonlinear inputs");
        }
        else
        {
            face0.update_device(nullptr, true);
            face1.update_device(nullptr, true);
            face2.update_device(nullptr, true);
            edge0.update_device(nullptr, true);
            edge1.update_device(nullptr, true);
            edge2.update_device(nullptr, true);
        }

        const auto ext = face0.real_block().extent();
        const int  ex = ext[0];
        const int  ey = ext[1];
        const int  ez = ext[2];

        const auto interior = block.descriptor().extent();
        const int  nx = interior[0];
        const int  ny = interior[1];
        const int  nz = interior[2];

        const auto lb = face0.lbuffer();
        const int  ox = lb[0];
        const int  oy = lb[1];
        const int  oz = lb[2];

        const dim3 threads(8, 8, 8);
        const dim3 blocks((nx + threads.x - 1) / threads.x,
            (ny + threads.y - 1) / threads.y,
            (nz + threads.z - 1) / threads.z);

        operators_gpu_cuda::nonlinear_kernel<<<blocks, threads>>>(face0.device_ptr(),
            face1.device_ptr(), face2.device_ptr(), edge0.device_ptr(),
            edge1.device_ptr(), edge2.device_ptr(), dst0.device_ptr(),
            dst1.device_ptr(), dst2.device_ptr(), ex, ey, ez, ox, oy, oz, nx,
            ny, nz);

        operators_gpu_cuda::cuda_check(
            cudaGetLastError(), "Nonlinear kernel launch");
        operators_gpu_cuda::cuda_check(
            cudaDeviceSynchronize(), "Nonlinear kernel synchronization");
#else
        domain::Operator::nonlinear<nonlinear_source_type, curl_target_type,
            nonlinear_target_type>(block);
#endif
    }

    template<class Block>
    void sync_targets_to_host(Block& block)
    {
#ifdef IBLGF_COMPILE_CUDA
        auto& lap_dst = block(lap_target, 0);
        auto& grad_dst0 = block(grad_target, 0);
        auto& grad_dst1 = block(grad_target, 1);
        auto& grad_dst2 = block(grad_target, 2);
        auto& div_dst = block(div_target, 0);
        auto& nonlin_dst0 = block(nonlinear_target, 0);
        auto& nonlin_dst1 = block(nonlinear_target, 1);
        auto& nonlin_dst2 = block(nonlinear_target, 2);

        static bool printed_d2h_final = false;
        if (!printed_d2h_final)
        {
            boost::mpi::communicator world;
            std::cout << "[operators_gpu][rank " << world.rank()
                      << "] Copying final operator targets Device->Host"
                      << std::endl;
            printed_d2h_final = true;
        }

        const auto n = lap_dst.data().size();
        if (all_equal_sizes({n, grad_dst0.data().size(), grad_dst1.data().size(),
                grad_dst2.data().size(), div_dst.data().size(),
                nonlin_dst0.data().size(), nonlin_dst1.data().size(),
                nonlin_dst2.data().size()}))
        {
            ensure_pack_capacity(d2h_pack_, d_d2h_pack_, d2h_pack_capacity_,
                8 * n);
            const int threads_1d = 256;
            const int blocks_1d = static_cast<int>((n + threads_1d - 1) /
                                                   threads_1d);
            operators_gpu_cuda::pack8_kernel<<<blocks_1d, threads_1d>>>(
                lap_dst.device_ptr(), grad_dst0.device_ptr(),
                grad_dst1.device_ptr(), grad_dst2.device_ptr(),
                div_dst.device_ptr(), nonlin_dst0.device_ptr(),
                nonlin_dst1.device_ptr(), nonlin_dst2.device_ptr(),
                d_d2h_pack_, n);
            operators_gpu_cuda::cuda_check(
                cudaGetLastError(), "Pack final targets");
            operators_gpu_cuda::cuda_check(
                cudaDeviceSynchronize(), "Sync pack final targets");
            operators_gpu_cuda::cuda_check(cudaMemcpy(d2h_pack_.data(),
                                             d_d2h_pack_,
                                             8 * n * sizeof(float_type),
                                             cudaMemcpyDeviceToHost),
                "Packed D2H final targets");

            std::copy(d2h_pack_.begin() + 0 * n, d2h_pack_.begin() + 1 * n,
                lap_dst.data().begin());
            std::copy(d2h_pack_.begin() + 1 * n, d2h_pack_.begin() + 2 * n,
                grad_dst0.data().begin());
            std::copy(d2h_pack_.begin() + 2 * n, d2h_pack_.begin() + 3 * n,
                grad_dst1.data().begin());
            std::copy(d2h_pack_.begin() + 3 * n, d2h_pack_.begin() + 4 * n,
                grad_dst2.data().begin());
            std::copy(d2h_pack_.begin() + 4 * n, d2h_pack_.begin() + 5 * n,
                div_dst.data().begin());
            std::copy(d2h_pack_.begin() + 5 * n, d2h_pack_.begin() + 6 * n,
                nonlin_dst0.data().begin());
            std::copy(d2h_pack_.begin() + 6 * n, d2h_pack_.begin() + 7 * n,
                nonlin_dst1.data().begin());
            std::copy(d2h_pack_.begin() + 7 * n, d2h_pack_.begin() + 8 * n,
                nonlin_dst2.data().begin());
        }
        else
        {
            auto copy_back = [](auto& f, const char* context) {
                operators_gpu_cuda::cuda_check(cudaMemcpy(f.data().data(),
                                                 f.device_ptr(),
                                                 f.data().size() * sizeof(float_type),
                                                 cudaMemcpyDeviceToHost),
                    context);
            };

            copy_back(lap_dst, "Copy laplace target back to host");
            copy_back(grad_dst0, "Copy gradient target x back to host");
            copy_back(grad_dst1, "Copy gradient target y back to host");
            copy_back(grad_dst2, "Copy gradient target z back to host");
            copy_back(div_dst, "Copy divergence target back to host");
            copy_back(nonlin_dst0, "Copy nonlinear target x back to host");
            copy_back(nonlin_dst1, "Copy nonlinear target y back to host");
            copy_back(nonlin_dst2, "Copy nonlinear target z back to host");
        }
#else
        (void)block;
#endif
    }

#ifdef IBLGF_COMPILE_CUDA
    ~OperatorTest()
    {
        for (auto& kv : level_pointer_tables_) free_level_pointer_tables(kv.second);
        if (compute_stream_) cudaStreamDestroy(compute_stream_);
        if (d_h2d_pack_) cudaFree(d_h2d_pack_);
        if (d_d2h_pack_) cudaFree(d_d2h_pack_);
    }
#endif

  private:
    bool all_equal_sizes(std::initializer_list<std::size_t> sizes) const
    {
        auto it = sizes.begin();
        if (it == sizes.end()) return true;
        const auto first = *it;
        ++it;
        for (; it != sizes.end(); ++it)
        {
            if (*it != first) return false;
        }
        return true;
    }

#ifdef IBLGF_COMPILE_CUDA
    void ensure_pack_capacity(std::vector<float_type>& host_buf,
        float_type*& device_buf, std::size_t& capacity, std::size_t needed)
    {
        if (capacity >= needed) return;
        host_buf.resize(needed);
        if (device_buf) cudaFree(device_buf);
        operators_gpu_cuda::cuda_check(
            cudaMalloc(&device_buf, needed * sizeof(float_type)),
            "Allocate packed transfer buffer");
        capacity = needed;
    }
#endif

    boost::mpi::communicator client_comm_;
    float_type               eps_grad_ = 1.0e6;
    int                      nLevels_ = 0;
    int                      global_refinement_;
    float_type               a_ = 100.0;

#ifdef IBLGF_COMPILE_CUDA
    std::vector<float_type> h2d_pack_;
    std::vector<float_type> d2h_pack_;
    float_type*             d_h2d_pack_ = nullptr;
    float_type*             d_d2h_pack_ = nullptr;
    std::size_t             h2d_pack_capacity_ = 0;
    std::size_t             d2h_pack_capacity_ = 0;
    cudaStream_t            compute_stream_ = nullptr;
    std::unordered_map<int, LevelPointerTables> level_pointer_tables_;
#endif
};

} // namespace iblgf

#endif // IBLGF_INCLUDED_POISSON_HPP
