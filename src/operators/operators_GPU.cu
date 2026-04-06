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

#include <cuda_runtime.h>
#include <cmath>
#include <iblgf/operators/operators_GPU.hpp>

namespace iblgf
{
namespace gpu
{
namespace ops
{
namespace
{
__device__ inline std::size_t index_3d(int x, int y, int z,
    const int* base, const int* extent)
{
    return static_cast<std::size_t>(x - base[0]) +
           static_cast<std::size_t>(extent[0]) *
               static_cast<std::size_t>(y - base[1]) +
           static_cast<std::size_t>(extent[0]) *
               static_cast<std::size_t>(extent[1]) *
               static_cast<std::size_t>(z - base[2]);
}

__device__ inline std::size_t index_2d(int x, int y,
    const int* base, const int* extent)
{
    return static_cast<std::size_t>(x - base[0]) +
           static_cast<std::size_t>(extent[0]) *
               static_cast<std::size_t>(y - base[1]);
}

__device__ inline void idx_to_ijk(std::size_t idx, int dim,
    const int* ext, int& i, int& j, int& k)
{
    if (dim == 3)
    {
        const std::size_t ex = static_cast<std::size_t>(ext[0]);
        const std::size_t ey = static_cast<std::size_t>(ext[1]);
        i = static_cast<int>(idx % ex);
        j = static_cast<int>((idx / ex) % ey);
        k = static_cast<int>(idx / (ex * ey));
    }
    else
    {
        const std::size_t ex = static_cast<std::size_t>(ext[0]);
        i = static_cast<int>(idx % ex);
        j = static_cast<int>(idx / ex);
        k = 0;
    }
}

__global__ void laplace_kernel(const types::float_type* src,
    types::float_type* dst, block_desc desc, types::float_type fac)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (idx >= total) return;

    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;

    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        const auto xm = index_3d(x - 1, y, z, desc.field_base, desc.field_extent);
        const auto xp = index_3d(x + 1, y, z, desc.field_base, desc.field_extent);
        const auto ym = index_3d(x, y - 1, z, desc.field_base, desc.field_extent);
        const auto yp = index_3d(x, y + 1, z, desc.field_base, desc.field_extent);
        const auto zm = index_3d(x, y, z - 1, desc.field_base, desc.field_extent);
        const auto zp = index_3d(x, y, z + 1, desc.field_base, desc.field_extent);
        dst[c] = fac * (-6.0 * src[c] + src[zm] + src[zp] + src[ym] +
                        src[yp] + src[xm] + src[xp]);
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        const auto xm = index_2d(x - 1, y, desc.field_base, desc.field_extent);
        const auto xp = index_2d(x + 1, y, desc.field_base, desc.field_extent);
        const auto ym = index_2d(x, y - 1, desc.field_base, desc.field_extent);
        const auto yp = index_2d(x, y + 1, desc.field_base, desc.field_extent);
        dst[c] = fac * (-4.0 * src[c] + src[ym] + src[yp] + src[xm] + src[xp]);
    }
}

__global__ void gradient_kernel(const types::float_type* src,
    types::float_type* dst0, types::float_type* dst1,
    types::float_type* dst2, block_desc desc, types::float_type fac)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (idx >= total) return;

    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;

    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        const auto xm = index_3d(x - 1, y, z, desc.field_base, desc.field_extent);
        const auto ym = index_3d(x, y - 1, z, desc.field_base, desc.field_extent);
        const auto zm = index_3d(x, y, z - 1, desc.field_base, desc.field_extent);
        dst0[c] = fac * (src[c] - src[xm]);
        dst1[c] = fac * (src[c] - src[ym]);
        dst2[c] = fac * (src[c] - src[zm]);
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        const auto xm = index_2d(x - 1, y, desc.field_base, desc.field_extent);
        const auto ym = index_2d(x, y - 1, desc.field_base, desc.field_extent);
        dst0[c] = fac * (src[c] - src[xm]);
        dst1[c] = fac * (src[c] - src[ym]);
    }
}

__global__ void divergence_kernel(const types::float_type* src0,
    const types::float_type* src1, const types::float_type* src2,
    types::float_type* dst, block_desc desc, types::float_type fac)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (idx >= total) return;

    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;

    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        const auto xp = index_3d(x + 1, y, z, desc.field_base, desc.field_extent);
        const auto yp = index_3d(x, y + 1, z, desc.field_base, desc.field_extent);
        const auto zp = index_3d(x, y, z + 1, desc.field_base, desc.field_extent);
        dst[c] = fac * (-src0[c] - src1[c] - src2[c] + src0[xp] + src1[yp] +
                        src2[zp]);
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        const auto xp = index_2d(x + 1, y, desc.field_base, desc.field_extent);
        const auto yp = index_2d(x, y + 1, desc.field_base, desc.field_extent);
        dst[c] = fac * (-src0[c] - src1[c] + src0[xp] + src1[yp]);
    }
}

__global__ void curl_kernel(const types::float_type* src0,
    const types::float_type* src1, const types::float_type* src2,
    types::float_type* dst0, types::float_type* dst1,
    types::float_type* dst2, block_desc desc, types::float_type fac)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (idx >= total) return;

    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;

    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        const auto ym = index_3d(x, y - 1, z, desc.field_base, desc.field_extent);
        const auto zm = index_3d(x, y, z - 1, desc.field_base, desc.field_extent);
        const auto xm = index_3d(x - 1, y, z, desc.field_base, desc.field_extent);

        dst0[c] = fac * (src2[c] - src2[ym] - src1[c] + src1[zm]);
        dst1[c] = fac * (src0[c] - src0[zm] - src2[c] + src2[xm]);
        dst2[c] = fac * (src1[c] - src1[xm] - src0[c] + src0[ym]);
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        const auto xm = index_2d(x - 1, y, desc.field_base, desc.field_extent);
        const auto ym = index_2d(x, y - 1, desc.field_base, desc.field_extent);
        dst0[c] = fac * (src1[c] - src1[xm] - src0[c] + src0[ym]);
    }
}

__global__ void curl_transpose_kernel(const types::float_type* src0,
    const types::float_type* src1, const types::float_type* src2,
    types::float_type* dst0, types::float_type* dst1,
    types::float_type* dst2, block_desc desc, types::float_type fac)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (idx >= total) return;

    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;

    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        const auto zp = index_3d(x, y, z + 1, desc.field_base, desc.field_extent);
        const auto yp = index_3d(x, y + 1, z, desc.field_base, desc.field_extent);
        const auto xp = index_3d(x + 1, y, z, desc.field_base, desc.field_extent);

        dst0[c] = fac * (src1[c] - src1[zp] + src2[yp] - src2[c]);
        dst1[c] = fac * (src2[c] - src2[xp] + src0[zp] - src0[c]);
        dst2[c] = fac * (src0[c] - src0[yp] + src1[xp] - src1[c]);
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        const auto yp = index_2d(x, y + 1, desc.field_base, desc.field_extent);
        const auto xp = index_2d(x + 1, y, desc.field_base, desc.field_extent);
        dst0[c] = fac * (-src0[c] + src0[yp]);
        dst1[c] = fac * (-src0[xp] + src0[c]);
    }
}

__global__ void nonlinear_kernel(const types::float_type* f0,
    const types::float_type* f1, const types::float_type* f2,
    const types::float_type* e0, const types::float_type* e1,
    const types::float_type* e2,
    types::float_type* d0, types::float_type* d1, types::float_type* d2,
    block_desc desc)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (idx >= total) return;

    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;

    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        const auto xm = index_3d(x - 1, y, z, desc.field_base, desc.field_extent);
        const auto ym = index_3d(x, y - 1, z, desc.field_base, desc.field_extent);
        const auto zm = index_3d(x, y, z - 1, desc.field_base, desc.field_extent);
        const auto xp = index_3d(x + 1, y, z, desc.field_base, desc.field_extent);
        const auto yp = index_3d(x, y + 1, z, desc.field_base, desc.field_extent);
        const auto zp = index_3d(x, y, z + 1, desc.field_base, desc.field_extent);

        d0[c] = 0.25 * (+e1[c] * (f2[c] + f2[xm]) +
                        e1[zp] * (f2[zp] + f2[index_3d(x - 1, y, z + 1, desc.field_base, desc.field_extent)]) -
                        e2[c] * (f1[c] + f1[xm]) -
                        e2[yp] * (f1[yp] + f1[index_3d(x - 1, y + 1, z, desc.field_base, desc.field_extent)]));

        d1[c] = 0.25 * (+e2[c] * (f0[c] + f0[ym]) +
                        e2[xp] * (f0[xp] + f0[index_3d(x + 1, y - 1, z, desc.field_base, desc.field_extent)]) -
                        e0[c] * (f2[c] + f2[ym]) -
                        e0[zp] * (f2[zp] + f2[index_3d(x, y - 1, z + 1, desc.field_base, desc.field_extent)]));

        d2[c] = 0.25 * (+e0[c] * (f1[c] + f1[zm]) +
                        e0[yp] * (f1[yp] + f1[index_3d(x, y + 1, z - 1, desc.field_base, desc.field_extent)]) -
                        e1[c] * (f0[c] + f0[zm]) -
                        e1[xp] * (f0[xp] + f0[index_3d(x + 1, y, z - 1, desc.field_base, desc.field_extent)]));
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        const auto xm = index_2d(x - 1, y, desc.field_base, desc.field_extent);
        const auto ym = index_2d(x, y - 1, desc.field_base, desc.field_extent);
        const auto xp = index_2d(x + 1, y, desc.field_base, desc.field_extent);
        const auto yp = index_2d(x, y + 1, desc.field_base, desc.field_extent);

        d0[c] = 0.25 * (-e0[c] * (f1[c] + f1[xm]) -
                        e0[yp] * (f1[yp] + f1[index_2d(x - 1, y + 1, desc.field_base, desc.field_extent)]));
        d1[c] = 0.25 * (+e0[c] * (f0[c] + f0[ym]) +
                        e0[xp] * (f0[xp] + f0[index_2d(x + 1, y - 1, desc.field_base, desc.field_extent)]));
    }
}

__global__ void smooth2zero_kernel(types::float_type* field,
    block_desc desc, std::size_t ngb_idx)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (idx >= total) return;

    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;

    const types::float_type fac = 10.0;
    const types::float_type shift = 0.2;
    const types::float_type c0 = 1.0 - (0.5 + 0.5 * tanh(fac * (1.0 - shift)));
    const auto f = [&](types::float_type v) {
        return (0.5 + 0.5 * tanh(fac * (v - shift))) + c0;
    };

    const int dim = 3;
    const std::size_t xx = ngb_idx % dim;
    const std::size_t yy = (ngb_idx / dim) % dim;
    const std::size_t zz = (ngb_idx / dim / dim) % dim;

    types::float_type square = 0.0;
    types::float_type c = 0.0;

    const types::float_type pct_x =
        (static_cast<types::float_type>(x - desc.block_base[0])) /
        static_cast<types::float_type>(desc.block_extent[0] - 1);
    const types::float_type pct_y =
        (static_cast<types::float_type>(y - desc.block_base[1])) /
        static_cast<types::float_type>(desc.block_extent[1] - 1);
    const types::float_type pct_z =
        (static_cast<types::float_type>(z - desc.block_base[2])) /
        static_cast<types::float_type>(desc.block_extent[2] - 1);

    if (desc.dim == 3)
    {
        if (zz == 0)
        {
            square = fmax(square, f(pct_z));
            c += 1.0;
        }
        else if (zz == (dim - 1))
        {
            square = fmax(square, f(1.0 - pct_z));
            c += 1.0;
        }
    }

    if (yy == 0)
    {
        square = fmax(square, f(pct_y));
        c += 1.0;
    }
    else if (yy == (dim - 1))
    {
        square = fmax(square, f(1.0 - pct_y));
        c += 1.0;
    }

    if (xx == 0)
    {
        square = fmax(square, f(pct_x));
        c += 1.0;
    }
    else if (xx == (dim - 1))
    {
        square = fmax(square, f(1.0 - pct_x));
        c += 1.0;
    }

    if (c > 0.0)
    {
        const auto idxf = (desc.dim == 3)
            ? index_3d(x, y, z, desc.field_base, desc.field_extent)
            : index_2d(x, y, desc.field_base, desc.field_extent);
        field[idxf] = field[idxf] * square;
    }
}

__global__ void zero_kernel(types::float_type* field, std::size_t n)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    field[idx] = 0.0;
}
} // namespace

void laplace_device(const types::float_type* src, types::float_type* dst,
    const block_desc& desc, types::float_type dx)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const types::float_type fac = 1.0 / (dx * dx);
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    laplace_kernel<<<grid, block>>>(src, dst, desc, fac);
}

void gradient_device(const types::float_type* src,
    types::float_type* dst0, types::float_type* dst1,
    types::float_type* dst2, const block_desc& desc,
    types::float_type dx, int dim)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const types::float_type fac = 1.0 / dx;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    block_desc d = desc;
    d.dim = dim;
    gradient_kernel<<<grid, block>>>(src, dst0, dst1, dst2, d, fac);
}

void divergence_device(const types::float_type* src0,
    const types::float_type* src1, const types::float_type* src2,
    types::float_type* dst, const block_desc& desc,
    types::float_type dx, int dim)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const types::float_type fac = 1.0 / dx;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    block_desc d = desc;
    d.dim = dim;
    divergence_kernel<<<grid, block>>>(src0, src1, src2, dst, d, fac);
}

void curl_device(const types::float_type* src0,
    const types::float_type* src1, const types::float_type* src2,
    types::float_type* dst0, types::float_type* dst1,
    types::float_type* dst2, const block_desc& desc,
    types::float_type dx, int dim)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const types::float_type fac = 1.0 / dx;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    block_desc d = desc;
    d.dim = dim;
    curl_kernel<<<grid, block>>>(src0, src1, src2, dst0, dst1, dst2, d, fac);
}

void curl_transpose_device(const types::float_type* src0,
    const types::float_type* src1, const types::float_type* src2,
    types::float_type* dst0, types::float_type* dst1,
    types::float_type* dst2, const block_desc& desc,
    types::float_type dx, types::float_type scale, int dim)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const types::float_type fac = (1.0 / dx) * scale;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    block_desc d = desc;
    d.dim = dim;
    curl_transpose_kernel<<<grid, block>>>(src0, src1, src2, dst0, dst1, dst2,
        d, fac);
}

void nonlinear_device(const types::float_type* face0,
    const types::float_type* face1, const types::float_type* face2,
    const types::float_type* edge0, const types::float_type* edge1,
    const types::float_type* edge2, types::float_type* dst0,
    types::float_type* dst1, types::float_type* dst2,
    const block_desc& desc, int dim)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    block_desc d = desc;
    d.dim = dim;
    nonlinear_kernel<<<grid, block>>>(face0, face1, face2, edge0, edge1, edge2,
        dst0, dst1, dst2, d);
}

void smooth2zero_device(types::float_type* field, const block_desc& desc,
    std::size_t ngb_idx)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    smooth2zero_kernel<<<grid, block>>>(field, desc, ngb_idx);
}

void zero_device(types::float_type* field, std::size_t count)
{
    if (count == 0) return;
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    zero_kernel<<<grid, block>>>(field, count);
}

__global__ void zero_boundary_kernel(types::float_type* field, block_desc desc,
    int width)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int i_max = desc.block_extent[0] - width;
    const int j_max = desc.block_extent[1] - width;
    const int k_max = desc.block_extent[2] - width;

    const bool on_boundary =
        (i < width) || (i >= i_max) || (j < width) || (j >= j_max) ||
        (desc.dim == 3 && ((k < width) || (k >= k_max)));
    if (!on_boundary) return;

    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;
    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        field[c] = 0.0;
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        field[c] = 0.0;
    }
}

void zero_boundary_device(types::float_type* field, const block_desc& desc,
    int width)
{
    if (width <= 0) return;
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    zero_boundary_kernel<<<grid, block>>>(field, desc, width);
}

__global__ void set_constant_kernel(types::float_type* field, block_desc desc,
    types::float_type value)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;
    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        field[c] = value;
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        field[c] = value;
    }
}

void set_constant_field_device(types::float_type* field, const block_desc& desc,
    types::float_type value)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    set_constant_kernel<<<grid, block>>>(field, desc, value);
}

__global__ void copy_field_kernel(const types::float_type* src,
    types::float_type* dst, block_desc desc, bool with_buffer)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    if (!with_buffer)
    {
        if (i == 0 || j == 0) return;
        if (i == desc.block_extent[0] - 1 || j == desc.block_extent[1] - 1)
            return;
        if (desc.dim == 3)
        {
            if (k == 0 || k == desc.block_extent[2] - 1) return;
        }
    }

    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;

    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        dst[c] = src[c];
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        dst[c] = src[c];
    }
}

void copy_field_device(const types::float_type* src,
    types::float_type* dst, const block_desc& desc, bool with_buffer)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    copy_field_kernel<<<grid, block>>>(src, dst, desc, with_buffer);
}

__global__ void scale_field_kernel(types::float_type* field, block_desc desc,
    types::float_type scale)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;
    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        field[c] *= scale;
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        field[c] *= scale;
    }
}

void scale_field_device(types::float_type* field, const block_desc& desc,
    types::float_type scale)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    scale_field_kernel<<<grid, block>>>(field, desc, scale);
}

__global__ void axpy_field_kernel(const types::float_type* src,
    types::float_type* dst, block_desc desc, types::float_type scale)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;
    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        dst[c] += src[c] * scale;
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        dst[c] += src[c] * scale;
    }
}

void axpy_field_device(const types::float_type* src,
    types::float_type* dst, const block_desc& desc, types::float_type scale)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    axpy_field_kernel<<<grid, block>>>(src, dst, desc, scale);
}

__global__ void product_field_kernel(const types::float_type* a,
    const types::float_type* b, types::float_type* dst, block_desc desc)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;
    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        dst[c] = a[c] * b[c];
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        dst[c] = a[c] * b[c];
    }
}

void product_field_device(const types::float_type* a,
    const types::float_type* b, types::float_type* dst,
    const block_desc& desc)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    product_field_kernel<<<grid, block>>>(a, b, dst, desc);
}

__global__ void invert_field_kernel(types::float_type* field, block_desc desc,
    types::float_type eps)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;
    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        const auto v = field[c];
        if (fabs(v) > eps) field[c] = 1.0 / v;
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        const auto v = field[c];
        if (fabs(v) > eps) field[c] = 1.0 / v;
    }
}

void invert_field_device(types::float_type* field, const block_desc& desc,
    types::float_type eps)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    invert_field_kernel<<<grid, block>>>(field, desc, eps);
}

__global__ void add_body_force_kernel(types::float_type* field, block_desc desc,
    types::float_type dx, types::float_type scale, types::float_type b_f_mag,
    types::float_type b_f_eps)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int i, j, k;
    idx_to_ijk(idx, desc.dim, desc.block_extent, i, j, k);
    const int x = desc.block_base[0] + i;
    const int y = desc.block_base[1] + j;
    const int z = desc.block_base[2] + k;
    const types::float_type yf = static_cast<types::float_type>(y) * dx;
    const types::float_type denom = yf * yf + b_f_eps;
    const types::float_type add = -scale * b_f_mag * yf / denom;
    if (desc.dim == 3)
    {
        const auto c = index_3d(x, y, z, desc.field_base, desc.field_extent);
        field[c] += add;
    }
    else
    {
        const auto c = index_2d(x, y, desc.field_base, desc.field_extent);
        field[c] += add;
    }
}

void add_body_force_device(types::float_type* field, const block_desc& desc,
    types::float_type dx, types::float_type scale,
    types::float_type b_f_mag, types::float_type b_f_eps)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.block_extent[0]) *
        static_cast<std::size_t>(desc.block_extent[1]) *
        static_cast<std::size_t>(desc.block_extent[2]);
    if (total == 0) return;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    add_body_force_kernel<<<grid, block>>>(field, desc, dx, scale, b_f_mag,
        b_f_eps);
}

} // namespace ops
} // namespace gpu
} // namespace iblgf
