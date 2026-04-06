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
#include <iblgf/interpolation/interpolation_GPU.hpp>

namespace iblgf
{
namespace gpu
{
namespace interp
{
namespace
{
__device__ inline std::size_t idx3(int x, int y, int z,
    const field_desc& d)
{
    return static_cast<std::size_t>(x - d.base[0]) +
           static_cast<std::size_t>(d.ext[0]) *
               static_cast<std::size_t>(y - d.base[1]) +
           static_cast<std::size_t>(d.ext[0]) *
               static_cast<std::size_t>(d.ext[1]) *
               static_cast<std::size_t>(z - d.base[2]);
}

__device__ inline std::size_t idx2(int x, int y,
    const field_desc& d)
{
    return static_cast<std::size_t>(x - d.base[0]) +
           static_cast<std::size_t>(d.ext[0]) *
               static_cast<std::size_t>(y - d.base[1]);
}

__global__ void intrp_kernel(const types::float_type* parent,
    types::float_type* child, const types::float_type* Mx,
    const types::float_type* My, const types::float_type* Mz,
    int nb, field_desc pd, field_desc cd, int dim)
{
    const std::size_t total =
        static_cast<std::size_t>(cd.ext[0]) *
        static_cast<std::size_t>(cd.ext[1]) *
        static_cast<std::size_t>(cd.ext[2]);
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int i, j, k;
    if (dim == 3)
    {
        const std::size_t ex = static_cast<std::size_t>(cd.ext[0]);
        const std::size_t ey = static_cast<std::size_t>(cd.ext[1]);
        i = static_cast<int>(idx % ex);
        j = static_cast<int>((idx / ex) % ey);
        k = static_cast<int>(idx / (ex * ey));
    }
    else
    {
        const std::size_t ex = static_cast<std::size_t>(cd.ext[0]);
        i = static_cast<int>(idx % ex);
        j = static_cast<int>(idx / ex);
        k = 0;
    }

    const int x = cd.base[0] + i;
    const int y = cd.base[1] + j;
    const int z = cd.base[2] + k;

    types::float_type sum = 0.0;
    if (dim == 3)
    {
        for (int a = 0; a < nb; ++a)
        {
            const types::float_type wx = Mx[a + nb * i];
            for (int b = 0; b < nb; ++b)
            {
                const types::float_type wxy = wx * My[b + nb * j];
                for (int c = 0; c < nb; ++c)
                {
                    const types::float_type w = wxy * Mz[c + nb * k];
                    const int px = pd.base[0] + a;
                    const int py = pd.base[1] + b;
                    const int pz = pd.base[2] + c;
                    sum += w * parent[idx3(px, py, pz, pd)];
                }
            }
        }
        child[idx3(x, y, z, cd)] = sum;
    }
    else
    {
        for (int a = 0; a < nb; ++a)
        {
            const types::float_type wx = Mx[a + nb * i];
            for (int b = 0; b < nb; ++b)
            {
                const types::float_type w = wx * My[b + nb * j];
                const int px = pd.base[0] + a;
                const int py = pd.base[1] + b;
                sum += w * parent[idx2(px, py, pd)];
            }
        }
        child[idx2(x, y, cd)] = sum;
    }
}

__global__ void antrp_kernel(const types::float_type* child,
    types::float_type* parent, const types::float_type* Mx,
    const types::float_type* My, const types::float_type* Mz,
    int nb, field_desc cd, field_desc pd, int dim)
{
    const std::size_t total =
        static_cast<std::size_t>(pd.ext[0]) *
        static_cast<std::size_t>(pd.ext[1]) *
        static_cast<std::size_t>(pd.ext[2]);
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int i, j, k;
    if (dim == 3)
    {
        const std::size_t ex = static_cast<std::size_t>(pd.ext[0]);
        const std::size_t ey = static_cast<std::size_t>(pd.ext[1]);
        i = static_cast<int>(idx % ex);
        j = static_cast<int>((idx / ex) % ey);
        k = static_cast<int>(idx / (ex * ey));
    }
    else
    {
        const std::size_t ex = static_cast<std::size_t>(pd.ext[0]);
        i = static_cast<int>(idx % ex);
        j = static_cast<int>(idx / ex);
        k = 0;
    }

    const int x = pd.base[0] + i;
    const int y = pd.base[1] + j;
    const int z = pd.base[2] + k;

    types::float_type sum = 0.0;
    if (dim == 3)
    {
        for (int a = 0; a < nb; ++a)
        {
            const types::float_type wx = Mx[i + nb * a];
            for (int b = 0; b < nb; ++b)
            {
                const types::float_type wxy = wx * My[j + nb * b];
                for (int c = 0; c < nb; ++c)
                {
                    const types::float_type w = wxy * Mz[k + nb * c];
                    const int cx = cd.base[0] + a;
                    const int cy = cd.base[1] + b;
                    const int cz = cd.base[2] + c;
                    sum += w * child[idx3(cx, cy, cz, cd)];
                }
            }
        }
        parent[idx3(x, y, z, pd)] += sum;
    }
    else
    {
        for (int a = 0; a < nb; ++a)
        {
            const types::float_type wx = Mx[i + nb * a];
            for (int b = 0; b < nb; ++b)
            {
                const types::float_type w = wx * My[j + nb * b];
                const int cx = cd.base[0] + a;
                const int cy = cd.base[1] + b;
                sum += w * child[idx2(cx, cy, cd)];
            }
        }
        parent[idx2(x, y, pd)] += sum;
    }
}
} // namespace

void intrp_child_device(const types::float_type* parent,
    types::float_type* child, const types::float_type* Mx,
    const types::float_type* My, const types::float_type* Mz, int nb,
    const field_desc& parent_desc, const field_desc& child_desc, int dim)
{
    const std::size_t total =
        static_cast<std::size_t>(child_desc.ext[0]) *
        static_cast<std::size_t>(child_desc.ext[1]) *
        static_cast<std::size_t>(child_desc.ext[2]);
    if (total == 0) return;
    const int block = 128;
    const int grid = static_cast<int>((total + block - 1) / block);
    intrp_kernel<<<grid, block>>>(parent, child, Mx, My, Mz, nb,
        parent_desc, child_desc, dim);
}

void antrp_child_device(const types::float_type* child,
    types::float_type* parent, const types::float_type* Mx,
    const types::float_type* My, const types::float_type* Mz, int nb,
    const field_desc& child_desc, const field_desc& parent_desc, int dim)
{
    const std::size_t total =
        static_cast<std::size_t>(parent_desc.ext[0]) *
        static_cast<std::size_t>(parent_desc.ext[1]) *
        static_cast<std::size_t>(parent_desc.ext[2]);
    if (total == 0) return;
    const int block = 128;
    const int grid = static_cast<int>((total + block - 1) / block);
    antrp_kernel<<<grid, block>>>(child, parent, Mx, My, Mz, nb,
        child_desc, parent_desc, dim);
}

__global__ void add_source_correction_kernel(const types::float_type* src,
    types::float_type* dst, field_desc d, int nb, types::float_type dx,
    types::float_type omega, int dim)
{
    const std::size_t total =
        static_cast<std::size_t>(d.ext[0]) *
        static_cast<std::size_t>(d.ext[1]) *
        static_cast<std::size_t>(d.ext[2]);
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int i, j, k;
    if (dim == 3)
    {
        const std::size_t ex = static_cast<std::size_t>(d.ext[0]);
        const std::size_t ey = static_cast<std::size_t>(d.ext[1]);
        i = static_cast<int>(idx % ex);
        j = static_cast<int>((idx / ex) % ey);
        k = static_cast<int>(idx / (ex * ey));
        if (i <= 0 || j <= 0 || k <= 0 || i >= nb - 1 || j >= nb - 1 ||
            k >= nb - 1)
        { return; }
        const int x = d.base[0] + i;
        const int y = d.base[1] + j;
        const int z = d.base[2] + k;
        const auto c = idx3(x, y, z, d);
        const auto xm = idx3(x - 1, y, z, d);
        const auto xp = idx3(x + 1, y, z, d);
        const auto ym = idx3(x, y - 1, z, d);
        const auto yp = idx3(x, y + 1, z, d);
        const auto zm = idx3(x, y, z - 1, d);
        const auto zp = idx3(x, y, z + 1, d);
        const types::float_type inv = 1.0 / (dx * dx);
        dst[c] += 6.0 * src[c] * inv;
        dst[c] -= src[zm] * inv;
        dst[c] -= src[zp] * inv;
        dst[c] -= src[ym] * inv;
        dst[c] -= src[yp] * inv;
        dst[c] -= src[xm] * inv;
        dst[c] -= src[xp] * inv;
        if (omega != 0.0)
        {
            dst[c] += omega * omega * src[c];
        }
    }
    else
    {
        const std::size_t ex = static_cast<std::size_t>(d.ext[0]);
        i = static_cast<int>(idx % ex);
        j = static_cast<int>(idx / ex);
        k = 0;
        if (i <= 0 || j <= 0 || i >= nb - 1 || j >= nb - 1) { return; }
        const int x = d.base[0] + i;
        const int y = d.base[1] + j;
        const auto c = idx2(x, y, d);
        const auto xm = idx2(x - 1, y, d);
        const auto xp = idx2(x + 1, y, d);
        const auto ym = idx2(x, y - 1, d);
        const auto yp = idx2(x, y + 1, d);
        const types::float_type inv = 1.0 / (dx * dx);
        dst[c] += 4.0 * src[c] * inv;
        dst[c] -= src[ym] * inv;
        dst[c] -= src[yp] * inv;
        dst[c] -= src[xm] * inv;
        dst[c] -= src[xp] * inv;
        if (omega != 0.0)
        {
            dst[c] += omega * omega * src[c];
        }
    }
}

void add_source_correction_device(const types::float_type* src,
    types::float_type* dst, const field_desc& desc, int nb,
    types::float_type dx, types::float_type omega, int dim)
{
    const std::size_t total =
        static_cast<std::size_t>(desc.ext[0]) *
        static_cast<std::size_t>(desc.ext[1]) *
        static_cast<std::size_t>(desc.ext[2]);
    if (total == 0) return;
    const int block = 128;
    const int grid = static_cast<int>((total + block - 1) / block);
    add_source_correction_kernel<<<grid, block>>>(src, dst, desc, nb, dx, omega,
        dim);
}

} // namespace interp
} // namespace gpu
} // namespace iblgf
