#include <iblgf/operators/operators_gpu.hpp>

#include <cuda_runtime.h>

namespace iblgf
{
namespace domain
{
namespace gpu
{
namespace
{
__host__ __device__ inline int idx3(
    int x, int y, int z, int nx, int ny, int nz) noexcept
{
    (void)nz;
    return x + nx * (y + ny * z);
}

__global__ void curl_face_to_edge_kernel(FieldView3D src0, FieldView3D src1,
    FieldView3D src2, FieldView3D dst0, FieldView3D dst1, FieldView3D dst2,
    int px, int py, int pz, double fac)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= px || j >= py || k >= pz) return;

    const int sx = i + src0.ox;
    const int sy = j + src0.oy;
    const int sz = k + src0.oz;
    const int dx = i + dst0.ox;
    const int dy = j + dst0.oy;
    const int dz = k + dst0.oz;

    const int s000 = idx3(sx, sy, sz, src0.nx, src0.ny, src0.nz);
    const int sm010 = idx3(sx, sy - 1, sz, src0.nx, src0.ny, src0.nz);
    const int sm001 = idx3(sx, sy, sz - 1, src0.nx, src0.ny, src0.nz);
    const int sm100 = idx3(sx - 1, sy, sz, src0.nx, src0.ny, src0.nz);
    const int d000 = idx3(dx, dy, dz, dst0.nx, dst0.ny, dst0.nz);

    dst0.data[d000] =
        (src2.data[s000] - src2.data[sm010] - src1.data[s000] +
            src1.data[sm001]) *
        fac;
    dst1.data[d000] =
        (src0.data[s000] - src0.data[sm001] - src2.data[s000] +
            src2.data[sm100]) *
        fac;
    dst2.data[d000] =
        (src1.data[s000] - src1.data[sm100] - src0.data[s000] +
            src0.data[sm010]) *
        fac;
}

__global__ void divergence_face_to_cell_kernel(FieldView3D src0,
    FieldView3D src1, FieldView3D src2, FieldView3D dst, int px, int py,
    int pz, double fac)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= px || j >= py || k >= pz) return;

    const int sx = i + src0.ox;
    const int sy = j + src0.oy;
    const int sz = k + src0.oz;
    const int dx = i + dst.ox;
    const int dy = j + dst.oy;
    const int dz = k + dst.oz;

    const int s000 = idx3(sx, sy, sz, src0.nx, src0.ny, src0.nz);
    dst.data[idx3(dx, dy, dz, dst.nx, dst.ny, dst.nz)] =
        (-src0.data[s000] - src1.data[s000] - src2.data[s000] +
            src0.data[idx3(sx + 1, sy, sz, src0.nx, src0.ny, src0.nz)] +
            src1.data[idx3(sx, sy + 1, sz, src1.nx, src1.ny, src1.nz)] +
            src2.data[idx3(sx, sy, sz + 1, src2.nx, src2.ny, src2.nz)]) *
        fac;
}

__global__ void gradient_cell_to_face_kernel(FieldView3D src, FieldView3D dst0,
    FieldView3D dst1, FieldView3D dst2, int px, int py, int pz, double fac)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= px || j >= py || k >= pz) return;

    const int sx = i + src.ox;
    const int sy = j + src.oy;
    const int sz = k + src.oz;
    const int d0x = i + dst0.ox;
    const int d0y = j + dst0.oy;
    const int d0z = k + dst0.oz;

    const int s000 = idx3(sx, sy, sz, src.nx, src.ny, src.nz);
    dst0.data[idx3(d0x, d0y, d0z, dst0.nx, dst0.ny, dst0.nz)] =
        (src.data[s000] -
            src.data[idx3(sx - 1, sy, sz, src.nx, src.ny, src.nz)]) *
        fac;
    dst1.data[idx3(d0x, d0y, d0z, dst1.nx, dst1.ny, dst1.nz)] =
        (src.data[s000] -
            src.data[idx3(sx, sy - 1, sz, src.nx, src.ny, src.nz)]) *
        fac;
    dst2.data[idx3(d0x, d0y, d0z, dst2.nx, dst2.ny, dst2.nz)] =
        (src.data[s000] -
            src.data[idx3(sx, sy, sz - 1, src.nx, src.ny, src.nz)]) *
        fac;
}

__global__ void nonlinear_face_edge_to_face_kernel(FieldView3D face0,
    FieldView3D face1, FieldView3D face2, FieldView3D edge0, FieldView3D edge1,
    FieldView3D edge2, FieldView3D dst0, FieldView3D dst1, FieldView3D dst2,
    int px, int py, int pz, double scale)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= px || j >= py || k >= pz) return;

    const int fx = i + face0.ox;
    const int fy = j + face0.oy;
    const int fz = k + face0.oz;
    const int ex = i + edge0.ox;
    const int ey = j + edge0.oy;
    const int ez = k + edge0.oz;
    const int dx = i + dst0.ox;
    const int dy = j + dst0.oy;
    const int dz = k + dst0.oz;

    const int e000 = idx3(ex, ey, ez, edge0.nx, edge0.ny, edge0.nz);
    const int f000 = idx3(fx, fy, fz, face0.nx, face0.ny, face0.nz);

    const double out0 =
        0.25 *
        (+edge1.data[e000] *
                (+face2.data[f000] +
                    face2.data[idx3(fx - 1, fy, fz, face2.nx, face2.ny,
                        face2.nz)]) +
            edge1.data[idx3(ex, ey, ez + 1, edge1.nx, edge1.ny, edge1.nz)] *
                (+face2.data[idx3(fx, fy, fz + 1, face2.nx, face2.ny,
                      face2.nz)] +
                    face2.data[idx3(fx - 1, fy, fz + 1, face2.nx, face2.ny,
                        face2.nz)]) -
            edge2.data[e000] *
                (+face1.data[f000] +
                    face1.data[idx3(fx - 1, fy, fz, face1.nx, face1.ny,
                        face1.nz)]) -
            edge2.data[idx3(ex, ey + 1, ez, edge2.nx, edge2.ny, edge2.nz)] *
                (+face1.data[idx3(fx, fy + 1, fz, face1.nx, face1.ny,
                      face1.nz)] +
                    face1.data[idx3(fx - 1, fy + 1, fz, face1.nx, face1.ny,
                        face1.nz)]));

    const double out1 =
        0.25 *
        (+edge2.data[e000] *
                (+face0.data[f000] +
                    face0.data[idx3(fx, fy - 1, fz, face0.nx, face0.ny,
                        face0.nz)]) +
            edge2.data[idx3(ex + 1, ey, ez, edge2.nx, edge2.ny, edge2.nz)] *
                (+face0.data[idx3(fx + 1, fy, fz, face0.nx, face0.ny,
                      face0.nz)] +
                    face0.data[idx3(fx + 1, fy - 1, fz, face0.nx, face0.ny,
                        face0.nz)]) -
            edge0.data[e000] *
                (+face2.data[f000] +
                    face2.data[idx3(fx, fy - 1, fz, face2.nx, face2.ny,
                        face2.nz)]) -
            edge0.data[idx3(ex, ey, ez + 1, edge0.nx, edge0.ny, edge0.nz)] *
                (+face2.data[idx3(fx, fy, fz + 1, face2.nx, face2.ny,
                      face2.nz)] +
                    face2.data[idx3(fx, fy - 1, fz + 1, face2.nx, face2.ny,
                        face2.nz)]));

    const double out2 =
        0.25 *
        (+edge0.data[e000] *
                (+face1.data[f000] +
                    face1.data[idx3(fx, fy, fz - 1, face1.nx, face1.ny,
                        face1.nz)]) +
            edge0.data[idx3(ex, ey + 1, ez, edge0.nx, edge0.ny, edge0.nz)] *
                (+face1.data[idx3(fx, fy + 1, fz, face1.nx, face1.ny,
                      face1.nz)] +
                    face1.data[idx3(fx, fy + 1, fz - 1, face1.nx, face1.ny,
                        face1.nz)]) -
            edge1.data[e000] *
                (+face0.data[f000] +
                    face0.data[idx3(fx, fy, fz - 1, face0.nx, face0.ny,
                        face0.nz)]) -
            edge1.data[idx3(ex + 1, ey, ez, edge1.nx, edge1.ny, edge1.nz)] *
                (+face0.data[idx3(fx + 1, fy, fz, face0.nx, face0.ny,
                      face0.nz)] +
                    face0.data[idx3(fx + 1, fy, fz - 1, face0.nx, face0.ny,
                        face0.nz)]));

    dst0.data[idx3(dx, dy, dz, dst0.nx, dst0.ny, dst0.nz)] = out0 * scale;
    dst1.data[idx3(dx, dy, dz, dst1.nx, dst1.ny, dst1.nz)] = out1 * scale;
    dst2.data[idx3(dx, dy, dz, dst2.nx, dst2.ny, dst2.nz)] = out2 * scale;
}

inline dim3 launch_grid(int px, int py, int pz) noexcept
{
    constexpr int bx = 8;
    constexpr int by = 4;
    constexpr int bz = 4;
    return dim3((px + bx - 1) / bx, (py + by - 1) / by, (pz + bz - 1) / bz);
}

inline dim3 launch_block() noexcept { return dim3(8, 4, 4); }

inline bool check_last_cuda(const char* name) noexcept
{
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        (void)name;
        return false;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}
} // namespace

bool curl_face_to_edge_3d(const FieldView3D& src0, const FieldView3D& src1,
    const FieldView3D& src2, const FieldView3D& dst0,
    const FieldView3D& dst1, const FieldView3D& dst2, int px, int py, int pz,
    types::float_type dx_level) noexcept
{
    curl_face_to_edge_kernel<<<launch_grid(px, py, pz), launch_block()>>>(
        src0, src1, src2, dst0, dst1, dst2, px, py, pz, 1.0 / dx_level);
    return check_last_cuda("curl_face_to_edge_3d");
}

bool divergence_face_to_cell_3d(const FieldView3D& src0,
    const FieldView3D& src1, const FieldView3D& src2, const FieldView3D& dst,
    int px, int py, int pz, types::float_type dx_level) noexcept
{
    divergence_face_to_cell_kernel<<<launch_grid(px, py, pz), launch_block()>>>(
        src0, src1, src2, dst, px, py, pz, 1.0 / dx_level);
    return check_last_cuda("divergence_face_to_cell_3d");
}

bool gradient_cell_to_face_3d(const FieldView3D& src, const FieldView3D& dst0,
    const FieldView3D& dst1, const FieldView3D& dst2, int px, int py, int pz,
    types::float_type dx_level, types::float_type scale) noexcept
{
    gradient_cell_to_face_kernel<<<launch_grid(px, py, pz), launch_block()>>>(
        src, dst0, dst1, dst2, px, py, pz, scale / dx_level);
    return check_last_cuda("gradient_cell_to_face_3d");
}

bool nonlinear_face_edge_to_face_3d(const FieldView3D& face0,
    const FieldView3D& face1, const FieldView3D& face2,
    const FieldView3D& edge0, const FieldView3D& edge1,
    const FieldView3D& edge2, const FieldView3D& dst0,
    const FieldView3D& dst1, const FieldView3D& dst2, int px, int py, int pz,
    types::float_type scale) noexcept
{
    nonlinear_face_edge_to_face_kernel<<<launch_grid(px, py, pz),
        launch_block()>>>(face0, face1, face2, edge0, edge1, edge2, dst0, dst1,
        dst2, px, py, pz, scale);
    return check_last_cuda("nonlinear_face_edge_to_face_3d");
}

} // namespace gpu
} // namespace domain
} // namespace iblgf
