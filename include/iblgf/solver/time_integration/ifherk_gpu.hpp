#ifndef IBLGF_INCLUDED_IFHERK_GPU_HPP
#define IBLGF_INCLUDED_IFHERK_GPU_HPP

#ifdef IBLGF_COMPILE_CUDA

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

#include <iblgf/global.hpp>

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

__global__ void curl_kernel(const float_type* src0, const float_type* src1,
    const float_type* src2, float_type* dst0, float_type* dst1,
    float_type* dst2, int ex, int ey, int ez, int ox, int oy, int oz, int nx,
    int ny, int nz, float_type inv_dx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    const int ii = i + ox;
    const int jj = j + oy;
    const int kk = k + oz;
    const int idx = (kk * ey + jj) * ex + ii;
    const int idx_im = (kk * ey + jj) * ex + (ii - 1);
    const int idx_jm = (kk * ey + (jj - 1)) * ex + ii;
    const int idx_km = ((kk - 1) * ey + jj) * ex + ii;

    // Match domain::Operator::curl<face,edge> stencil exactly.
    dst0[idx] = (src2[idx] - src2[idx_jm] - src1[idx] + src1[idx_km]) * inv_dx;
    dst1[idx] = (src0[idx] - src0[idx_km] - src2[idx] + src2[idx_im]) * inv_dx;
    dst2[idx] = (src1[idx] - src1[idx_im] - src0[idx] + src0[idx_jm]) * inv_dx;
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

    const int ii = i + ox;
    const int jj = j + oy;
    const int kk = k + oz;

    const int idx = (kk * ey + jj) * ex + ii;
    const int idx_im = (kk * ey + jj) * ex + (ii - 1);
    const int idx_ip = (kk * ey + jj) * ex + (ii + 1);
    const int idx_jm = (kk * ey + (jj - 1)) * ex + ii;
    const int idx_jp = (kk * ey + (jj + 1)) * ex + ii;
    const int idx_km = ((kk - 1) * ey + jj) * ex + ii;
    const int idx_kp = ((kk + 1) * ey + jj) * ex + ii;

    // Match domain::Operator::nonlinear<face,edge,face> stencil exactly.
    dst0[idx] = 0.25 *
                (edge1[idx] * (face2[idx] + face2[idx_im]) +
                    edge1[idx_kp] * (face2[idx_kp] + face2[idx_kp - 1]) -
                    edge2[idx] * (face1[idx] + face1[idx_im]) -
                    edge2[idx_jp] * (face1[idx_jp] + face1[idx_jp - 1]));

    dst1[idx] = 0.25 *
                (edge2[idx] * (face0[idx] + face0[idx_jm]) +
                    edge2[idx_ip] * (face0[idx_ip] + face0[idx_ip - ex]) -
                    edge0[idx] * (face2[idx] + face2[idx_jm]) -
                    edge0[idx_kp] * (face2[idx_kp] + face2[idx_kp - ex]));

    dst2[idx] = 0.25 *
                (edge0[idx] * (face1[idx] + face1[idx_km]) +
                    edge0[idx_jp] * (face1[idx_jp] + face1[idx_jp - ex * ey]) -
                    edge1[idx] * (face0[idx] + face0[idx_km]) -
                    edge1[idx_ip] * (face0[idx_ip] + face0[idx_ip - ex * ey]));
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
    curl_kernel<<<blocks, threads>>>(p_src0, p_src1, p_src2, p_dst0, p_dst1,
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
    nonlinear_kernel<<<blocks, threads>>>(p_face0, p_face1, p_face2, p_edge0,
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

} // namespace ifherk_gpu
} // namespace solver
} // namespace iblgf

#endif // IBLGF_COMPILE_CUDA

#endif // IBLGF_INCLUDED_IFHERK_GPU_HPP
