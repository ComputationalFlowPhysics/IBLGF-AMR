#pragma once
#ifdef IBLGF_COMPILE_CUDA

#include <cuda_runtime.h>
#include <iblgf/types.hpp>

namespace iblgf
{
namespace domain
{
namespace gpu
{

// Curl: face-centered velocity → edge-centered vorticity
// Matches Operator::curl<Source, Dest>(block, dx_level) in operators.hpp:1483
__global__ void curl_kernel_3d(
    const double* __restrict__ face_x,
    const double* __restrict__ face_y,
    const double* __restrict__ face_z,
    double* __restrict__ edge_x,
    double* __restrict__ edge_y,
    double* __restrict__ edge_z,
    int nx, int ny, int nz,
    double inv_dx);

// Nonlinear cross product: ω×u → face-centered dest
// Matches Operator::nonlinear<Face, Edge, Dest>(block) in operators.hpp:2460
__global__ void nonlinear_kernel_3d(
    const double* __restrict__ face_x,
    const double* __restrict__ face_y,
    const double* __restrict__ face_z,
    const double* __restrict__ edge_x,
    const double* __restrict__ edge_y,
    const double* __restrict__ edge_z,
    double* __restrict__ dest_x,
    double* __restrict__ dest_y,
    double* __restrict__ dest_z,
    int nx, int ny, int nz);

// Element-wise scale: field[i] *= scale
__global__ void scale_field_kernel(double* __restrict__ field, int n, double scale);

// Per-block launchers — called from ifherk.hpp GPU code path

template<class Source, class EdgeAux, class Block>
void launch_curl_gpu(Block& block, double inv_dx, cudaStream_t stream)
{
    constexpr auto src_tag  = Source::tag();
    constexpr auto edge_tag = EdgeAux::tag();

    const auto& ext = block.data_r(src_tag, 0).real_block().extent();
    const int nx = ext[0], ny = ext[1], nz = ext[2];

    // Upload source fields (ghost cells already filled by MPI on host)
    for (int c = 0; c < 3; ++c)
        block.data_r(src_tag,  c).update_device(stream, /*force=*/true);

    // Ensure edge dest buffers are allocated on device
    for (int c = 0; c < 3; ++c)
        block.data_r(edge_tag, c).device_ptr(); // triggers ensure_device()

    dim3 blk(8, 8, 4);
    dim3 grd((nx + blk.x - 1) / blk.x,
             (ny + blk.y - 1) / blk.y,
             (nz + blk.z - 1) / blk.z);

    curl_kernel_3d<<<grd, blk, 0, stream>>>(
        block.data_r(src_tag,  0).device_ptr(),
        block.data_r(src_tag,  1).device_ptr(),
        block.data_r(src_tag,  2).device_ptr(),
        block.data_r(edge_tag, 0).device_ptr(),
        block.data_r(edge_tag, 1).device_ptr(),
        block.data_r(edge_tag, 2).device_ptr(),
        nx, ny, nz, inv_dx);
}

template<class FaceAux, class EdgeAux, class Target, class Block>
void launch_nonlinear_gpu(Block& block, double scale, cudaStream_t stream)
{
    constexpr auto face_tag = FaceAux::tag();
    constexpr auto edge_tag = EdgeAux::tag();
    constexpr auto dest_tag = Target::tag();

    const auto& ext = block.data_r(face_tag, 0).real_block().extent();
    const int nx = ext[0], ny = ext[1], nz = ext[2];

    // Upload edge and face fields (ghost cells filled by MPI)
    for (int c = 0; c < 3; ++c) {
        block.data_r(face_tag, c).update_device(stream, /*force=*/true);
        block.data_r(edge_tag, c).update_device(stream, /*force=*/true);
    }

    // Ensure dest buffers are allocated on device
    for (int c = 0; c < 3; ++c)
        block.data_r(dest_tag, c).device_ptr();

    dim3 blk(8, 8, 4);
    dim3 grd((nx + blk.x - 1) / blk.x,
             (ny + blk.y - 1) / blk.y,
             (nz + blk.z - 1) / blk.z);

    nonlinear_kernel_3d<<<grd, blk, 0, stream>>>(
        block.data_r(face_tag, 0).device_ptr(),
        block.data_r(face_tag, 1).device_ptr(),
        block.data_r(face_tag, 2).device_ptr(),
        block.data_r(edge_tag, 0).device_ptr(),
        block.data_r(edge_tag, 1).device_ptr(),
        block.data_r(edge_tag, 2).device_ptr(),
        block.data_r(dest_tag, 0).device_ptr(),
        block.data_r(dest_tag, 1).device_ptr(),
        block.data_r(dest_tag, 2).device_ptr(),
        nx, ny, nz);

    // Scale in-place on GPU (replaces CPU lin_data *= _scale)
    const int n = nx * ny * nz;
    int blk1d = 256;
    int grd1d = (n + blk1d - 1) / blk1d;
    for (int c = 0; c < 3; ++c)
        scale_field_kernel<<<grd1d, blk1d, 0, stream>>>(
            block.data_r(dest_tag, c).device_ptr(), n, scale);
}

} // namespace gpu
} // namespace domain
} // namespace iblgf

#endif // IBLGF_COMPILE_CUDA
