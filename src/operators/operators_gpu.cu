#include <iblgf/operators/operators_gpu.hpp>
#include <cuda_runtime.h>

namespace iblgf
{
namespace domain
{
namespace gpu
{

// Curl kernel — matches Operator::curl<Source, Dest> (operators.hpp:1494-1504)
//
// CPU stencil (3D):
//   edge[0] = (face[2](i,j,k) - face[2](i,j-1,k) - face[1](i,j,k) + face[1](i,j,k-1)) * inv_dx
//   edge[1] = (face[0](i,j,k) - face[0](i,j,k-1) - face[2](i,j,k) + face[2](i-1,j,k)) * inv_dx
//   edge[2] = (face[1](i,j,k) - face[1](i-1,j,k) - face[0](i,j,k) + face[0](i,j-1,k)) * inv_dx
//
// Data layout: flat row-major, x-fastest. idx = i + j*nx + k*nx*ny.
// Ghost cells at index 0 (and nx-1/ny-1/nz-1) are filled by MPI before this kernel runs.
// Skip i==0, j==0, k==0 to avoid out-of-bounds negative-index reads (same as CPU block iterator).
__global__ void curl_kernel_3d(
    const double* __restrict__ fx,
    const double* __restrict__ fy,
    const double* __restrict__ fz,
    double* __restrict__ ex,
    double* __restrict__ ey,
    double* __restrict__ ez,
    int nx, int ny, int nz,
    double inv_dx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;
    if (i == 0 || j == 0 || k == 0)   return;

    const int s   = nx * ny;
    const int idx = i + j * nx + k * s;

    ex[idx] = (fz[idx] - fz[idx - nx] - fy[idx] + fy[idx - s])  * inv_dx;
    ey[idx] = (fx[idx] - fx[idx - s]  - fz[idx] + fz[idx - 1])  * inv_dx;
    ez[idx] = (fy[idx] - fy[idx - 1]  - fx[idx] + fx[idx - nx]) * inv_dx;
}

// Nonlinear kernel — matches Operator::nonlinear<Face, Edge, Dest> (operators.hpp:2473-2512)
//
// Computes ω×u (vorticity × velocity) using face-centered velocity and edge-centered vorticity.
//
// dest_x = 0.25*(  ey[i,j,k]  *(fz[i,j,k]+fz[i-1,j,k])
//                + ey[i,j,k+1]*(fz[i,j,k+1]+fz[i-1,j,k+1])
//                - ez[i,j,k]  *(fy[i,j,k]+fy[i-1,j,k])
//                - ez[i,j+1,k]*(fy[i,j+1,k]+fy[i-1,j+1,k]) )
//
// dest_y = 0.25*(  ez[i,j,k]  *(fx[i,j,k]+fx[i,j-1,k])
//                + ez[i+1,j,k]*(fx[i+1,j,k]+fx[i+1,j-1,k])
//                - ex[i,j,k]  *(fz[i,j,k]+fz[i,j-1,k])
//                - ex[i,j,k+1]*(fz[i,j,k+1]+fz[i,j-1,k+1]) )
//
// dest_z = 0.25*(  ex[i,j,k]  *(fy[i,j,k]+fy[i,j,k-1])
//                + ex[i,j+1,k]*(fy[i,j+1,k]+fy[i,j+1,k-1])
//                - ey[i,j,k]  *(fx[i,j,k]+fx[i,j,k-1])
//                - ey[i+1,j,k]*(fx[i+1,j,k]+fx[i+1,j,k-1]) )
__global__ void nonlinear_kernel_3d(
    const double* __restrict__ fx,
    const double* __restrict__ fy,
    const double* __restrict__ fz,
    const double* __restrict__ ex,
    const double* __restrict__ ey,
    const double* __restrict__ ez,
    double* __restrict__ dx,
    double* __restrict__ dy,
    double* __restrict__ dz,
    int nx, int ny, int nz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;
    // Skip boundary nodes where ±1 offsets go out of range
    if (i == 0 || j == 0 || k == 0)   return;
    if (i == nx - 1 || j == ny - 1 || k == nz - 1) return;

    const int s   = nx * ny;
    const int idx = i + j * nx + k * s;

    dx[idx] = 0.25 * (
          ey[idx]       * (fz[idx]      + fz[idx - 1])
        + ey[idx + s]   * (fz[idx + s]  + fz[idx + s - 1])
        - ez[idx]       * (fy[idx]      + fy[idx - 1])
        - ez[idx + nx]  * (fy[idx + nx] + fy[idx + nx - 1]));

    dy[idx] = 0.25 * (
          ez[idx]       * (fx[idx]          + fx[idx - nx])
        + ez[idx + 1]   * (fx[idx + 1]      + fx[idx + 1 - nx])
        - ex[idx]       * (fz[idx]          + fz[idx - nx])
        - ex[idx + s]   * (fz[idx + s]      + fz[idx + s - nx]));

    dz[idx] = 0.25 * (
          ex[idx]       * (fy[idx]          + fy[idx - s])
        + ex[idx + nx]  * (fy[idx + nx]     + fy[idx + nx - s])
        - ey[idx]       * (fx[idx]          + fx[idx - s])
        - ey[idx + 1]   * (fx[idx + 1]      + fx[idx + 1 - s]));
}

__global__ void scale_field_kernel(double* __restrict__ field, int n, double scale)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        field[i] *= scale;
}

} // namespace gpu
} // namespace domain
} // namespace iblgf
