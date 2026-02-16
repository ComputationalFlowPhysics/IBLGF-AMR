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

#ifndef IBLGF_INCLUDED_OPERATORS_GPU_HPP
#define IBLGF_INCLUDED_OPERATORS_GPU_HPP

#include <cuda_runtime.h>
#include <iblgf/types.hpp>

namespace iblgf {
namespace operators {
namespace gpu {

using float_type = double;

// ============================================================================
// Field Operations (AXPY, copy, scale, clean)
// ============================================================================

// Generic AXPY: dest = alpha * src + beta * dest
__global__ void axpy_kernel(
    const float_type* src,
    float_type* dest,
    float_type alpha,
    float_type beta,
    size_t n_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        dest[idx] = alpha * src[idx] + beta * dest[idx];
    }
}

// Copy with scale: dest = alpha * src
__global__ void copy_scale_kernel(
    const float_type* src,
    float_type* dest,
    float_type alpha,
    size_t n_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        dest[idx] = alpha * src[idx];
    }
}

// Zero out field
__global__ void clean_kernel(
    float_type* data,
    size_t n_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        data[idx] = 0.0;
    }
}

// Element-wise multiply
__global__ void multiply_kernel(
    const float_type* src1,
    const float_type* src2,
    float_type* dest,
    size_t n_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        dest[idx] = src1[idx] * src2[idx];
    }
}

// ============================================================================
// Differential Operators (Gradient, Divergence, Curl)
// ============================================================================

// Gradient: cell-centered to face-centered (3D)
// Computes face[i] = (cell[i] - cell[i-1]) / dx
__global__ void gradient_x_kernel(
    const float_type* cell,
    float_type* face_x,
    int nx, int ny, int nz,
    float_type inv_dx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny && k < nz && i > 0) {
        int idx = i + j * nx + k * nx * ny;
        int idx_prev = (i-1) + j * nx + k * nx * ny;
        face_x[idx] = (cell[idx] - cell[idx_prev]) * inv_dx;
    }
}

__global__ void gradient_y_kernel(
    const float_type* cell,
    float_type* face_y,
    int nx, int ny, int nz,
    float_type inv_dy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny && k < nz && j > 0) {
        int idx = i + j * nx + k * nx * ny;
        int idx_prev = i + (j-1) * nx + k * nx * ny;
        face_y[idx] = (cell[idx] - cell[idx_prev]) * inv_dy;
    }
}

__global__ void gradient_z_kernel(
    const float_type* cell,
    float_type* face_z,
    int nx, int ny, int nz,
    float_type inv_dz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny && k < nz && k > 0) {
        int idx = i + j * nx + k * nx * ny;
        int idx_prev = i + j * nx + (k-1) * nx * ny;
        face_z[idx] = (cell[idx] - cell[idx_prev]) * inv_dz;
    }
}

// Divergence: face-centered to cell-centered
// Computes cell[i] = (face_x[i+1] - face_x[i]) / dx + (face_y[i+ny] - face_y[i]) / dy + ...
__global__ void divergence_kernel(
    const float_type* face_x,
    const float_type* face_y,
    const float_type* face_z,
    float_type* cell,
    int nx, int ny, int nz,
    float_type inv_dx,
    float_type inv_dy,
    float_type inv_dz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx-1 && j < ny-1 && k < nz-1) {
        int idx = i + j * nx + k * nx * ny;
        int idx_xp = (i+1) + j * nx + k * nx * ny;
        int idx_yp = i + (j+1) * nx + k * nx * ny;
        int idx_zp = i + j * nx + (k+1) * nx * ny;
        
        cell[idx] = (face_x[idx_xp] - face_x[idx]) * inv_dx +
                    (face_y[idx_yp] - face_y[idx]) * inv_dy +
                    (face_z[idx_zp] - face_z[idx]) * inv_dz;
    }
}

// Curl: face-centered velocity to edge-centered vorticity
// ω_x = ∂w/∂y - ∂v/∂z (on x-edges)
__global__ void curl_x_kernel(
    const float_type* face_y,  // v component
    const float_type* face_z,  // w component
    float_type* edge_x,        // ω_x component
    int nx, int ny, int nz,
    float_type inv_dy,
    float_type inv_dz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny-1 && k < nz-1) {
        int idx = i + j * nx + k * nx * ny;
        int idx_yp = i + (j+1) * nx + k * nx * ny;
        int idx_zp = i + j * nx + (k+1) * nx * ny;
        
        float_type dw_dy = (face_z[idx_yp] - face_z[idx]) * inv_dy;
        float_type dv_dz = (face_y[idx_zp] - face_y[idx]) * inv_dz;
        
        edge_x[idx] = dw_dy - dv_dz;
    }
}

// ω_y = ∂u/∂z - ∂w/∂x (on y-edges)
__global__ void curl_y_kernel(
    const float_type* face_x,  // u component
    const float_type* face_z,  // w component
    float_type* edge_y,        // ω_y component
    int nx, int ny, int nz,
    float_type inv_dx,
    float_type inv_dz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx-1 && j < ny && k < nz-1) {
        int idx = i + j * nx + k * nx * ny;
        int idx_xp = (i+1) + j * nx + k * nx * ny;
        int idx_zp = i + j * nx + (k+1) * nx * ny;
        
        float_type du_dz = (face_x[idx_zp] - face_x[idx]) * inv_dz;
        float_type dw_dx = (face_z[idx_xp] - face_z[idx]) * inv_dx;
        
        edge_y[idx] = du_dz - dw_dx;
    }
}

// ω_z = ∂v/∂x - ∂u/∂y (on z-edges)
__global__ void curl_z_kernel(
    const float_type* face_x,  // u component
    const float_type* face_y,  // v component
    float_type* edge_z,        // ω_z component
    int nx, int ny, int nz,
    float_type inv_dx,
    float_type inv_dy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx-1 && j < ny-1 && k < nz) {
        int idx = i + j * nx + k * nx * ny;
        int idx_xp = (i+1) + j * nx + k * nx * ny;
        int idx_yp = i + (j+1) * nx + k * nx * ny;
        
        float_type dv_dx = (face_y[idx_xp] - face_y[idx]) * inv_dx;
        float_type du_dy = (face_x[idx_yp] - face_x[idx]) * inv_dy;
        
        edge_z[idx] = dv_dx - du_dy;
    }
}

// ============================================================================
// Nonlinear Operator: ∇×(u × ω)
// ============================================================================

// Nonlinear advection term for x-component on faces
// nl_x = ∂/∂y(u_z * ω_y) - ∂/∂z(u_y * ω_z)
__global__ void nonlinear_x_kernel(
    const float_type* face_y,  // u_y
    const float_type* face_z,  // u_z
    const float_type* edge_y,  // ω_y
    const float_type* edge_z,  // ω_z
    float_type* nl_x,
    int nx, int ny, int nz,
    float_type inv_dy,
    float_type inv_dz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j > 0 && j < ny && k > 0 && k < nz) {
        int idx = i + j * nx + k * nx * ny;
        int idx_ym = i + (j-1) * nx + k * nx * ny;
        int idx_zm = i + j * nx + (k-1) * nx * ny;
        
        // Average to face locations
        float_type uz_avg = 0.5 * (face_z[idx] + face_z[idx_ym]);
        float_type wy_avg = 0.5 * (edge_y[idx] + edge_y[idx_ym]);
        
        float_type uy_avg = 0.5 * (face_y[idx] + face_y[idx_zm]);
        float_type wz_avg = 0.5 * (edge_z[idx] + edge_z[idx_zm]);
        
        float_type term1 = uz_avg * wy_avg;
        float_type term2 = uy_avg * wz_avg;
        
        nl_x[idx] = (term1 - (i > 0 ? (face_z[idx_ym] * edge_y[idx_ym]) : 0.0)) * inv_dy
                  - (term2 - (i > 0 ? (face_y[idx_zm] * edge_z[idx_zm]) : 0.0)) * inv_dz;
    }
}

// Similar kernels for y and z components would follow the same pattern...

// ============================================================================
// Laplacian Operator (for viscous terms)
// ============================================================================

__global__ void laplacian_kernel(
    const float_type* field,
    float_type* laplacian,
    int nx, int ny, int nz,
    float_type inv_dx2,
    float_type inv_dy2,
    float_type inv_dz2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
        int idx = i + j * nx + k * nx * ny;
        int idx_xm = (i-1) + j * nx + k * nx * ny;
        int idx_xp = (i+1) + j * nx + k * nx * ny;
        int idx_ym = i + (j-1) * nx + k * nx * ny;
        int idx_yp = i + (j+1) * nx + k * nx * ny;
        int idx_zm = i + j * nx + (k-1) * nx * ny;
        int idx_zp = i + j * nx + (k+1) * nx * ny;
        
        laplacian[idx] = (field[idx_xp] - 2.0*field[idx] + field[idx_xm]) * inv_dx2
                       + (field[idx_yp] - 2.0*field[idx] + field[idx_ym]) * inv_dy2
                       + (field[idx_zp] - 2.0*field[idx] + field[idx_zm]) * inv_dz2;
    }
}

} // namespace gpu
} // namespace operators
} // namespace iblgf

#endif // IBLGF_INCLUDED_OPERATORS_GPU_HPP
