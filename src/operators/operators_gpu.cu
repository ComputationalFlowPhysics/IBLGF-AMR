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

#include <iblgf/operators/operators_gpu.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

namespace iblgf {
namespace operators {
namespace gpu {

// ============================================================================
// Helper: Launch configuration
// ============================================================================

struct LaunchConfig {
    dim3 blockDim;
    dim3 gridDim;
    
    LaunchConfig(int nx, int ny, int nz) {
        // Use 8x8x8 thread blocks for 3D kernels
        blockDim = dim3(8, 8, 8);
        gridDim = dim3(
            (nx + blockDim.x - 1) / blockDim.x,
            (ny + blockDim.y - 1) / blockDim.y,
            (nz + blockDim.z - 1) / blockDim.z
        );
    }
    
    LaunchConfig(size_t n_elements) {
        // Use 256 threads for 1D kernels
        blockDim = dim3(256, 1, 1);
        gridDim = dim3((n_elements + 255) / 256, 1, 1);
    }
};

// ============================================================================
// Host wrapper functions
// ============================================================================

void axpy(
    const float_type* src_d,
    float_type* dest_d,
    float_type alpha,
    float_type beta,
    size_t n_elements,
    cudaStream_t stream)
{
    LaunchConfig cfg(n_elements);
    axpy_kernel<<<cfg.gridDim, cfg.blockDim, 0, stream>>>(
        src_d, dest_d, alpha, beta, n_elements);
}

void copy_scale(
    const float_type* src_d,
    float_type* dest_d,
    float_type alpha,
    size_t n_elements,
    cudaStream_t stream)
{
    LaunchConfig cfg(n_elements);
    copy_scale_kernel<<<cfg.gridDim, cfg.blockDim, 0, stream>>>(
        src_d, dest_d, alpha, n_elements);
}

void clean(
    float_type* data_d,
    size_t n_elements,
    cudaStream_t stream)
{
    // Use cudaMemsetAsync for better performance
    cudaMemsetAsync(data_d, 0, n_elements * sizeof(float_type), stream);
}

void multiply(
    const float_type* src1_d,
    const float_type* src2_d,
    float_type* dest_d,
    size_t n_elements,
    cudaStream_t stream)
{
    LaunchConfig cfg(n_elements);
    multiply_kernel<<<cfg.gridDim, cfg.blockDim, 0, stream>>>(
        src1_d, src2_d, dest_d, n_elements);
}

void gradient_x(
    const float_type* cell_d,
    float_type* face_x_d,
    int nx, int ny, int nz,
    float_type inv_dx,
    cudaStream_t stream)
{
    LaunchConfig cfg(nx, ny, nz);
    gradient_x_kernel<<<cfg.gridDim, cfg.blockDim, 0, stream>>>(
        cell_d, face_x_d, nx, ny, nz, inv_dx);
}

void gradient_y(
    const float_type* cell_d,
    float_type* face_y_d,
    int nx, int ny, int nz,
    float_type inv_dy,
    cudaStream_t stream)
{
    LaunchConfig cfg(nx, ny, nz);
    gradient_y_kernel<<<cfg.gridDim, cfg.blockDim, 0, stream>>>(
        cell_d, face_y_d, nx, ny, nz, inv_dy);
}

void gradient_z(
    const float_type* cell_d,
    float_type* face_z_d,
    int nx, int ny, int nz,
    float_type inv_dz,
    cudaStream_t stream)
{
    LaunchConfig cfg(nx, ny, nz);
    gradient_z_kernel<<<cfg.gridDim, cfg.blockDim, 0, stream>>>(
        cell_d, face_z_d, nx, ny, nz, inv_dz);
}

void divergence(
    const float_type* face_x_d,
    const float_type* face_y_d,
    const float_type* face_z_d,
    float_type* cell_d,
    int nx, int ny, int nz,
    float_type inv_dx,
    float_type inv_dy,
    float_type inv_dz,
    cudaStream_t stream)
{
    LaunchConfig cfg(nx, ny, nz);
    divergence_kernel<<<cfg.gridDim, cfg.blockDim, 0, stream>>>(
        face_x_d, face_y_d, face_z_d, cell_d, 
        nx, ny, nz, inv_dx, inv_dy, inv_dz);
}

void curl_x(
    const float_type* face_y_d,
    const float_type* face_z_d,
    float_type* edge_x_d,
    int nx, int ny, int nz,
    float_type inv_dy,
    float_type inv_dz,
    cudaStream_t stream)
{
    LaunchConfig cfg(nx, ny, nz);
    curl_x_kernel<<<cfg.gridDim, cfg.blockDim, 0, stream>>>(
        face_y_d, face_z_d, edge_x_d, nx, ny, nz, inv_dy, inv_dz);
}

void curl_y(
    const float_type* face_x_d,
    const float_type* face_z_d,
    float_type* edge_y_d,
    int nx, int ny, int nz,
    float_type inv_dx,
    float_type inv_dz,
    cudaStream_t stream)
{
    LaunchConfig cfg(nx, ny, nz);
    curl_y_kernel<<<cfg.gridDim, cfg.blockDim, 0, stream>>>(
        face_x_d, face_z_d, edge_y_d, nx, ny, nz, inv_dx, inv_dz);
}

void curl_z(
    const float_type* face_x_d,
    const float_type* face_y_d,
    float_type* edge_z_d,
    int nx, int ny, int nz,
    float_type inv_dx,
    float_type inv_dy,
    cudaStream_t stream)
{
    LaunchConfig cfg(nx, ny, nz);
    curl_z_kernel<<<cfg.gridDim, cfg.blockDim, 0, stream>>>(
        face_x_d, face_y_d, edge_z_d, nx, ny, nz, inv_dx, inv_dy);
}

void nonlinear_x(
    const float_type* face_y_d,
    const float_type* face_z_d,
    const float_type* edge_y_d,
    const float_type* edge_z_d,
    float_type* nl_x_d,
    int nx, int ny, int nz,
    float_type inv_dy,
    float_type inv_dz,
    cudaStream_t stream)
{
    LaunchConfig cfg(nx, ny, nz);
    nonlinear_x_kernel<<<cfg.gridDim, cfg.blockDim, 0, stream>>>(
        face_y_d, face_z_d, edge_y_d, edge_z_d, nl_x_d,
        nx, ny, nz, inv_dy, inv_dz);
}

void laplacian(
    const float_type* field_d,
    float_type* laplacian_d,
    int nx, int ny, int nz,
    float_type inv_dx2,
    float_type inv_dy2,
    float_type inv_dz2,
    cudaStream_t stream)
{
    LaunchConfig cfg(nx, ny, nz);
    laplacian_kernel<<<cfg.gridDim, cfg.blockDim, 0, stream>>>(
        field_d, laplacian_d, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2);
}

} // namespace gpu
} // namespace operators
} // namespace iblgf
