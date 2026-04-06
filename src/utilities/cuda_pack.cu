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
#include <iblgf/utilities/cuda_pack.hpp>
#include <iblgf/types.hpp>

namespace iblgf
{
namespace gpu
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

__global__ void pack_view_kernel(const types::float_type* field_device,
    block_view_desc desc, types::float_type* out_device, std::size_t count)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const int ex = desc.view_extent[0];
    const int ey = desc.view_extent[1];

    const int i = static_cast<int>(idx % static_cast<std::size_t>(ex));
    const int j = static_cast<int>((idx / static_cast<std::size_t>(ex)) %
                                   static_cast<std::size_t>(ey));
    const int k = static_cast<int>(idx / (static_cast<std::size_t>(ex) *
                                          static_cast<std::size_t>(ey)));

    const int x = desc.view_base[0] + i * desc.view_stride[0];
    const int y = desc.view_base[1] + j * desc.view_stride[1];
    const int z = desc.view_base[2] + k * desc.view_stride[2];

    const std::size_t fidx =
        index_3d(x, y, z, desc.field_base, desc.field_extent);
    out_device[idx] = field_device[fidx];
}

__global__ void unpack_view_kernel(const types::float_type* in_device,
    block_view_desc desc, types::float_type* field_device, std::size_t count)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const int ex = desc.view_extent[0];
    const int ey = desc.view_extent[1];

    const int i = static_cast<int>(idx % static_cast<std::size_t>(ex));
    const int j = static_cast<int>((idx / static_cast<std::size_t>(ex)) %
                                   static_cast<std::size_t>(ey));
    const int k = static_cast<int>(idx / (static_cast<std::size_t>(ex) *
                                          static_cast<std::size_t>(ey)));

    const int x = desc.view_base[0] + i * desc.view_stride[0];
    const int y = desc.view_base[1] + j * desc.view_stride[1];
    const int z = desc.view_base[2] + k * desc.view_stride[2];

    const std::size_t fidx =
        index_3d(x, y, z, desc.field_base, desc.field_extent);
    field_device[fidx] = in_device[idx];
}
} // namespace

void pack_view_device_to_device(const types::float_type* field_device,
    const block_view_desc& desc, types::float_type* out_device, std::size_t count)
{
    if (count == 0) return;
    constexpr int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    pack_view_kernel<<<grid, block>>>(field_device, desc, out_device, count);
}

void unpack_view_device_to_device(const types::float_type* in_device,
    const block_view_desc& desc, types::float_type* field_device, std::size_t count)
{
    if (count == 0) return;
    constexpr int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    unpack_view_kernel<<<grid, block>>>(in_device, desc, field_device, count);
}

} // namespace gpu
} // namespace iblgf
