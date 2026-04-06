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

#include <cstddef>
#include <iblgf/types.hpp>

namespace iblgf
{
namespace gpu
{
namespace ops
{

struct block_desc
{
    int block_base[3];
    int block_extent[3];
    int field_base[3];
    int field_extent[3];
    int dim = 3;
};

void laplace_device(const types::float_type* src,
    types::float_type* dst, const block_desc& desc,
    types::float_type dx);

void gradient_device(const types::float_type* src,
    types::float_type* dst0, types::float_type* dst1,
    types::float_type* dst2, const block_desc& desc,
    types::float_type dx, int dim);

void divergence_device(const types::float_type* src0,
    const types::float_type* src1, const types::float_type* src2,
    types::float_type* dst, const block_desc& desc,
    types::float_type dx, int dim);

void curl_device(const types::float_type* src0,
    const types::float_type* src1, const types::float_type* src2,
    types::float_type* dst0, types::float_type* dst1,
    types::float_type* dst2, const block_desc& desc,
    types::float_type dx, int dim);

void curl_transpose_device(const types::float_type* src0,
    const types::float_type* src1, const types::float_type* src2,
    types::float_type* dst0, types::float_type* dst1,
    types::float_type* dst2, const block_desc& desc,
    types::float_type dx, types::float_type scale, int dim);

void nonlinear_device(const types::float_type* face0,
    const types::float_type* face1, const types::float_type* face2,
    const types::float_type* edge0, const types::float_type* edge1,
    const types::float_type* edge2,
    types::float_type* dst0, types::float_type* dst1,
    types::float_type* dst2, const block_desc& desc, int dim);

void smooth2zero_device(types::float_type* field, const block_desc& desc,
    std::size_t ngb_idx);

void zero_device(types::float_type* field, std::size_t count);

void zero_boundary_device(types::float_type* field, const block_desc& desc,
    int width);

void set_constant_field_device(types::float_type* field, const block_desc& desc,
    types::float_type value);

void copy_field_device(const types::float_type* src,
    types::float_type* dst, const block_desc& desc, bool with_buffer);

void scale_field_device(types::float_type* field, const block_desc& desc,
    types::float_type scale);

void axpy_field_device(const types::float_type* src,
    types::float_type* dst, const block_desc& desc, types::float_type scale);

void product_field_device(const types::float_type* a,
    const types::float_type* b, types::float_type* dst,
    const block_desc& desc);

void invert_field_device(types::float_type* field, const block_desc& desc,
    types::float_type eps);

void add_body_force_device(types::float_type* field, const block_desc& desc,
    types::float_type dx, types::float_type scale,
    types::float_type b_f_mag, types::float_type b_f_eps);

} // namespace ops
} // namespace gpu
} // namespace iblgf

#endif
