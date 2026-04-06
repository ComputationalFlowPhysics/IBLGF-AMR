#ifndef IBLGF_INCLUDED_OPERATORS_GPU_HPP
#define IBLGF_INCLUDED_OPERATORS_GPU_HPP

#include <iblgf/types.hpp>

namespace iblgf
{
namespace domain
{
namespace gpu
{
struct FieldView3D
{
    types::float_type* data = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int ox = 0;
    int oy = 0;
    int oz = 0;
};

bool curl_face_to_edge_3d(const FieldView3D& src0, const FieldView3D& src1,
    const FieldView3D& src2, const FieldView3D& dst0,
    const FieldView3D& dst1, const FieldView3D& dst2, int px, int py, int pz,
    types::float_type dx_level) noexcept;

bool divergence_face_to_cell_3d(const FieldView3D& src0,
    const FieldView3D& src1, const FieldView3D& src2, const FieldView3D& dst,
    int px, int py, int pz, types::float_type dx_level) noexcept;

bool gradient_cell_to_face_3d(const FieldView3D& src, const FieldView3D& dst0,
    const FieldView3D& dst1, const FieldView3D& dst2, int px, int py, int pz,
    types::float_type dx_level, types::float_type scale = 1.0) noexcept;

bool nonlinear_face_edge_to_face_3d(const FieldView3D& face0,
    const FieldView3D& face1, const FieldView3D& face2,
    const FieldView3D& edge0, const FieldView3D& edge1,
    const FieldView3D& edge2, const FieldView3D& dst0,
    const FieldView3D& dst1, const FieldView3D& dst2, int px, int py, int pz,
    types::float_type scale = 1.0) noexcept;

} // namespace gpu
} // namespace domain
} // namespace iblgf

#endif
