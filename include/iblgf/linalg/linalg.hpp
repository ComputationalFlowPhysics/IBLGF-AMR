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

#ifndef IBLGF_INCLUDED_LINALG_HPP
#define IBLGF_INCLUDED_LINALG_HPP

#define XTENSOR_USE_XSIMD
#include <iostream>
#include <vector>
#include <tuple>

#include <iblgf/types.hpp>
#include <iblgf/linalg/linalg_xtensor.hpp>

namespace iblgf
{
namespace linalg
{
template<class Policy>
struct Cube : public Policy
{
  public:
    Cube() = default;
    Cube(const Cube& rhs) = delete;
    Cube& operator=(const Cube&) & = delete;

    Cube(types::float_type* ptr_aux_mem, size_t n_rows, size_t n_cols,
        size_t n_slices)
    : data_(Policy::cube_wrap(ptr_aux_mem, n_rows, n_cols, n_slices))
    {
    }

    Cube(types::float_type* ptr_aux_mem, size_t n_rows, size_t n_cols)
    : data_(Policy::cube_wrap(ptr_aux_mem, n_rows, n_cols))
    {
    }

    auto cube_noalias_view() { return Policy::cube_noalias_view(data_); }

    template<typename cube_t, typename stride_t_1, typename stride_t_2,
        typename stride_t_3>
    auto cube_noalias_view(
        cube_t& cube, stride_t_1 x1, stride_t_2 x2, stride_t_3 x3)
    {
        return Policy::cube_noalias_view(cube, x1, x2, x3);
    }


    template<typename cube_t, typename stride_t_1, typename stride_t_2>
    auto cube_noalias_view(
        cube_t& cube, stride_t_1 x1, stride_t_2 x2)
    {
        return Policy::cube_noalias_view(cube, x1, x2);
    }

    // member
    types::float_type                          tmp;
    decltype(Policy::cube_wrap(&tmp, 2, 2, 2)) data_;
};

template<class Policy>
struct Mat : public Policy
{
  public:
    Mat() = default;

    Mat(types::float_type* ptr_aux_mem, int n_rows, int n_cols)
    : data_(Policy::mat_wrap(ptr_aux_mem, n_rows, n_cols))
    {
    }

    decltype(Policy::mat_wrap(0, 1, 1)) data_;
};

using Cube_t = Cube<L_xtensor>;
using Mat_t = Mat<L_xtensor>;

} // namespace linalg
} // namespace iblgf
#endif
