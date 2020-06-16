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

#include <iostream>
#include <algorithm>

#include "xtensor/xnoalias.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

namespace linalg
{
class L_xtensor
{
  public:
    using tensor_t = typename xt::xarray<float_type>;
    using cube_t = tensor_t;
    using mat_t = tensor_t;

    static auto cube_wrap(types::float_type* ptr_aux_mem, size_t n_rows,
        size_t n_cols, size_t n_slices)
    {
        int size = n_rows * n_cols * n_slices;
        return xt::adapt<xt::layout_type::column_major>(ptr_aux_mem, size,
            xt::no_ownership(),
            std::vector<std::size_t>{{n_rows, n_cols, n_slices}});
    }

    template<typename cube_t>
    static auto cube_noalias_view(cube_t& cube)
    {
        return xt::noalias(cube);
    }

    template<typename cube_t, typename stride_t_1, typename stride_t_2,
        typename stride_t_3>
    static auto cube_noalias_view(
        cube_t& cube, stride_t_1 x1, stride_t_2 x2, stride_t_3 x3)
    {
        return xt::noalias(xt::view(cube, x1, x2, x3));
    }

    static auto mat_wrap(
        types::float_type* ptr_aux_mem, size_t n_rows, size_t n_cols)
    {
        int size = n_rows * n_cols;
        return xt::adapt<xt::layout_type::column_major>(ptr_aux_mem, size,
            xt::no_ownership(), std::vector<std::size_t>{{n_rows, n_cols}});
    }

    static auto mat_col(tensor_t& data_, int n)
    {
        return xt::view(data_, xt::all(), n);
    }

    template<typename cube_t, typename stride_t_1, typename stride_t_2>
    static auto mat_noalias_view(cube_t cube, stride_t_1 x1, stride_t_2 x2)
    {
        return xt::noalias(xt::view(cube, x1, x2));
    }
};

} // namespace linalg

