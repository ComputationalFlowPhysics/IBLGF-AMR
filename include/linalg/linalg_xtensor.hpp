#include <iostream>
#include <algorithm>

#include "xtensor/xnoalias.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include "xtensor-blas/xlinalg.hpp"

namespace linalg
{

class L_xtensor
{

public:

    using tensor_t = typename xt::xarray<float_type>;
    using cube_t   = tensor_t;
    using mat_t    = tensor_t;

    static auto cube_wrap(types::float_type* ptr_aux_mem,
            auto n_rows, auto n_cols, auto n_slices)
    {
        int size = n_rows * n_cols * n_slices;
        return xt::adapt(ptr_aux_mem, size, xt::no_ownership(),
                std::vector<std::size_t>{{n_rows, n_cols, n_slices}});
    }

    static auto cube_noalias_view(auto cube, auto x1, auto x2, auto x3)
    {
        return xt::noalias( xt::view(cube, x1, x2, x3) );
    }

    static auto mat_wrap(types::float_type* ptr_aux_mem,
            size_t n_rows, size_t n_cols)
    {
        int size = n_rows * n_cols;
        return (xt::adapt(ptr_aux_mem, size, xt::no_ownership(),
                std::vector<std::size_t>{{n_rows, n_cols}}));
    }

    static auto mat_col(tensor_t& data_, int n)
    {
        return xt::view(data_, xt::all(), n);
    }

    static auto mat_noalias_view(auto cube, auto x1, auto x2)
    {
        return xt::noalias( xt::view(cube, x1, x2) );
    }


};

}


