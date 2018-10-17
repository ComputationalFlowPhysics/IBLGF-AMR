#ifndef IBLGF_INCLUDED_LINALG_HPP
#define IBLGF_INCLUDED_LINALG_HPP

#include <iostream>
#include <vector>
#include <tuple>

#include <types.hpp>
//#include <linalg/linalg_arma.hpp>
#include <linalg/linalg_xtensor.hpp>

namespace linalg
{

template<class Policy>
struct Cube : public Policy
{

public:
    Cube() = default;
    Cube(const Cube& rhs)=delete;
	Cube& operator=(const Cube&) & = delete ;

    Cube(types::float_type* ptr_aux_mem, auto  n_rows, auto n_cols, auto n_slices)
        : data_(Policy::cube_wrap(ptr_aux_mem, n_rows, n_cols, n_slices))
    {}

    auto cube_noalias_view(auto cube, auto x1, auto x2, auto x3)
    {
        return Policy::cube_noalias_view(cube, x1, x2, x3);
    }

    // member
    types::float_type tmp;
    decltype(Policy::cube_wrap(&tmp, 1, 1, 1)) data_;

};

template<class Policy>
struct Mat : public Policy
{

public:
    Mat() = default;

    Mat(types::float_type* ptr_aux_mem, int n_rows, int n_cols)
        :data_(Policy::mat_wrap(ptr_aux_mem, n_rows, n_cols))
    {}

    //auto submatrix( std::tuple<int,int> nx, std::tuple<int,int> ny)
    //{
    //    return Policy::submatrix(data_, nx, ny);
    //}

    //auto col(int n )
    //{
    //    return Policy::col(data_, n);
    //}

    //auto Mat::operator()(int i, int j)
    //{
    //    return data_(i,j)
    //}

    // member
    //typename Policy::mat_t data_;
    decltype(Policy::mat_wrap(0, 1, 1)) data_;

};


using Cube_t = Cube<L_xtensor>;
using Mat_t  = Mat <L_xtensor>;

}
#endif
