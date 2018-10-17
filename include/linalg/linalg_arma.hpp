#include <types.hpp>
#include <armadillo>


namespace linalg
{

class L_armadillo
{

public:

    //typename cube_t = typename arma::cube;
    using cube_t = typename arma::cube;
    using mat_t  = typename arma::mat;
    using span = arma::span;

    static auto cube_wrap(types::float_type* ptr_aux_mem, int n_rows, int n_cols, int n_slices)
    {
        return arma::cube(ptr_aux_mem, n_rows, n_cols, n_slices, false, true );
    }

    static auto mat_wrap(types::float_type* ptr_aux_mem, int n_rows, int n_cols)
    {
        return arma::mat(ptr_aux_mem, n_rows, n_cols, false, true );
    }

    static auto submatrix(mat_t& data_, std::tuple<int,int> nx, std::tuple<int,int> ny)
    {
        return data_(arma::span(std::get<0>(nx),std::get<1>(nx)) , arma::span(std::get<0>(ny),std::get<1>(ny))) ;
    }

private:

};

}

