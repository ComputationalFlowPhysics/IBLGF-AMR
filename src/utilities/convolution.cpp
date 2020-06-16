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

#include <xsimd/xsimd.hpp>
#include <xsimd/stl/algorithms.hpp>

#include <iblgf/global.hpp>
#include <iblgf/utilities/misc_math_functions.hpp>
#include <iblgf/utilities/convolution.hpp>

namespace fft
{
//using namespace domain;

dfft_r2c::dfft_r2c(dims_t _dims_padded, dims_t _dims_non_zero)
: dims_input_(_dims_padded)
, input_(_dims_padded[2] * _dims_padded[1] * _dims_padded[0], 0.0)
, output_1(_dims_padded[2] * _dims_padded[1] * ((_dims_padded[0] / 2) + 1))
, output_2(_dims_padded[2] * _dims_padded[1] * ((_dims_padded[0] / 2) + 1))
, output_(_dims_padded[2] * _dims_padded[1] * ((_dims_padded[0] / 2) + 1))
{
    plan = (fftw_plan_dft_r2c_3d(_dims_padded[2], _dims_padded[1],
        _dims_padded[0], &input_[0],
        reinterpret_cast<fftw_complex*>(&output_[0]), FFTW_PATIENT));

    r2c_1d_plans.resize(_dims_non_zero[0]);

    int dim_half = (_dims_padded[2] / 2) + 1;

    for (int i_plan = 0; i_plan < _dims_non_zero[0]; ++i_plan)
    {
        r2c_1d_plans[i_plan] =
            fftw_plan_many_dft_r2c(1, &_dims_padded[2], _dims_non_zero[1],
                &input_[i_plan * _dims_padded[2] * _dims_padded[1]], NULL, 1,
                _dims_padded[2],
                reinterpret_cast<fftw_complex*>(
                    &output_1[i_plan * dim_half * _dims_padded[1]]),
                NULL, 1, dim_half, FFTW_PATIENT);
    }

    c2c_1d_plans_dir_2.resize(_dims_non_zero[0]);
    for (int i_plan = 0; i_plan < _dims_non_zero[0]; ++i_plan)
    {
        c2c_1d_plans_dir_2[i_plan] =
            fftw_plan_many_dft(1, &_dims_padded[1], dim_half,
                reinterpret_cast<fftw_complex*>(
                    &output_1[i_plan * dim_half * _dims_padded[1]]),
                NULL, dim_half, 1,
                reinterpret_cast<fftw_complex*>(
                    &output_2[i_plan * dim_half * _dims_padded[1]]),
                NULL, dim_half, 1, FFTW_FORWARD, FFTW_PATIENT);
    }

    c2c_1d_plans_dir_3 =
        fftw_plan_many_dft(1, &_dims_padded[0], dim_half * _dims_padded[1],
            reinterpret_cast<fftw_complex*>(&output_2[0]), NULL,
            dim_half * _dims_padded[1], 1,
            reinterpret_cast<fftw_complex*>(&output_[0]), NULL,
            dim_half * _dims_padded[1], 1, FFTW_FORWARD, FFTW_PATIENT);
}

void
dfft_r2c::execute_whole()
{
    fftw_execute(plan);
}

void
dfft_r2c::execute()
{
    //Fisrt direction
    for (std::size_t i = 0; i < r2c_1d_plans.size(); ++i)
        fftw_execute(r2c_1d_plans[i]);

    ////Second direction
    for (std::size_t i = 0; i < c2c_1d_plans_dir_2.size(); ++i)
        fftw_execute(c2c_1d_plans_dir_2[i]);

    fftw_execute(c2c_1d_plans_dir_3);
}

template<class Vector>
void
dfft_r2c::copy_input(const Vector& _v, dims_t _dims_v)
{
    if (_v.size() == input_.size())
    { std::copy(_v.begin(), _v.end(), input_.begin()); }
    else
    {
        throw std::runtime_error("ERROR! LGF SIZE NOT MATCHING");
    }
}
template void dfft_r2c::copy_input<std::vector<double>>(
    const std::vector<double>& _v, dims_t _dims_v);

/******************************************************************************/

dfft_c2r::dfft_c2r(dims_t _dims, dims_t _dims_small)
: input_(
      _dims[2] * _dims[1] * ((_dims[0] / 2) + 1), std::complex<float_type>(0.0))
, tmp_1_(
      _dims[2] * _dims[1] * ((_dims[0] / 2) + 1), std::complex<float_type>(0.0))
, tmp_2_(
      _dims[2] * _dims[1] * ((_dims[0] / 2) + 1), std::complex<float_type>(0.0))
, output_(_dims[2] * _dims[1] * _dims[0], 0.0)
{
    //int status = fftw_init_threads();
    //fftw_plan_with_nthreads(nthreads);
    plan = fftw_plan_dft_c2r_3d(_dims[2], _dims[1], _dims[0],
        reinterpret_cast<fftw_complex*>(&input_[0]), &output_[0], FFTW_PATIENT);

    int dim_half = (_dims[2] / 2) + 1;
    c2c_dir_1 = fftw_plan_many_dft(1, &_dims[0], _dims[1] * dim_half,
        reinterpret_cast<fftw_complex*>(&input_[0]), NULL, _dims[1] * dim_half,
        1, reinterpret_cast<fftw_complex*>(&tmp_1_[0]), NULL,
        _dims[1] * dim_half, 1, FFTW_BACKWARD, FFTW_PATIENT);

    ////Dir 1
    c2c_dir_2.resize(_dims_small[0]);
    for (int i_plan = 0; i_plan < _dims_small[0]; ++i_plan)
    {
        c2c_dir_2[i_plan] = fftw_plan_many_dft(1, &_dims[1], dim_half,
            reinterpret_cast<fftw_complex*>(
                &tmp_1_[(i_plan + _dims_small[0] - 1) * dim_half * _dims[1]]),
            NULL, dim_half, 1,
            reinterpret_cast<fftw_complex*>(
                &tmp_2_[(i_plan + _dims_small[0] - 1) * dim_half * _dims[1]]),
            NULL, dim_half, 1, FFTW_BACKWARD, FFTW_PATIENT);
    }

    //// Dir 2
    c2r_dir_3.resize(_dims_small[0]);
    for (int i_plan = 0; i_plan < _dims_small[0]; ++i_plan)
    {
        c2r_dir_3[i_plan] = fftw_plan_many_dft_c2r(1, &_dims[2], _dims_small[1],
            reinterpret_cast<fftw_complex*>(
                &tmp_2_[(i_plan + _dims_small[0] - 1) * dim_half * _dims[1] +
                        dim_half * (_dims_small[1] - 1)]),
            NULL, 1, dim_half,
            &output_[(i_plan + _dims_small[0] - 1) * _dims[2] * _dims[1] +
                     _dims[2] * (_dims_small[1] - 1)],
            NULL, 1, _dims[2], FFTW_PATIENT);
    }
}

void
dfft_c2r::execute()
{
    fftw_execute(c2c_dir_1);

    for (std::size_t i = 0; i < c2c_dir_2.size(); ++i)
        fftw_execute(c2c_dir_2[i]);

    for (std::size_t i = 0; i < c2r_dir_3.size(); ++i)
        fftw_execute(c2r_dir_3[i]);
}

/******************************************************************************/

Convolution::Convolution(dims_t _dims0, dims_t _dims1)
: padded_dims_(_dims0 + _dims1 - 1)
, padded_dims_next_pow_2_({math::next_pow_2(padded_dims_[0]),
      math::next_pow_2(padded_dims_[1]), math::next_pow_2(padded_dims_[2])})
, dims0_(_dims0)
, dims1_(_dims1)
, fft_forward0_(padded_dims_next_pow_2_, _dims0)
, fft_forward1_(padded_dims_next_pow_2_, _dims1)
, fft_backward_(padded_dims_next_pow_2_, _dims1)
, padded_size_(padded_dims_next_pow_2_[0] * padded_dims_next_pow_2_[1] *
               padded_dims_next_pow_2_[2])
, tmp_prod(padded_size_, std::complex<float_type>(0.0))
{
}

void
Convolution::simd_prod_complex_add(
    const complex_vector_t& a, const complex_vector_t& b, complex_vector_t& res)
{
    std::size_t           size = a.size();
    constexpr std::size_t simd_size =
        xsimd::simd_type<std::complex<float_type>>::size;
    std::size_t vec_size = size - size % simd_size;

    for (std::size_t i = 0; i < vec_size; i += simd_size)
    {
        auto ba = xsimd::load_aligned(&a[i]);
        auto bb = xsimd::load_aligned(&b[i]);
        auto res_old = xsimd::load_aligned(&res[i]);
        auto bres = ba * bb + res_old;
        bres.store_aligned(&res[i]);
    }
    for (std::size_t i = vec_size; i < size; ++i) { res[i] += a[i] * b[i]; }
}
Convolution::complex_vector_t&
Convolution::dft_r2c(std::vector<float_type>& _vec)
{
    fft_forward0_.copy_input(_vec, dims0_);
    fft_forward0_.execute_whole();
    return fft_forward0_.output();
}
void
Convolution::fft_backward_field_clean()
{
    std::fill(fft_backward_.input().begin(), fft_backward_.input().end(), 0);
}

} //namespace fft

