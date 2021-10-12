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

#ifndef INCLUDED_HELMHOLTZ_FFT_HPP
#define INCLUDED_HELMHOLTZ_FFT_HPP

#include <complex>
#include <cassert>
#include <cstring>
#include <vector>
#include <map>
#include <complex>
#include <fftw3.h>
#include <iblgf/utilities/misc_math_functions.hpp>

namespace iblgf
{
namespace fft
{
class helm_dfft_r2c
{
  public:
    using float_type = double;
    using complex_vector_t = std::vector<std::complex<float_type>,
        xsimd::aligned_allocator<std::complex<float_type>, 32>>;
    using real_vector_t =
        std::vector<float_type, xsimd::aligned_allocator<float_type, 32>>;

  public: //Ctors:
    helm_dfft_r2c(const helm_dfft_r2c& other) = default;
    helm_dfft_r2c(helm_dfft_r2c&& other) = default;
    helm_dfft_r2c& operator=(const helm_dfft_r2c& other) & = default;
    helm_dfft_r2c& operator=(helm_dfft_r2c&& other) & = default;
    ~helm_dfft_r2c() { fftw_destroy_plan(plan); }

    /*helm_dfft_r2c(int _padded_dim, int _dim_nonzero, dims_2D _block_dim,
        int numComp = 3),
        input_(_block_dim[1] * _block_dim[0] * _padded_dim * numComp, 0.0),
        output_(_block_dim[1] * _block_dim[0] * (_padded_dim / 2 + 1) * numComp)
    {
        int numTransform = _block_dim[0] * _block_dim[1] * numComp;
        int dim_half = _padded_dim / 2 + 1;
        plan = fftw_plan_many_dft_r2c(1, numTransform, input_, NULL, 1,
            _padded_dim, reinterpret_cast<fftw_complex*>(&output_[0]), NULL, 1,
            dim_half, FFTW_PATIENT);
    }*/

    helm_dfft_r2c(int _padded_dim, int _dim_nonzero, int dim_0, int dim_1,
        int numComp = 3)
    : input_(dim_0 * dim_1 * _padded_dim * numComp, 0.0)
    , output_(dim_0 * dim_1 * (_padded_dim / 2 + 1) * numComp,
          std::complex<float_type>(0.0))
    {
        int numTransform = dim_0 * dim_1 * numComp;
        int dim_half = _padded_dim / 2 + 1;
        int secDim[1] = {_padded_dim};
        plan = fftw_plan_many_dft_r2c(1, &secDim[0], numTransform, &input_[0], NULL, 1,
            _padded_dim, reinterpret_cast<fftw_complex*>(&output_[0]), NULL, 1,
            dim_half, FFTW_PATIENT);
    }

  public: //Interface
    void         execute() { fftw_execute(plan); }
    inline auto& input() { return input_; }
    inline auto& output() { return output_; }
    inline auto  output_copy() { return output_; }

    template<class Vector>
    void copy_field(const Vector& _v) noexcept
    {
        if (_v.size() != input_.size())
        {
            std::cout
                << "Number of elements in copying vector for Helmholtz fft does not match in r2c"
                << std::endl;
        }
        for (int i = 0; i < _v.size(); i++) { input_[i] = _v[i]; }
    }

    void output_field(std::vector<std::complex<float_type>>& _v) noexcept
    {
        if (_v.size() != output_.size())
        {
            std::cout
                << "Number of elements in output vector for Helmholtz fft does not match in r2c"
                << std::endl;
            _v.resize(output_.size());
        }
        for (int i = 0; i < _v.size(); i++) { _v[i] = output_[i]; }
    }

  private:
    real_vector_t    input_;
    complex_vector_t output_;

    fftw_plan plan;
};

class helm_dfft_c2r
{
  public:
    using float_type = double;
    using complex_vector_t = std::vector<std::complex<float_type>,
        xsimd::aligned_allocator<std::complex<float_type>, 32>>;
    using real_vector_t =
        std::vector<float_type, xsimd::aligned_allocator<float_type, 32>>;

  public: //Ctors:
    helm_dfft_c2r(const helm_dfft_c2r& other) = default;
    helm_dfft_c2r(helm_dfft_c2r&& other) = default;
    helm_dfft_c2r& operator=(const helm_dfft_c2r& other) & = default;
    helm_dfft_c2r& operator=(helm_dfft_c2r&& other) & = default;
    ~helm_dfft_c2r() { fftw_destroy_plan(plan); }

    helm_dfft_c2r(int _padded_dim, int _dim_nonzero, int dim_0, int dim_1,
        int numComp = 3)
    : input_(dim_0 * dim_1 * (_padded_dim / 2 + 1) * numComp,
          std::complex<float_type>(0.0))
    , output_(dim_0 * dim_1 * _padded_dim * numComp, 0.0)
    {
        int numTransform = dim_0 * dim_1 * numComp;
        int dim_half = _padded_dim / 2 + 1;
        int secDim[1] = {_padded_dim};
        plan = fftw_plan_many_dft_c2r(1, &secDim[0], numTransform,
            reinterpret_cast<fftw_complex*>(&input_[0]), NULL, 1, dim_half,
            &output_[0], NULL, 1, _padded_dim, FFTW_PATIENT);
    }

  public: //Interface
    void         execute() { fftw_execute(plan); }
    inline auto& input() { return input_; }
    inline auto& output() { return output_; }
    inline auto  output_copy() { return output_; }

    template<class Vector>
    void copy_field(const Vector& _v) noexcept
    {
        if (_v.size() != input_.size())
        {
            std::cout
                << "Number of elements in copying vector for Helmholtz fft does not match in c2r"
                << std::endl;
        }
        for (int i = 0; i < _v.size(); i++) { input_[i] = _v[i]; }
    }

    void output_field(std::vector<float_type>& _v) noexcept
    {
        if (_v.size() != output_.size())
        {
            std::cout
                << "Number of elements in output vector for Helmholtz fft does not match in c2r"
                << std::endl;
            _v.resize(output_.size());
        }
        for (int i = 0; i < _v.size(); i++) { _v[i] = output_[i]; }
    }

  private:
    complex_vector_t    input_;
    real_vector_t       output_;

    fftw_plan plan;
};

} // namespace fft
} // namespace iblgf
#endif
