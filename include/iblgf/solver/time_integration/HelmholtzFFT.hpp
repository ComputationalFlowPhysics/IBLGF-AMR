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

    helm_dfft_r2c(int _padded_dim, int _dim_nonzero, int _dim_0, int _dim_1,
        int _numComp = 3)
    : dim_0(_dim_0)
    , dim_1(_dim_1)
    , numComp(_numComp)
    , padded_dim(_padded_dim)
    , dim_nonzero(_dim_nonzero)
    , input_(_dim_0 * _dim_1 * _padded_dim * _numComp, 0.0)
    , output_(_dim_0 * _dim_1 * (_padded_dim / 2 + 1) * _numComp,
          std::complex<float_type>(0.0))
    {
        std::cout << "init constructor done" << std::endl;
        std::cout << "padded dim is " << padded_dim << std::endl;
        int numTransform = _dim_0 * _dim_1 * _numComp;
        int numCells = _dim_0 * _dim_1;
        int dim_half = _padded_dim / 2 + 1;
        int secDim[1] = {_padded_dim};
        plan = fftw_plan_many_dft_r2c(1, &secDim[0], numTransform, &input_[0],
            NULL, 1, _padded_dim, reinterpret_cast<fftw_complex*>(&output_[0]),
            NULL, 1, dim_half, FFTW_PATIENT);
        std::cout << "constructed r2c" << std::endl;
    }

  public: //Interface
    void         execute() { fftw_execute(plan); }
    inline auto& input() { return input_; }
    inline auto& output() { return output_; }
    inline auto  output_copy() { return output_; }

    template<class Vector>
    void copy_field(const Vector& _v) noexcept
    {
        /*if (_v.size() != input_.size())
                {
                    std::cout
                        << "Number of elements in copying vector for Helmholtz fft does not match in r2c"
                        << std::endl;
                }*/
        //for (int i = 0; i < _v.size(); i++) { input_[i] = _v[i]; }

        int numNonZero = padded_dim * dim_0 * dim_1 * numComp;
        if (_v.size() != numNonZero)
        {
            std::cout
                << "Number of elements in input vector for Helmholtz fft does not match in r2c"
                << std::endl;
            //_v.resize(numNonZero);
        }
        int numTransform = dim_0 * dim_1 * numComp;
        for (int i = 0; i < numTransform; i++)
        {
            for (int j = 0; j < padded_dim; j++)
            {
                int idx_to = i * padded_dim + j;
                int idx_from =
                    i * padded_dim + j;
                input_[idx_from] = _v[idx_to];
            }
        }
    }

    void output_field(std::vector<std::complex<float_type>>& _v) noexcept
    {
        int numNonZero = (dim_nonzero / 2 + 1) * dim_0 * dim_1 * numComp;
        if (_v.size() <= numNonZero)
        {
            std::cout
                << "Number of elements in output vector for Helmholtz fft does not match in r2c"
                << std::endl;
            _v.resize(numNonZero);
        }
        int numTransform = dim_0 * dim_1 * numComp;
        for (int i = 0; i < numTransform; i++)
        {
            for (int j = 0; j < (dim_nonzero / 2 + 1); j++)
            {
                int idx_to = i * (dim_nonzero / 2 + 1) + j;
                int idx_from = i * (padded_dim / 2 + 1) + j;
                _v[idx_to] = output_[idx_from];
            }
        }
    }

    void output_field_neglect_last(std::vector<std::complex<float_type>>& _v) noexcept
    {
        int numNonZero = (dim_nonzero / 2) * dim_0 * dim_1 * numComp;
        if (_v.size() <= numNonZero)
        {
            std::cout
                << "Number of elements in output vector for Helmholtz fft does not match in r2c"
                << std::endl;
            _v.resize(numNonZero);
        }
        int numTransform = dim_0 * dim_1 * numComp;
        for (int i = 0; i < numTransform; i++)
        {
            for (int j = 0; j < (dim_nonzero / 2); j++)
            {
                int idx_to = i * (dim_nonzero / 2) + j;
                int idx_from = i * (padded_dim / 2 + 1) + j;
                _v[idx_to] = output_[idx_from];
            }
        }
    }

  private:
    real_vector_t    input_;
    complex_vector_t output_;

    int padded_dim;
    int dim_nonzero;
    int dim_0;
    int dim_1;
    int numComp;

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

    helm_dfft_c2r(int _padded_dim, int _dim_nonzero, int _dim_0, int _dim_1,
        int _numComp = 3)
    : dim_0(_dim_0)
    , dim_1(_dim_1)
    , numComp(_numComp)
    , padded_dim(_padded_dim)
    , dim_nonzero(_dim_nonzero)
    , input_(_dim_0 * _dim_1 * (_padded_dim / 2 + 1) * _numComp,
          std::complex<float_type>(0.0))
    , output_(_dim_0 * _dim_1 * _padded_dim * _numComp, 0.0)
    {
        int numTransform = _dim_0 * _dim_1 * _numComp;
        int numCells = dim_0 * dim_1;
        int dim_half = _padded_dim / 2 + 1;
        int secDim[1] = {padded_dim};
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
        /*if (_v.size() != input_.size())
                {
                    std::cout
                        << "Number of elements in copying vector for Helmholtz fft does not match in c2r"
                        << std::endl;
                }*/
        //for (int i = 0; i < _v.size(); i++) { input_[i] = _v[i]; }

        int numNonZero = (dim_nonzero / 2 + 1) * dim_0 * dim_1 * numComp;
        if (_v.size() != numNonZero)
        {
            std::cout
                << "Number of elements in input vector for Helmholtz fft does not match in c2r"
                << std::endl;
            //_v.resize(numNonZero);
        }
        int numTransform = dim_0 * dim_1 * numComp;
        for (int i = 0; i < numTransform; i++)
        {
            for (int j = 0; j < (dim_nonzero / 2 + 1); j++)
            {
                int idx_to = i * (dim_nonzero / 2 + 1) + j;
                int idx_from = i * (padded_dim / 2 + 1) + j;
                input_[idx_from] = _v[idx_to];
            }
        }
    }

    void output_field(std::vector<float_type>& _v) noexcept
    {
        int numNonZero = dim_nonzero * dim_0 * dim_1 * numComp;
        if (_v.size() <= numNonZero)
        {
            std::cout
                << "Number of elements in output vector for Helmholtz fft does not match in c2r"
                << std::endl;
            _v.resize(numNonZero);
        }
        int numTransform = dim_0 * dim_1 * numComp;
        for (int i = 0; i < numTransform; i++)
        {
            for (int j = 0; j < dim_nonzero; j++)
            {
                int idx_to = i * dim_nonzero + j;
                int idx_from = i * padded_dim + j;
                _v[idx_to] = output_[idx_from];
            }
        }
    }

    void output_field_padded(std::vector<float_type>& _v) noexcept
    {
        int Padded_Num = padded_dim * dim_0 * dim_1 * numComp;
        if (_v.size() <= Padded_Num)
        {
            /*std::cout
                << "Number of elements in output vector for Helmholtz fft does not match in c2r"
                << std::endl;*/
            _v.resize(Padded_Num);
        }
        for (int i = 0; i < output_.size(); i++) { _v[i] = output_[i]; }
    }

  private:
    complex_vector_t input_;
    real_vector_t    output_;

    int padded_dim;
    int dim_nonzero;
    int dim_0;
    int dim_1;
    int numComp;

    fftw_plan plan;
};

} // namespace fft
} // namespace iblgf
#endif
