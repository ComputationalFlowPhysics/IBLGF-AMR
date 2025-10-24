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

#ifndef INCLUDED_CONVOLUTION_IBLGF_HPP
#define INCLUDED_CONVOLUTION_IBLGF_HPP

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
class dfft_r2c
{
  public:
    using float_type = double;
    using complex_vector_t = std::vector<std::complex<float_type>,
        xsimd::aligned_allocator<std::complex<float_type>, 32>>;
    using real_vector_t =
        std::vector<float_type, xsimd::aligned_allocator<float_type, 32>>;
    using dims_3D = types::vector_type<int, 3>;
    using dims_2D = types::vector_type<int, 2>;

  public: //Ctors:
    dfft_r2c(const dfft_r2c& other) = default;
    dfft_r2c(dfft_r2c&& other) = default;
    dfft_r2c& operator=(const dfft_r2c& other) & = default;
    dfft_r2c& operator=(dfft_r2c&& other) & = default;
    ~dfft_r2c() { fftw_destroy_plan(plan); }

    dfft_r2c(dims_3D _dims_padded, dims_3D _dims_non_zero);
    dfft_r2c(dims_2D _dims_padded, dims_2D _dims_non_zero);

  public: //Interface
    void         execute_whole();
    void         execute();
    inline auto& input() { return input_; }
    inline auto& output() { return output_; }
    inline auto  output_copy() { return output_; }

    template<class Vector>
    void copy_input(const Vector& _v, dims_3D _dims_v);

    template<class Vector>
    void copy_input(const Vector& _v, dims_2D _dims_v);

    template<class Vector>
    void copy_field(const Vector& _v, dims_3D _dims_v) noexcept
    {
        for (int k = 0; k < _dims_v[2]; ++k)
        {
            for (int j = 0; j < _dims_v[1]; ++j)
            {
                for (int i = 0; i < _dims_v[0]; ++i)
                {
                    input_[i + dims_input_3D[0] * j +
                           dims_input_3D[0] * dims_input_3D[1] * k] =
                        _v.get_real_local(i, j, k);
                }
            }
        }
    }

    template<class Vector>
    void copy_field(const Vector& _v, dims_2D _dims_v) noexcept
    {
        for (int j = 0; j < _dims_v[1]; ++j)
        {
            for (int i = 0; i < _dims_v[0]; ++i)
            { input_[i + dims_input_2D[0] * j] = _v.get_real_local(i, j); }
        }
    }

  private:
    dims_3D          dims_input_3D;
    dims_2D          dims_input_2D;
    real_vector_t    input_;
    complex_vector_t output_1, output_2, output_;

    fftw_plan plan;

    std::vector<fftw_plan> r2c_1d_plans;
    std::vector<fftw_plan> c2c_1d_plans_dir_2;
    fftw_plan              c2c_1d_plans_dir_3;
};

class dfft_c2r
{
  public:
    //const int nthreads = 1;
    using complex_vector_t = std::vector<std::complex<float_type>,
        xsimd::aligned_allocator<std::complex<float_type>, 32>>;
    using real_vector_t =
        std::vector<float_type, xsimd::aligned_allocator<float_type, 32>>;
    using dims_3D = types::vector_type<int, 3>;
    using dims_2D = types::vector_type<int, 2>;

  public: //Ctors:
    dfft_c2r(const dfft_c2r& other) = default;
    dfft_c2r(dfft_c2r&& other) = default;
    dfft_c2r& operator=(const dfft_c2r& other) & = default;
    dfft_c2r& operator=(dfft_c2r&& other) & = default;
    ~dfft_c2r() { fftw_destroy_plan(plan); }

    dfft_c2r(dims_3D _dims, dims_3D _dims_small);
    dfft_c2r(dims_2D _dims, dims_2D _dims_small);

  public: //Interface
    void         execute();
    inline auto& input() { return input_; }
    inline auto& output() { return output_; }

  private:
    complex_vector_t input_, tmp_1_, tmp_2_;
    real_vector_t    output_;

    fftw_plan              plan, c2c_dir_1;
    std::vector<fftw_plan> c2c_dir_2, c2r_dir_3;
};

///template<class Kernel>
template<int Dim>
class Convolution
{
  public:
    static constexpr int dimension = Dim;
    using dims_t = types::vector_type<int, Dim>;
    using complex_vector_t = std::vector<std::complex<float_type>,
        xsimd::aligned_allocator<std::complex<float_type>, 32>>;
    using real_vector_t =
        std::vector<float_type, xsimd::aligned_allocator<float_type, 32>>;
    //using lgf_key_t = std::tuple<int, int, int>;
    /*using lgf_matrix_ptr_map_type =
        std::map<lgf_key_t, std::unique_ptr<complex_vector_t>>;*/

  public: //Ctors
    Convolution(const Convolution& other) = default;
    Convolution(Convolution&& other) = default;
    Convolution& operator=(const Convolution& other) & = default;
    Convolution& operator=(Convolution&& other) & = default;
    ~Convolution() = default;

    //Convolution(dims_t _dims0, dims_t _dims1);
    Convolution(dims_t _dims0, dims_t _dims1)
    : padded_dims_(_dims0 + _dims1 - 1)
    , padded_dims_next_pow_2_(helper_next_pow_2(padded_dims_))
    , dims0_(_dims0)
    , dims1_(_dims1)
    , fft_forward0_(padded_dims_next_pow_2_, _dims0)
    , fft_forward1_(padded_dims_next_pow_2_, _dims1)
    , fft_backward_(padded_dims_next_pow_2_, _dims1)
    , padded_size_(helper_all_prod(padded_dims_next_pow_2_))
    , tmp_prod(padded_size_, std::complex<float_type>(0.0))
    {
    }

    dims_t helper_next_pow_2(dims_t v)
    {
        dims_t tmp;
        for (int i = 0; i < dimension; i++) { tmp[i] = v[i]; }
        return tmp;
    }
    int helper_all_prod(dims_t v)
    {
        int tmp = 1;
        for (int i = 0; i < dimension; i++) { tmp *= v[i]; }
        return tmp;
    }

  public: //Members:
    complex_vector_t& dft_r2c(std::vector<float_type>& _vec)
    {
        fft_forward0_.copy_input(_vec, dims0_);
        fft_forward0_.execute_whole();
        return fft_forward0_.output();
    }

    void fft_backward_field_clean()
    {
        number_fwrd_executed = 0;
        std::fill(
            fft_backward_.input().begin(), fft_backward_.input().end(), 0);
    }

    void simd_prod_complex_add(const complex_vector_t& a,
        const complex_vector_t& b, complex_vector_t& res)
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

    auto& output() { return fft_backward_.output(); }

    template<typename Source, typename BlockType, class Kernel>
    void apply_forward_add(const BlockType& _lgf_block, Kernel* _kernel,
        int _level_diff, const Source& _source)
    {
        execute_fwrd_field(_lgf_block, _kernel, _level_diff, _source);
    }

    template<typename Target, typename BlockType>
    void apply_backward(
        const BlockType& _extractor, Target& _target, float_type _extra_scale)
    {
        if (number_fwrd_executed == 0) return;
        float_type scale = 1.0;
        for (int i = 0; i < dimension; i++)
        { scale /= padded_dims_next_pow_2_[i]; }
        scale *= _extra_scale;
        for (std::size_t i = 0; i < fft_backward_.input().size(); ++i)
        { fft_backward_.input()[i] *= scale; }

        fft_backward_.execute();
        add_solution(_extractor, _target);
    }

    template<class Field, class BlockType, class Kernel>
    void execute_fwrd_field(const BlockType& _lgf_block, Kernel* _kernel,
        int _level_diff, const Field& _b)
    {
        
        auto& f0 = _kernel->dft(
            _lgf_block, padded_dims_next_pow_2_, this, _level_diff); // gets dft of LGF, checks if already computed

        if (f0.size() == 0) return;
        number_fwrd_executed++;

        fft_forward1_.copy_field(_b, dims1_);
        fft_forward1_.execute(); // forward DFT of source
        auto& f1 = fft_forward1_.output();

        complex_vector_t prod(f0.size());
        simd_prod_complex_add(f0, f1, fft_backward_.input()); // convolution by multiplication in Fourier space, placed in backward input
        // all induced fields are accumulated in the backward input and then one backward FFT is performed later in fmm.compute_influence_field()
    }

    template<class Block, class Field>
    void add_solution(const Block& _b, Field& _F)
    {
        if (dimension == 3)
        {
            for (int k = dims0_[2] - 1; k < dims0_[2] + _b.extent()[2] - 1; ++k)
            {
                for (int j = dims0_[1] - 1; j < dims0_[1] + _b.extent()[1] - 1;
                     ++j)
                {
                    for (int i = dims0_[0] - 1;
                         i < dims0_[0] + _b.extent()[0] - 1; ++i)
                    {
                        _F.get_real_local(i - dims0_[0] + 1, j - dims0_[1] + 1,
                            k - dims0_[2] + 1) +=
                            fft_backward_
                                .output()[i + j * padded_dims_next_pow_2_[0] +
                                          k * padded_dims_next_pow_2_[0] *
                                              padded_dims_next_pow_2_[1]];
                    }
                }
            }
        }
        else
        {
            for (int j = dims0_[1] - 1; j < dims0_[1] + _b.extent()[1] - 1; ++j)
            {
                for (int i = dims0_[0] - 1; i < dims0_[0] + _b.extent()[0] - 1;
                     ++i)
                {
                    _F.get_real_local(i - dims0_[0] + 1, j - dims0_[1] + 1) +=
                        fft_backward_
                            .output()[i + j * padded_dims_next_pow_2_[0]];
                }
            }
        }
    }

    //void simd_prod_complex_add(const complex_vector_t& a, const complex_vector_t& b, complex_vector_t& res);

    //complex_vector_t& dft_r2c(std::vector<float_type>& _vec);

    //void fft_backward_field_clean();

  public:
    int fft_count_ = 0;
    int number_fwrd_executed = 0;  //register the number of forward procedure applied, if no forward applied, then don't need to apply backward one as well.

  private:
    dims_t padded_dims_;
    dims_t padded_dims_next_pow_2_;
    dims_t dims0_;
    dims_t dims1_;

    dfft_r2c fft_forward0_;
    dfft_r2c fft_forward1_;
    dfft_c2r fft_backward_;

    unsigned int     padded_size_;
    complex_vector_t tmp_prod;
};

} // namespace fft
} // namespace iblgf
#endif
