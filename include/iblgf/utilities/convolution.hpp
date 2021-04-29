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
#include <iblgf/domain/octree/key.hpp>
#include <unordered_map>

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
    using dims_t = types::vector_type<int, 3>;

  public: //Ctors:
    dfft_r2c(const dfft_r2c& other) = default;
    dfft_r2c(dfft_r2c&& other) = default;
    dfft_r2c& operator=(const dfft_r2c& other) & = default;
    dfft_r2c& operator=(dfft_r2c&& other) & = default;
    ~dfft_r2c() { fftw_destroy_plan(plan); }

    dfft_r2c(dims_t _dims_padded, dims_t _dims_non_zero);

  public: //Interface
    void         execute_whole();
    void         execute();
    inline auto& input() { return input_; }
    inline auto& output() { return output_; }
    inline auto  output_copy() { return output_; }

    template<class Vector>
    void copy_input(const Vector& _v, dims_t _dims_v);

    template<class Vector>
    void copy_field(const Vector& _v, dims_t _dims_v) noexcept
    {
        for (int k = 0; k < _dims_v[2]; ++k)
        {
            for (int j = 0; j < _dims_v[1]; ++j)
            {
                for (int i = 0; i < _dims_v[0]; ++i)
                {
                    input_[i + dims_input_[0] * j +
                           dims_input_[0] * dims_input_[1] * k] =
                        _v.get_real_local(i, j, k);
                }
            }
        }
    }

  private:
    dims_t           dims_input_;
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
    using dims_t = types::vector_type<int, 3>;

  public: //Ctors:
    dfft_c2r(const dfft_c2r& other) = default;
    dfft_c2r(dfft_c2r&& other) = default;
    dfft_c2r& operator=(const dfft_c2r& other) & = default;
    dfft_c2r& operator=(dfft_c2r&& other) & = default;
    ~dfft_c2r() { fftw_destroy_plan(plan); }

    dfft_c2r(dims_t _dims, dims_t _dims_small);

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
class Convolution
{
  public:
    using dims_t = types::vector_type<int, 3>;
    using complex_vector_t = std::vector<std::complex<float_type>,
        xsimd::aligned_allocator<std::complex<float_type>, 32>>;
    using real_vector_t =
        std::vector<float_type, xsimd::aligned_allocator<float_type, 32>>;
    using lgf_key_t = std::tuple<int, int, int>;
    using lgf_matrix_ptr_map_type =
        std::map<lgf_key_t, std::unique_ptr<complex_vector_t>>;
    using sr_fft_map_type =
        std::map<unsigned long long int, complex_vector_t>;
    using b_type = xsimd::simd_type<std::complex<float_type>>;

  public: //Ctors
    Convolution(const Convolution& other) = default;
    Convolution(Convolution&& other) = default;
    Convolution& operator=(const Convolution& other) & = default;
    Convolution& operator=(Convolution&& other) & = default;
    ~Convolution() = default;

    Convolution(dims_t _dims0, dims_t _dims1);

  public: //Members:
    complex_vector_t& dft_r2c(std::vector<float_type>& _vec);
    void              fft_backward_field_clean();
    void              simd_copy(const complex_vector_t& a, complex_vector_t& res);
    void              simd_prod_complex_add(const complex_vector_t& a,
                     const complex_vector_t& b, complex_vector_t& res);
    auto&             output() { return fft_backward_.output(); }

    template<typename Source, typename FMM_source_tmp, typename BlockType, class Kernel>
    void apply_forward_add(const BlockType& _lgf_block, Kernel* _kernel,
        int _level_diff, const Source& _source, FMM_source_tmp& _fmm_source_fft, bool& reuse_fft)
    {
        execute_fwrd_field(_lgf_block, _kernel, _level_diff, _source, _fmm_source_fft, reuse_fft);
    }

    template<typename Target, typename BlockType>
    void apply_backward(
        const BlockType& _extractor, Target& _target, float_type _extra_scale)
    {
        const float_type scale =
            1.0 /
            (padded_dims_next_pow_2_[0] * padded_dims_next_pow_2_[1] *
                padded_dims_next_pow_2_[2]) *
            _extra_scale;
        for (std::size_t i = 0; i < fft_backward_.input().size(); ++i)
        { fft_backward_.input()[i] *= scale; }

        fft_backward_.execute();
        add_solution(_extractor, _target);
    }

    template<class Field, class BlockType, typename FMM_source_tmp , class Kernel>
    void execute_fwrd_field(const BlockType& _lgf_block, Kernel* _kernel,
        int _level_diff, const Field& _b, FMM_source_tmp& _fmm_source_fft, bool& reuse_fft)
    {
        auto& f0 = _kernel->dft(
            _lgf_block, padded_dims_next_pow_2_, this, _level_diff);

        if (reuse_fft)
        {
            simd_prod_complex_add(f0, _fmm_source_fft, fft_backward_.input());
        }
        else
        {
            fft_forward1_.copy_field(_b, dims1_);
            fft_forward1_.execute();
            auto& f1 = fft_forward1_.output();

            simd_prod_complex_add(f0, f1, fft_backward_.input());
            simd_copy(f1, _fmm_source_fft);
            reuse_fft=true;
        }
    }

    template<class Block, class Field>
    void add_solution(const Block& _b, Field& _F)
    {
        for (int k = dims0_[2] - 1; k < dims0_[2] + _b.extent()[2] - 1; ++k)
        {
            for (int j = dims0_[1] - 1; j < dims0_[1] + _b.extent()[1] - 1; ++j)
            {
                for (int i = dims0_[0] - 1; i < dims0_[0] + _b.extent()[0] - 1;
                     ++i)
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


    void sr_fft_map_clear()
    {
        sr_fft_map_.clear();
    }

  public:
    int fft_count_ = 0;

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
    sr_fft_map_type sr_fft_map_;

    b_type bres1_, bres2_;
};

} // namespace fft
} // namespace iblgf
#endif

