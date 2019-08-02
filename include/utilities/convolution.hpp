#ifndef INCLUDED_CONVOLUTION_IBLGF_HPP
#define INCLUDED_CONVOLUTION_IBLGF_HPP

#include <complex>
#include <cassert>
#include <cstring>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <global.hpp>
//#include <boost/align/aligned_allocator_adaptor.hpp>

#include <lgf/lgf.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>

#include <utilities/misc_math_functions.hpp>
#include <xsimd/xsimd.hpp>
#include <xsimd/stl/algorithms.hpp>

namespace fft
{

using namespace domain;

//TODO: Base these things also all on fields to exploit base/extent and
//      stride

class dfft_r2c
{
    //const int nthreads = 1;

public:
    using float_type=double;
    using complex_vector_t = std::vector<std::complex<float_type>,
          xsimd::aligned_allocator<std::complex<float_type>, 32>>;

    using real_vector_t = std::vector<float_type,
          xsimd::aligned_allocator<float_type, 32>>;

    //using complex_vector_t = std::vector<std::complex<float_type>,
    //      boost::alignment::aligned_allocator_adaptor<
    //          std::allocator<std::complex<float_type>>,32>> ;

    //using real_vector_t = std::vector<float_type,
    //      boost::alignment::aligned_allocator_adaptor<
    //            std::allocator<float_type>,32>>;

    using dims_t = types::vector_type<int,3>;


public: //Ctors:

    dfft_r2c(const dfft_r2c& other)              = default;
    dfft_r2c(dfft_r2c&& other)                   = default;
    dfft_r2c& operator=(const dfft_r2c& other) & = default;
    dfft_r2c& operator=(dfft_r2c&& other)      & = default;
    ~dfft_r2c() { fftw_destroy_plan(plan); }

    dfft_r2c( dims_t _dims_padded, dims_t _dims_non_zero )
    :dims_input_(_dims_padded),
     input_  (_dims_padded[2]*_dims_padded[1]*_dims_padded[0],0.0),
     output_1(_dims_padded[2]*_dims_padded[1]*((_dims_padded[0]/2)+1)),
     output_2(_dims_padded[2]*_dims_padded[1]*((_dims_padded[0]/2)+1)),
     output_ (_dims_padded[2]*_dims_padded[1]*((_dims_padded[0]/2)+1))
    {

        plan = (fftw_plan_dft_r2c_3d(_dims_padded[2], _dims_padded[1], _dims_padded[0],
                 &input_[0], reinterpret_cast<fftw_complex*>(&output_[0]),
                 FFTW_PATIENT ));

        r2c_1d_plans.resize(_dims_non_zero[0]);

        int dim_half = (_dims_padded[2]/2)+1;

        for (int i_plan = 0; i_plan<_dims_non_zero[0]; ++i_plan)
        {
            r2c_1d_plans[i_plan] =
                fftw_plan_many_dft_r2c(1, &_dims_padded[2], _dims_non_zero[1],
                            &input_[i_plan*_dims_padded[2]*_dims_padded[1] ], NULL,
                            1, _dims_padded[2],
                            reinterpret_cast<fftw_complex*>
                                (&output_1[i_plan*dim_half*_dims_padded[1]]), NULL,
                                 1, dim_half, FFTW_PATIENT
                            );

        }

        c2c_1d_plans_dir_2.resize(_dims_non_zero[0]);
        for (int i_plan = 0; i_plan<_dims_non_zero[0]; ++i_plan)
        {
            c2c_1d_plans_dir_2[i_plan] =
                fftw_plan_many_dft(1, &_dims_padded[1], dim_half,
                                reinterpret_cast<fftw_complex*>
                                    (&output_1[i_plan*dim_half*_dims_padded[1] ]), NULL,
                                dim_half, 1,
                                reinterpret_cast<fftw_complex*>
                                    (&output_2[i_plan*dim_half*_dims_padded[1] ]), NULL,
                                dim_half, 1,
                                FFTW_FORWARD, FFTW_PATIENT
                        );
        }

        c2c_1d_plans_dir_3 = fftw_plan_many_dft(1, &_dims_padded[0], dim_half * _dims_padded[1],
                reinterpret_cast<fftw_complex*>
                    (&output_2[0]), NULL,
                dim_half*_dims_padded[1], 1,
                reinterpret_cast<fftw_complex*>
                    (&output_[0]), NULL,
                dim_half*_dims_padded[1], 1,
                FFTW_FORWARD, FFTW_PATIENT
                );


    }

public: //Interface

    void execute_whole()
    {
        fftw_execute(plan);
    }

    void execute()
    {
        //Fisrt direction
        for (std::size_t i =0; i<r2c_1d_plans.size(); ++i)
            fftw_execute(r2c_1d_plans[i]);

        ////Second direction
        for (std::size_t i =0; i<c2c_1d_plans_dir_2.size(); ++i)
            fftw_execute(c2c_1d_plans_dir_2[i]);

        fftw_execute(c2c_1d_plans_dir_3);
    }


    auto& input(){return input_;}
    auto& output(){return output_;}
    auto output_copy(){return output_;}


    template<class Vector>
    void copy_input(const Vector& _v, dims_t _dims_v)
    {
        if(_v.size()==input_.size())
        {
            std::copy(_v.begin(),_v.end(),input_.begin() );
        }
        else
        {
            throw std::runtime_error("ERROR! LGF SIZE NOT MATCHING");
            ////Naive impl:
            //std::fill(input_.begin(), input_.end(),0);
            //for(int k=0;k<_dims_v[2];++k)
            //{
            //    for(int j=0;j<_dims_v[1];++j)
            //    {
            //        for(int i=0;i<_dims_v[0];++i)
            //        {
            //            input_[ i+dims_input_[0]*j+ dims_input_[0]*dims_input_[1]*k ]=
            //            _v[i+_dims_v[0]*j+_dims_v[0]*_dims_v[1]*k];
            //        }
            //    }
            //}
        }
    }

    //FIXME: Starting from zero is not general
    template<class Vector>
    void copy_field(const Vector& _v, dims_t _dims_v) noexcept
    {
        //Naive impl:
        //std::fill(input_.begin(), input_.end(),0);

        for(int k=0;k<_dims_v[2];++k)
        {
            for(int j=0;j<_dims_v[1];++j)
            {
                //std::copy(&_v.get_real_local(0,j,k),
                //            &_v.get_real_local(0,j,k) + _dims_v[0],
                //            &input_[dims_input_[0]*j+ dims_input_[0]*dims_input_[1]*k] );

                for(int i=0;i<_dims_v[0];++i)
                {
                    input_[ i+dims_input_[0]*j+ dims_input_[0]*dims_input_[1]*k ]=
                     _v.get_real_local(i,j,k);
                }
            }
        }
    }


private:

    dims_t dims_input_;
    real_vector_t input_;
    complex_vector_t output_1, output_2, output_;

    fftw_plan plan;

    std::vector<fftw_plan> r2c_1d_plans;
    std::vector<fftw_plan> c2c_1d_plans_dir_2;
    fftw_plan c2c_1d_plans_dir_3;

    fftw_plan r2c_plan_1d;

};

class dfft_c2r
{
public:
    //const int nthreads = 1;
    using complex_vector_t = std::vector<std::complex<float_type>,
          xsimd::aligned_allocator<std::complex<float_type>, 32>>;

    using real_vector_t = std::vector<float_type,
          xsimd::aligned_allocator<float_type, 32>>;

    //using complex_vector_t = std::vector<std::complex<float_type>,
    //      boost::alignment::aligned_allocator_adaptor<
    //          std::allocator<std::complex<float_type>>,32>> ;

    //using real_vector_t = std::vector<float_type,
    //      boost::alignment::aligned_allocator_adaptor<
    //          std::allocator<float_type>,32>>;

    using dims_t = types::vector_type<int,3>;


public: //Ctors:

    dfft_c2r(const dfft_c2r& other)              = default;
    dfft_c2r(dfft_c2r&& other)                   = default;
    dfft_c2r& operator=(const dfft_c2r& other) & = default;
    dfft_c2r& operator=(dfft_c2r&& other)      & = default;
    ~dfft_c2r() { fftw_destroy_plan(plan); }

    dfft_c2r( dims_t _dims, dims_t _dims_small )
    :input_(_dims[2]*_dims[1]*((_dims[0]/2)+1),std::complex<float_type>(0.0)),
    tmp_1_ (_dims[2]*_dims[1]*((_dims[0]/2)+1),std::complex<float_type>(0.0)),
    tmp_2_ (_dims[2]*_dims[1]*((_dims[0]/2)+1),std::complex<float_type>(0.0)),
    output_(_dims[2]*_dims[1]*_dims[0],0.0)
    {
        //int status = fftw_init_threads();
        //fftw_plan_with_nthreads(nthreads);
        plan = fftw_plan_dft_c2r_3d(_dims[2], _dims[1], _dims[0],
                 reinterpret_cast<fftw_complex*>(&input_[0]), &output_[0],
                 FFTW_PATIENT);

        int dim_half = (_dims[2]/2)+1;
        c2c_dir_1 = fftw_plan_many_dft(1, &_dims[0], _dims[1] * dim_half,
                        reinterpret_cast<fftw_complex*>
                            (&input_[0]), NULL,
                             _dims[1] * dim_half, 1,
                        reinterpret_cast<fftw_complex*>
                            (&tmp_1_[0]), NULL,
                             _dims[1] * dim_half, 1,
                        FFTW_BACKWARD, FFTW_PATIENT
                );

        ////Dir 1
        c2c_dir_2.resize(_dims_small[0]);
        for (int i_plan = 0; i_plan<_dims_small[0]; ++i_plan)
        {
            c2c_dir_2[i_plan] =
                fftw_plan_many_dft(1, &_dims[1], dim_half,
                                reinterpret_cast<fftw_complex*>
                                    (&tmp_1_[ (i_plan + _dims_small[0]-1)*dim_half*_dims[1] ]), NULL,
                                dim_half, 1,
                                reinterpret_cast<fftw_complex*>
                                    (&tmp_2_[ (i_plan + _dims_small[0]-1)*dim_half*_dims[1] ]), NULL,
                                dim_half, 1,
                        FFTW_BACKWARD, FFTW_PATIENT
                );

        }

        //// Dir 2
        c2r_dir_3.resize(_dims_small[0]);
        for (int i_plan = 0; i_plan<_dims_small[0]; ++i_plan)
        {
            c2r_dir_3[i_plan] =
                fftw_plan_many_dft_c2r(1, &_dims[2], _dims_small[1],
                                reinterpret_cast<fftw_complex*>
                                    (&tmp_2_[ (i_plan + _dims_small[0]-1)*dim_half*_dims[1] + dim_half*(_dims_small[1]-1) ]), NULL,
                                1, dim_half,
                                &output_[ (i_plan + _dims_small[0]-1)*_dims[2]*_dims[1] + _dims[2]*(_dims_small[1]-1)  ] , NULL,
                                1, _dims[2],
                                FFTW_PATIENT
                                );
        }

    }

public: //Interface

    void execute()
    {
        fftw_execute(c2c_dir_1);

        for (std::size_t i=0; i<c2c_dir_2.size(); ++i)
            fftw_execute(c2c_dir_2[i]);

        for (std::size_t i=0; i<c2r_dir_3.size(); ++i)
            fftw_execute(c2r_dir_3[i]);
    }

    auto& input(){return input_;}
    auto& output(){return output_;}

private:

    dims_t dims_next_pow_2_;
    complex_vector_t input_, tmp_1_, tmp_2_;
    real_vector_t output_;

    fftw_plan plan, c2c_dir_1;
    std::vector<fftw_plan> c2c_dir_2, c2r_dir_3;

    int dim_half;

};


///template<class Kernel>
class Convolution
{

public:

    using dims_t = types::vector_type<int,3>;

    using complex_vector_t = std::vector<std::complex<float_type>,
          xsimd::aligned_allocator<std::complex<float_type>, 32>>;

    using real_vector_t = std::vector<float_type,
          xsimd::aligned_allocator<float_type, 32>>;

    //using complex_vector_t = std::vector<std::complex<float_type>,
    //      boost::alignment::aligned_allocator_adaptor<
    //          std::allocator<std::complex<float_type>>,32>> ;

    //using real_vector_t = std::vector<float_type,
    //      boost::alignment::aligned_allocator_adaptor<
    //          std::allocator<float_type>,32>>;

    using lgf_key_t = std::tuple<int, int, int>;
    using lgf_matrix_ptr_map_type = std::map<lgf_key_t, std::unique_ptr<complex_vector_t> >;

public: //Ctors

    Convolution(const Convolution& other)              = default;
    Convolution(Convolution&& other)                   = default;
    Convolution& operator=(const Convolution& other) & = default;
    Convolution& operator=(Convolution&& other) &      = default;
    ~Convolution()                                     = default;


    Convolution(dims_t _dims0, dims_t _dims1)
    :padded_dims_(_dims0 + _dims1 - 1),
     padded_dims_next_pow_2_({math::next_pow_2(padded_dims_[0]),
                              math::next_pow_2(padded_dims_[1]),
                              math::next_pow_2(padded_dims_[2])}),
     dims0_(_dims0),
     dims1_(_dims1),
     fft_forward0_(padded_dims_next_pow_2_, _dims0),
     fft_forward1_(padded_dims_next_pow_2_, _dims1),
     fft_backward_(padded_dims_next_pow_2_, _dims1),
     padded_size_(padded_dims_next_pow_2_[0]*padded_dims_next_pow_2_[1]*padded_dims_next_pow_2_[2]),
     tmp_prod(padded_size_,std::complex<float_type>(0.0))
    {
    }

    void fft_backward_field_clean()
    {
        std::fill (fft_backward_.input().begin(),fft_backward_.input().end(),0);
    }

    template<
        typename Source,
        typename BlockType, class Kernel>
    void apply_forward_add( const BlockType& _lgf_block,
                    Kernel* _kernel,
                    int _level_diff,
                    const Source& _source)
    {
            execute_fwrd_field(_lgf_block, _kernel, _level_diff, _source);
    }

    template<
        typename Target,
        typename BlockType>
    void apply_backward(const BlockType& _extractor,
                    Target& _target,
                    float_type _extra_scale)
    {
        const float_type scale = 1.0 / (padded_dims_next_pow_2_[0] *
                                        padded_dims_next_pow_2_[1] *
                                        padded_dims_next_pow_2_[2]) *
                                        _extra_scale;
        for(std::size_t i = 0; i < fft_backward_.input().size(); ++i)
        {
            fft_backward_.input()[i] *= scale;
        }

        fft_backward_.execute();
        add_solution(_extractor, _target);
    }

    //template<
    //    typename Source, typename Target,
    //    typename BlockType, class Kernel>
    //void apply_lgf( const BlockType& _lgf_block,
    //                Kernel* _kernel,
    //                int _level_diff,
    //                const Source& _source,
    //                const BlockType& _extractor,
    //                Target& _target,
    //                float_type _scale )
    //{
    //        execute_field(_lgf_block, _kernel, _level_diff, _source, _scale);
    //        add_solution(_extractor, _target);
    //        fft_count_ ++;
    //}

    template<class Field, class BlockType, class Kernel>
    void execute_fwrd_field(const BlockType& _lgf_block,
                       Kernel* _kernel,
                       int _level_diff, const Field& _b)
    {
        auto& f0 = _kernel->dft(_lgf_block, padded_dims_next_pow_2_, this, _level_diff);

        fft_forward1_.copy_field(_b, dims1_);
        fft_forward1_.execute();
        auto& f1 = fft_forward1_.output();

        complex_vector_t prod(f0.size());

        //xsimd::transform(f0.begin(), f0.end(), f1.begin(), tmp_prod.begin(),
        //        [](const auto& x, const auto& y) {return x*y; });

        //xsimd::transform(tmp_prod.begin(), tmp_prod.end(),
        //            fft_backward_.input().begin(), fft_backward_.input().begin(),
        //        [](const auto& x, const auto& y) {return x+y; });

        simd_prod_complex_add(f0, f1,fft_backward_.input());
    }

    void simd_prod_complex_add(const complex_vector_t& a,
            const complex_vector_t& b,
            complex_vector_t& res)
    {
        std::size_t size = a.size();
        constexpr std::size_t simd_size = xsimd::simd_type<std::complex<float_type>>::size;
        std::size_t vec_size = size - size % simd_size;

        for(std::size_t i = 0; i < vec_size; i += simd_size)
        {
            auto ba = xsimd::load_aligned(&a[i]);
            auto bb = xsimd::load_aligned(&b[i]);
            auto res_old = xsimd::load_aligned(&res[i]);
            auto bres = ba*bb+res_old;
            bres.store_aligned(&res[i]);
        }
        for(std::size_t i = vec_size; i < size; ++i)
        {
            res[i] += a[i]*b[i];
        }
    }

    //template<class Field, class BlockType, class Kernel>
    //void execute_field(const BlockType& _lgf_block,
    //                   Kernel* _kernel,
    //                   int _level_diff, const Field& _b,
    //                   const float_type _extra_scale)
    //{
    //    auto& f0 = _kernel->dft(_lgf_block, padded_dims_next_pow_2_, this, _level_diff);

    //    fft_forward1_.copy_field(_b, dims1_);
    //    fft_forward1_.execute();
    //    auto& f1 = fft_forward1_.output();

    //    complex_vector_t prod(f0.size());
    //    const float_type scale = 1.0 / (padded_dims_next_pow_2_[0] *
    //                                    padded_dims_next_pow_2_[1] *
    //                                    padded_dims_next_pow_2_[2]) * _extra_scale;
    //    for(std::size_t i = 0; i < prod.size(); ++i)
    //    {
    //        fft_backward_.input()[i] = f0[i]*f1[i]*scale;
    //    }

    //    fft_backward_.execute();
    //}

    auto& dft_r2c(std::vector<float_type>& _vec )
    {
        fft_forward0_.copy_input(_vec, dims0_);
        fft_forward0_.execute_whole();
        return fft_forward0_.output();
    }


    auto& output()
    {
        return fft_backward_.output();
    }

    template<class Block,class Field>
    void add_solution(const Block& _b, Field& _F)
    {
        for (int k = dims0_[2]-1; k < dims0_[2]+_b.extent()[2]-1; ++k)
        {
            for (int j = dims0_[1]-1; j < dims0_[1]+_b.extent()[1]-1; ++j)
            {
                for (int i = dims0_[0]-1; i < dims0_[0]+_b.extent()[0]-1; ++i)
                {
                    _F.get_real_local(i-dims0_[0]+1,j-dims0_[1]+1,k-dims0_[2]+1 ) +=
                    fft_backward_.output() [
                        i+j*padded_dims_next_pow_2_[0]+k*padded_dims_next_pow_2_[0]*padded_dims_next_pow_2_[1]
                    ];
                }
            }
        }
    }

public:
    int fft_count_=0;

private:
    dims_t padded_dims_;
    dims_t padded_dims_next_pow_2_;
    dims_t dims0_;
    dims_t dims1_;


    dfft_r2c fft_forward0_;
    dfft_r2c fft_forward1_;
    dfft_c2r fft_backward_;

    unsigned int padded_size_;
    complex_vector_t tmp_prod;

};


} //namespace
#endif

