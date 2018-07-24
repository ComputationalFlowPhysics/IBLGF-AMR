#ifndef INCLUDED_CONVOLUTION_HPP
#define INCLUDED_CONVOLUTION_HPP


#include <cassert>
#include <cstring>
#include <vector>
#include <complex.h>
#include <fftw3.h>
#include <global.hpp>

namespace fft
{

class dfft_r2c
{
public:
    using complex_vector_t = std::vector<std::complex<float_type>,
          boost::alignment::aligned_allocator_adaptor<
              std::allocator<std::complex<float_type>>,32>> ;

    using real_vector_t = std::vector<float_type,
          boost::alignment::aligned_allocator_adaptor<
              std::allocator<float_type>,32>>;

    using dims_t = types::vector_type<int,3>;


public: //Ctors:

    dfft_r2c(const dfft_r2c& other)              = delete;
    dfft_r2c(dfft_r2c&& other)                   = default;
    dfft_r2c& operator=(const dfft_r2c& other) & = delete;
    dfft_r2c& operator=(dfft_r2c&& other)      & = default;
    ~dfft_r2c() { fftwf_destroy_plan(plan); }

    dfft_r2c( dims_t _dims )
    :input_(_dims[2]*_dims[1]*_dims[0],0.0),
     output_(_dims[2]*_dims[1]*((_dims[0]/2)+1))

    {
        plan=fftwf_plan_dft_r2c_3d(_dims[0], _dims[1], _dims[2],
                &input_[0], reinterpret_cast<fftwf_complex*>(&output_[0]),
                FFTW_ESTIMATE| FFTW_PRESERVE_INPUT );
    }

public: //Interface
    
    void execute()
    {
        fftwf_execute(plan);
    }    

    auto& input(){return input_;}
    auto& output(){return output_;}

    void copy_input(const real_vector_t& _v) noexcept
    {
        std::copy(_v.begin(),_v.end(),input_.begin() );
        std::fill_n(input_.begin()+ _v.size(), input_.size() - _v.size(),0);
    }

private:

    complex_vector_t output_;
    real_vector_t input_;
    fftwf_plan plan;
}; 

class dfft_c2r
{
public:
    using complex_vector_t = std::vector<std::complex<float_type>,
          boost::alignment::aligned_allocator_adaptor<
              std::allocator<std::complex<float_type>>,32>> ;

    using real_vector_t = std::vector<float_type,
          boost::alignment::aligned_allocator_adaptor<
              std::allocator<float_type>,32>>;

    using dims_t = types::vector_type<int,3>;


public: //Ctors:

    dfft_c2r(const dfft_c2r& other)              = delete;
    dfft_c2r(dfft_c2r&& other)                   = default;
    dfft_c2r& operator=(const dfft_c2r& other) & = delete;
    dfft_c2r& operator=(dfft_c2r&& other)      & = default;
    ~dfft_c2r() { fftwf_destroy_plan(plan); }

    dfft_c2r( dims_t _dims )
    :input_(_dims[2]*_dims[1]*((_dims[0]/2)+1)),
     output_(_dims[2]*_dims[1]*_dims[0],0.0)

    {
        plan=fftwf_plan_dft_c2r_3d(_dims[0], _dims[1], _dims[2],
                reinterpret_cast<fftwf_complex*>(&input_[0]), &output_[0],
                FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

    }

public: //Interface
    
    void execute()
    {
        fftwf_execute(plan);
    }    

    auto& input(){return input_;}
    auto& output(){return output_;}

    void copy_input(const complex_vector_t& _v) noexcept
    {
        std::copy(_v.begin(),_v.end(),input_.begin() );
        std::fill_n(input_.begin()+ _v.size(), input_.size() - _v.size(),0);
    }

private:

    complex_vector_t input_;
    real_vector_t output_;
    fftwf_plan plan;
}; 


class Convolution
{

public:

    using complex_vector_t = std::vector<std::complex<float_type>,
          boost::alignment::aligned_allocator_adaptor<
              std::allocator<std::complex<float_type>>,32>> ;

    using real_vector_t = std::vector<float_type,
          boost::alignment::aligned_allocator_adaptor<
              std::allocator<float_type>,32>>;

    using dims_t = types::vector_type<int,3>;

    friend dfft_c2r;
    friend dfft_r2c;


public: //Ctors

    Convolution(const Convolution& other)              = delete;
    Convolution(Convolution&& other)                   = default;
    Convolution& operator=(const Convolution& other) & = delete;
    Convolution& operator=(Convolution&& other) &      = default;
    ~Convolution()                                     = default;


    Convolution(dims_t _dims0, dims_t _dims1)
    :padded_dims(_dims0 + _dims1 - 1),
     fft_forward0(padded_dims),
     fft_forward1(padded_dims),
     fft_backward(padded_dims)
    {
        padded_dims = _dims0 + _dims1 - 1;
        fftw_forward(padded_dims);
    }

    void execute(real_vector_t& _a, real_vector_t& _b)
    {
        fft_forward0.copy_input(_a);
        fft_forward0.execute();

        fft_forward1.copy_input(_b);
        fft_forward1.execute();
        
        auto& f0=fft_forward0.output();
        auto& f1=fft_forward1.output();
        complex_vector_t prod(f0.size());
        for(std::size_t i=0; i< prod.size();++i)
        {
            prod[i] = f0[i]*f1[i];
        }

        fft_backward.copy_input(prod);
        fft_backward.execute();
    }

    auto& output()
    {
        fft_backward.output();
    }

private:
    padded_dims;
    dfft_r2c fft_forward0;
    dfft_r2c fft_forward1;
    dfft_c2r fft_backward;

};


} //namespace
#endif

