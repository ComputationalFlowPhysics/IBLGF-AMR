#ifndef INCLUDED_CONVOLUTION_HPP
#define INCLUDED_CONVOLUTION_HPP


#include <cassert>
#include <cstring>
#include <vector>
#include <complex.h>
#include <fftw3.h>
#include <global.hpp>
#include <boost/align/aligned_allocator_adaptor.hpp>

namespace fft
{

class dfft_r2c
{
public:
    using float_type=double;
    using complex_vector_t = std::vector<std::complex<float_type>, 
          boost::alignment::aligned_allocator_adaptor< 
              std::allocator<std::complex<float_type>>,32>> ;

    using real_vector_t = std::vector<float_type, 
          boost::alignment::aligned_allocator_adaptor< 
              std::allocator<float_type>,32>>;

    using dims_t = types::vector_type<int,3>;


public: //Ctors:

    dfft_r2c(const dfft_r2c& other)              = default;
    dfft_r2c(dfft_r2c&& other)                   = default;
    dfft_r2c& operator=(const dfft_r2c& other) & = default;
    dfft_r2c& operator=(dfft_r2c&& other)      & = default;
    ~dfft_r2c() { fftw_destroy_plan(plan); }

    dfft_r2c( dims_t _dims )
    :dims_input_(_dims),
     input_(_dims[2]*_dims[1]*_dims[0],0.0),
     output_(_dims[2]*_dims[1]*((_dims[0]/2)+1)),
     plan(fftw_plan_dft_r2c_3d(_dims[2], _dims[1], _dims[0],
                 &input_[0], reinterpret_cast<fftw_complex*>(&output_[0]),
                 FFTW_ESTIMATE ))
    {
    }

public: //Interface
    
    void execute()
    {
        fftw_execute(plan);
    }    

    auto& input(){return input_;}
    auto& output(){return output_;}


    template<class Vector>
    void copy_input(const Vector& _v, dims_t _dims_v) noexcept
    {
        if(_v.size()==input_.size())
        {
            std::copy(_v.begin(),_v.end(),input_.begin() );
        } 
        else
        {
            //Naive impl:
            std::fill(input_.begin(), input_.end(),0);
            for(int k=0;k<_dims_v[2];++k)
            {
                for(int j=0;j<_dims_v[1];++j)
                {
                    for(int i=0;i<_dims_v[0];++i)
                    {
                        input_[ i+dims_input_[0]*j+ dims_input_[0]*dims_input_[1]*k ]=
                        _v[i+_dims_v[0]*j+_dims_v[0]*_dims_v[1]*k];
                    }
                }
            }
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
                for(int i=0;i<_dims_v[0];++i)
                {
                    input_[ i+dims_input_[0]*j+ dims_input_[0]*dims_input_[1]*k ]=
                        //_v[i+_dims_v[0]*j+_dims_v[0]*_dims_v[1]*k];
                        _v.get_local(i,j,k);
                }
            }
        }
    }        
        

private:

    dims_t dims_input_;
    real_vector_t input_;
    complex_vector_t output_;
    fftw_plan plan;
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

    dfft_c2r(const dfft_c2r& other)              = default;
    dfft_c2r(dfft_c2r&& other)                   = default;
    dfft_c2r& operator=(const dfft_c2r& other) & = default;
    dfft_c2r& operator=(dfft_c2r&& other)      & = default;
    ~dfft_c2r() { fftw_destroy_plan(plan); }

    dfft_c2r( dims_t _dims )
    :input_(_dims[2]*_dims[1]*((_dims[0]/2)+1),std::complex<float_type>(0.0)),
     output_(_dims[2]*_dims[1]*_dims[0],0.0),
     plan(fftw_plan_dft_c2r_3d(_dims[2], _dims[1], _dims[0],
                 reinterpret_cast<fftw_complex*>(&input_[0]), &output_[0],
                 FFTW_ESTIMATE ))
    {

    }

public: //Interface
    
    void execute()
    {
        fftw_execute(plan);
    }    

    auto& input(){return input_;}
    auto& output(){return output_;}

private:
    complex_vector_t input_;
    real_vector_t output_;
    fftw_plan plan;
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


public: //Ctors

    Convolution(const Convolution& other)              = default;
    Convolution(Convolution&& other)                   = default;
    Convolution& operator=(const Convolution& other) & = default;
    Convolution& operator=(Convolution&& other) &      = default;
    ~Convolution()                                     = default;


    Convolution(dims_t _dims0, dims_t _dims1)
    :padded_dims(_dims0 + _dims1 - 1),
     dims0_(_dims0),
     dims1_(_dims1),
     fft_forward0(padded_dims),
     fft_forward1(padded_dims),
     fft_backward(padded_dims)
    {
    }

    template<class Vector>
    void execute(Vector& _a, Vector& _b)
    {
        fft_forward0.copy_input(_a, dims0_);
        fft_forward0.execute();

        fft_forward1.copy_input(_b,dims1_);
        fft_forward1.execute();

        auto& f0 = fft_forward0.output();
        auto& f1 = fft_forward1.output();
        complex_vector_t prod(f0.size());
        const float_type scale = 1.0 / (padded_dims[0] * 
                                        padded_dims[1] * 
                                        padded_dims[2]);
        for(std::size_t i = 0; i < prod.size(); ++i)
        {
            fft_backward.input()[i] = f0[i]*f1[i]*scale;
        }
        fft_backward.execute();
    }

    template<class Vector, class Field>
    void execute_field(Vector& _a, Field& _b)
    {
        fft_forward0.copy_input(_a, dims0_);
        fft_forward0.execute();

        fft_forward1.copy_field(_b,dims1_);
        fft_forward1.execute();

        auto& f0 = fft_forward0.output();
        auto& f1 = fft_forward1.output();
        complex_vector_t prod(f0.size());
        const float_type scale = 1.0 / (padded_dims[0] * 
                                        padded_dims[1] * 
                                        padded_dims[2]);
        for(std::size_t i = 0; i < prod.size(); ++i)
        {
            fft_backward.input()[i] = f0[i]*f1[i]*scale;
        }
        fft_backward.execute();
    }


    auto& output()
    {
        return fft_backward.output();
    }

    template<class Block,class Field>
    void add_solution(const Block& _b, Field& F, const float_type _scale)
    {
        for (int k = dims0_[2]-1; k < dims0_[2]+_b.extent()[2]-1; ++k)
        {
            for (int j = dims0_[1]-1; j < dims0_[1]+_b.extent()[1]-1; ++j)
            {
                for (int i = dims0_[0]-1; i < dims0_[0]+_b.extent()[0]-1; ++i)
                {
                    F.get_local(i-dims0_[0]+1,j-dims0_[1]+1,k-dims0_[2]+1 ) += 
                    _scale*fft_backward.output() [ 
                        i+j*padded_dims[0]+k*padded_dims[0]*padded_dims[1]
                    ];
                }
            }
        }
    }
    

private:
    dims_t padded_dims;
    dims_t dims0_;
    dims_t dims1_;
    dfft_r2c fft_forward0;
    dfft_r2c fft_forward1;
    dfft_c2r fft_backward;

};


} //namespace
#endif

