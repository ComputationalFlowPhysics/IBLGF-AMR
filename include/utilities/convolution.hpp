#ifndef INCLUDED_CONVOLUTION_IBLGF_HPP
#define INCLUDED_CONVOLUTION_IBLGF_HPP


#include <cassert>
#include <cstring>
#include <vector>
#include <complex.h>
#include <fftw3.h>
#include <global.hpp>
#include <boost/align/aligned_allocator_adaptor.hpp>

#include <lgf/lgf.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
namespace fft
{

using namespace domain;

//TODO: Base these things also all on fields to exploit base/extent and
//      stride

class dfft_r2c
{
    const int nthreads = 8;

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
     output_(_dims[2]*_dims[1]*((_dims[0]/2)+1))
    {
        //int status = fftw_init_threads();
        fftw_plan_with_nthreads(nthreads);
        plan = (fftw_plan_dft_r2c_3d(_dims[2], _dims[1], _dims[0],
                 &input_[0], reinterpret_cast<fftw_complex*>(&output_[0]),
                 FFTW_MEASURE ));
    }

public: //Interface

    void execute()
    {
        fftw_execute(plan);
    }

    auto& input(){return input_;}
    auto& output(){return output_;}
    auto output_copy(){return output_;}


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
                     _v.get_real_local(i,j,k);
                        //_v[i+_dims_v[0]*j+_dims_v[0]*_dims_v[1]*k];
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
    const int nthreads = 8;
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
     output_(_dims[2]*_dims[1]*_dims[0],0.0)
    {
        //int status = fftw_init_threads();
        fftw_plan_with_nthreads(nthreads);
        plan = fftw_plan_dft_c2r_3d(_dims[2], _dims[1], _dims[0],
                 reinterpret_cast<fftw_complex*>(&input_[0]), &output_[0],
                 FFTW_MEASURE );
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

    using dims_t = types::vector_type<int,3>;

    using datablock_t = DataBlock<3, node>;
    using block_descriptor_t = typename datablock_t::block_descriptor_type;
    using coordinate_t = typename block_descriptor_t::base_t;

    using complex_vector_t = std::vector<std::complex<float_type>,
          boost::alignment::aligned_allocator_adaptor<
              std::allocator<std::complex<float_type>>,32>> ;

    using real_vector_t = std::vector<float_type,
          boost::alignment::aligned_allocator_adaptor<
              std::allocator<float_type>,32>>;

    using lgf_key_t = std::tuple<int, int, int>;
    using lgf_matrix_ptr_map_type = std::map<lgf_key_t, std::unique_ptr<complex_vector_t> >;

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
    construct_lgf_matrix_level_maps();
    }

    void construct_lgf_matrix_level_maps()
    {
        int max_lgf_map_level = 20;
        lgf_level_maps_.clear();
        lgf_level_maps_.resize(max_lgf_map_level);
    }

    template<
        typename lgf_block_t,
        typename source_t,
        typename target_t,
        typename extractor_t>
    void apply_lgf( lgf_block_t lgf_block,
                    int level_diff,
                    source_t& source,
                    extractor_t extractor,
                    target_t& target,
                    float_t scale )
    {
            execute_field(lgf_block, level_diff, source);
            add_solution(extractor, target, scale);
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

    template<class Field,
            typename block_dsrp_t>
    void execute_field(block_dsrp_t lgf_block_dsrp, int level_diff, Field& _b)
    {
        // use lgf_block.shift and level_diff to check if it has been saved or
        // not

        const auto base = lgf_block_dsrp.base();

        complex_vector_t* f_ptr;

        lgf_key_t k_(base[0],base[1],base[2]);
        auto it = lgf_level_maps_[level_diff].find( k_ );

        if ( it == lgf_level_maps_[level_diff].end() )
        {
            lgf_.get_subblock( lgf_block_dsrp, lgf, level_diff);
            fft_forward0.copy_input(lgf, dims0_);
            fft_forward0.execute();

            f_ptr = &fft_forward0.output();
            lgf_level_maps_[level_diff].emplace(k_,
                    std::unique_ptr<complex_vector_t> ( new complex_vector_t(*f_ptr) ) );
        } else
        {

            f_ptr = (it->second).get();
        }

        auto& f0 = *f_ptr;

        fft_forward1.copy_field(_b, dims1_);
        fft_forward1.execute();
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
                    F.get_real_local(i-dims0_[0]+1,j-dims0_[1]+1,k-dims0_[2]+1 ) +=
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


    std::vector<float_type> lgf;
    lgf::LGF<lgf::Lookup>   lgf_;       ///< Lookup for the LGFs

    std::vector<lgf_matrix_ptr_map_type> lgf_level_maps_;   ///< Octants per level
};


} //namespace
#endif

