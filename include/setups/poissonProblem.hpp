#ifndef IBLGF_INCLUDED_POISSON_HPP
#define IBLGF_INCLUDED_POISSON_HPP

#include <iostream>
#include <vector>
#include <tuple>

#include <algorithm>
#include <list>
#include <vector>
#include <iostream>
#include <numeric>
#include <random>
#include <cmath>
#include <functional>

#include <fftw3.h>

// IBLGF-specific
#include <global.hpp>
#include <simulation.hpp>
#include <domain/domain.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <domain/octree/tree.hpp>
#include <post-processing/parallel_ostream.hpp>
#include <lgf/lgf.hpp>

#include<utilities/convolution.hpp>

const int Dim = 3;

using namespace domain;
using namespace octree;
using namespace types;
using namespace dictionary;
using namespace fft;

struct PoissonProblem
{
    using vel_type    = vector_type<float_type, Dim>;
    using size_v_type = vector_type<int       , Dim>;

    //              name                type
    make_field_type(phi_num         , float_type)
    make_field_type(source          , float_type)
    make_field_type(lgf_field_lookup, float_type)
    make_field_type(phi_exact       , float_type)
    make_field_type(lgf             , float_type)
    make_field_type(error           , float_type)
    make_field_type(error2          , float_type)
    make_field_type(lapace_field    , float_type)
    make_field_type(lapace_error    , float_type)
    make_field_type(dummy_field     , float_type)


    using datablock_t = DataBlock<
        Dim, node,
        phi_num,
        source,
        lgf_field_lookup,
        phi_exact,
        error,
        error2,
        lapace_field,
        lapace_error,
        dummy_field     
    >;

    using datablock_t_2 = DataBlock<Dim, node, lgf>;

    using block_descriptor_t = typename datablock_t::block_descriptor_type;
    using tree_t             = Tree<Dim,datablock_t>;
    using octant_t           = typename tree_t::octant_type;
    using coordinate_t       = typename datablock_t::coordinate_type;
    using domain_t           = domain::Domain<Dim,datablock_t>;
    using b_descriptor       = BlockDescriptor<int, Dim>;
    using base_t             = typename b_descriptor::base_t;;
    using extent_t           = typename b_descriptor::extent_t;;

    
    PoissonProblem(Dictionary* _d) 
    : simulation_(_d->get_dictionary("simulation_parameters")),
        lgf_(), conv(simulation_.domain_.block_extent(),
                                simulation_.domain_.block_extent())
    {
        pcout << "\n Setup:  LGF PoissonProblem \n" << std::endl;
        pcout << "Simulation: \n" << simulation_    << std::endl;

        const float_type L = simulation_.dictionary_->
            template get_or<float_type>("L", 1);

        auto tmp = L / (simulation_.domain_.bounding_box().extent()-1);
        dx = tmp[0];
        this->initialize();
    }                               
    
    
    /*
     * It initializes the Poisson problem using a manufactured solutions.
     */
    void initialize()
    {

        int count=0;
        for (auto it  = simulation_.domain_.begin_leafs();
                  it != simulation_.domain_.end_leafs(); ++it)
        {
            if (count++ ==0)simulation_.domain_.refine(it);

        }

        auto center = (simulation_.domain_.bounding_box().max() -
                       simulation_.domain_.bounding_box().min()) / 2.0 +
                       simulation_.domain_.bounding_box().min();

        const float_type a  = 10.;
        const float_type a2 = a*a;
   
        for (auto it  = simulation_.domain_.begin_leafs();
                  it != simulation_.domain_.end_leafs(); ++it)
        {
            
            auto dx_level =  dx/std::pow(2,it->real_level());
            auto scaling =  std::pow(2,it->real_level());


            // ijk-way of initializing
            auto base = it->data()->descriptor().base();
            auto max  = it->data()->descriptor().max();
            for (auto k = base[2]; k <= max[2]; ++k)
            {
                for (auto j = base[1]; j <= max[1]; ++j)
                {
                    for (auto i = base[0]; i <= max[0]; ++i)
                    {
                        it->data()->get<source>(i,j,k)  = 1.0;
                        it->data()->get<phi_num>(i,j,k) = 0.0;

                        
                        // manufactured solution:
                        float_type x = static_cast<float_type>(i-center[0]*scaling)*dx_level;
                        float_type y = static_cast<float_type>(j-center[1]*scaling)*dx_level;
                        float_type z = static_cast<float_type>(k-center[2]*scaling)*dx_level;
                        const auto x2 = x*x;
                        const auto y2 = y*y;
                        const auto z2 = z*z;


                        it->data()->get<source>(i,j,k) =
                            a*std::exp(-a*(x2)-a*(y2)-a*(z2))*(-6.0)+ 
                            (a2)*(x2)*std::exp(-a*(x2)-a*(y2)-a*(z2))*4.0 + 
                            (a2)*(y2)*std::exp(-a*(x2)-a*(y2)-a*(z2))*4.0+
                            (a2)*(z2)*std::exp(-a*(x2)-a*(y2)-a*(z2))*4.0;

                        it->data()->get<phi_exact>(i,j,k) =
                            std::exp((-a*x2 - a*y2 - a*z2));
                    }
                }
            }
        }
    }

    void level_test()
    {
        for(int l=simulation_.domain_.tree()->base_level(); 
                l< simulation_.domain_.tree()->depth();++l)
        {
            
            for(auto it=simulation_.domain_.begin(l); 
                    it!=simulation_.domain_.end(l);++it)
            {
                auto base = it->data()->descriptor().base();
                auto max  = it->data()->descriptor().max();
                for (auto k = base[2]; k <= max[2]; ++k)
                {
                    for (auto j = base[1]; j <= max[1]; ++j)
                    {
                        for (auto i = base[0]; i <= max[0]; ++i)
                        {
                            it->data()->get<dummy_field>(i,j,k)=it->real_level();
                        }
                    }
                }
            }
        }
    }

    void simple_lapace_fd()
    {
        //Only in interior for simplicity:
        for (auto it  = simulation_.domain_.begin_leafs();
                it != simulation_.domain_.end_leafs(); ++it)
        {

            auto base = it->data()->descriptor().base();
            auto max  = it->data()->descriptor().max();
            for (auto k = base[2]+1; k < max[2]; ++k)
            {
                for (auto j = base[1]+1; j < max[1]; ++j)
                {
                    for (auto i = base[0]+1; i < max[0]; ++i)
                    {
                        it->data()->get<lapace_field>(i,j,k) =
                            -6.0*it->data()->get<phi_num>(i,j,k)+ 
                                 it->data()->get<phi_num>(i+1,j,k)+ 
                                 it->data()->get<phi_num>(i-1,j,k)+
                                 it->data()->get<phi_num>(i,j+1,k)+
                                 it->data()->get<phi_num>(i,j-1,k)+
                                 it->data()->get<phi_num>(i,j,k+1)+
                                 it->data()->get<phi_num>(i,j,k-1);
                        it->data()->get<lapace_field>(i,j,k)/=dx*dx;
                        it->data()->get<lapace_error>(i,j,k)=
                            std::fabs(it->data()->get<lapace_field>(i,j,k)-
                                      it->data()->get<source>(i,j,k));
                    }
                }
            }
        }
    }


    
    /**
     *  It solves the Poisson problem with homogeneous boundary conditions
     *
     *  \nabla^2 \phi = s, on \Omega, with
     *  \phi|_{\partial\Omega} = 0,
     *
     *  via the LGF approach, that is: \phi = IFFT(FFT(G * s)), where
     *  - \phi: is the numerical solution of ,
     *  - G: is the lattice Green's function,
     *  - s: is the source term,
     *  - FFT: is the fast-Fourier transform,
     *  - IFFT: is the inverse of the FFT
     */
    void solve()
    {
        // allocate lgf
        std::vector<float_type> lgf;
        
        for (auto it_i  = simulation_.domain_.begin_leafs();
                  it_i != simulation_.domain_.end_leafs(); ++it_i)
        {
            auto i = std::distance(simulation_.domain_.begin_leafs(), it_i);
            pcout << "i     = " << i << std::endl;
            pcout << "level = " << it_i->data()->descriptor().level() << std::endl;
            pcout << "size  = " << it_i->data()->descriptor().size()  << std::endl;
            pcout << "index = " << it_i->index() << std::endl;



            
            for (auto it_j  = simulation_.domain_.begin_leafs();
                      it_j != simulation_.domain_.end_leafs(); ++it_j)
            {
                // Self-level convolutions (for blocks at the same level)
                if (it_i->data()->descriptor().level() ==
                    it_j->data()->descriptor().level())
                {
                    const auto ibase   = it_i->data()->descriptor().base();
                    const auto jbase   = it_j->data()->descriptor().base();
                    const auto jextent = it_j->data()->descriptor().extent();
                    const auto shift   = ibase - jbase;
                    
                    const auto base_lgf   = shift - (jextent - 1);
                    const auto extent_lgf = 2 * (jextent) - 1;
                
                    pcout << "SELF" << std::endl;
                    pcout << "base_lgf   = " << base_lgf   << std::endl;
                    pcout << "extent_lgf = " << extent_lgf << std::endl;
                    
                    lgf_.get_subblock(block_descriptor_t(base_lgf,
                                                         extent_lgf), lgf);

                    conv.execute(lgf, it_j->data()->get<source>().data());
                    block_descriptor_t extractor(jbase, jextent);
                    conv.add_solution(
                        extractor,
                        it_i->data()->get<phi_num>().data(),
                        dx*dx);
                }
                // Cross-level convolutions (for target finer than source)
                else if (it_i->data()->descriptor().level() >
                         it_j->data()->descriptor().level())
                {
                    //octant_t* parent_block = it_i->parent();

                    const auto ibase   = it_i->parent()->data()->descriptor().base();
                    const auto jbase   = it_j->data()->descriptor().base();
                    const auto jextent = it_j->data()->descriptor().extent();
                    const auto shift   = ibase - jbase;

                    const auto base_lgf   = shift - (jextent - 1);
                    const auto extent_lgf = (jextent) - 1;
                    
                    lgf_.get_subblock(block_descriptor_t(base_lgf,
                                                         extent_lgf), lgf);
                    
                    conv.execute(lgf, it_j->data()->get<source>().data());
                    block_descriptor_t extractor(jbase, jextent);
                    conv.add_solution(
                        extractor,
                        it_i->parent()->data()->get<phi_num>().data(),
                        dx*dx);
                    
                    this->interpolate(
                        it_i->parent()->data()->descriptor(),
                        it_i->parent()->data()->get<phi_num>().data(),
                        it_i->data()->get<phi_num>().data());
                    
                    // Advance to next block skipping children
                    std::advance (it_i, 8);
                    
                    pcout << "TARGET FINER" << std::endl;
                    pcout << "base_lgf   = " << base_lgf   << std::endl;
                    pcout << "extent_lgf = " << extent_lgf << std::endl;
                }
                // Cross-level convolutions (for target coarse than source)
                else
                {
                    const auto ibase = it_i->data()->descriptor().base();
                    const auto jbase   = it_j->data()->descriptor().base();
                    const auto jextent = it_j->data()->descriptor().extent();
                    const auto shift   = ibase - jbase;
                    
                    const auto base_lgf   = shift - (jextent - 1);
                    const auto extent_lgf = 2 * (jextent) - 1;
                    
                    pcout << "TARGET COARSER" << std::endl;
                    pcout << "base_lgf   = " << base_lgf   << std::endl;
                    pcout << "extent_lgf = " << extent_lgf << std::endl;
                }
            }
        }
        
        //simple_lapace_fd();
        compute_errors();
        level_test();
        pcout << "Writing solution " << std::endl;
        simulation_.write("solution.vtk");
    }


  
    
    /*
     * Interpolate a given field from corser to finer level.
     * Note: maximum jump allowed is one level.
     */
    template<class Block, class Field>
    void interpolate(const Block& _b, Field& F1, Field& F2)
    {
        int count1  = 0;
        pcout << "extent 2 = " << _b.extent()[2]-1 << std::endl;
        pcout << "extent 1 = " << _b.extent()[1]-1 << std::endl;
        pcout << "extent 0 = " << _b.extent()[0]-1 << std::endl;
        for (auto c = 0; c < 8; ++c)
        {
            int count20 = 0;
            int count21 = 1;
            for (int k = 0; k < _b.extent()[2]-1; ++k)
            {
                for (int j = 0; j < _b.extent()[1]-1; ++j)
                {
                    for (int i = 0; i < _b.extent()[0]-1; ++i)
                    {
                        // Copy value
                        F2[count20] += F1[count1];
                        F2[count21] += (F1[count1+1] + F1[count1-1]) / 2.0;
                    
                        count1++;
                        count20 += 2;
                        count21 += 2;
                    }
                }
            }
        }
    }
    
    
    
    /*
     * Coarsify given field from finer to coarser level.
     * Note: maximum jump allowed is one level.
     */
    void coarsify()
    {
        for (auto it_i  = simulation_.domain_.begin_leafs();
                  it_i != simulation_.domain_.end_leafs(); ++it_i)
        {
            if (it_i->is_hanging()) continue;
            
            for (std::size_t i = 0; i < it_i->data()->nodes().size(); ++i)
            {

            }
        }
    }
    
    
    
    /*
     * Calculate the L2 and LInf errors.
     */
    void compute_errors()
    {
        auto L2   = 0.;
        auto LInf = -1.0;

        for (auto it_i  = simulation_.domain_.begin_leafs();
             it_i != simulation_.domain_.end_leafs(); ++it_i)
        {
            if (it_i->is_hanging()) continue;

            for (std::size_t i = 0; i < it_i->data()->nodes().size(); ++i)
            {
               it_i->data()->get<error>().data()[i] = std::abs(
                    it_i->data()->get<phi_num>().data()[i] -
                    it_i->data()->get<phi_exact>().data()[i]);
                    
                it_i->data()->get<error2>().data()[i] =
                    it_i->data()->get<error>().data()[i] *
                    it_i->data()->get<error>().data()[i];
                    
                L2 += it_i->data()->get<error2>().data()[i];
                    
                if ( it_i->data()->get<error>().data()[i] > LInf)
                {
                    LInf = it_i->data()->get<error>().data()[i];
                }
            }
            pcout << "L2   = " << L2/it_i->data()->nodes().size() << std::endl;
            pcout << "LInf = " << LInf << std::endl;
        }
    }
    
private:

    Simulation<domain_t>              simulation_;
    parallel_ostream::ParallelOstream pcout;
    lgf::LGF<lgf::Lookup>             lgf_;
    Convolution                       conv;
    float_type                        dx;
};


#endif // IBLGF_INCLUDED_POISSON_HPP
