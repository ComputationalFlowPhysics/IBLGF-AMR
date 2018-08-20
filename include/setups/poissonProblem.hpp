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
    make_field_type(phi_num_tmp         , float_type)
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
        phi_num_tmp,
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
            //if (count++ ==0)simulation_.domain_.refine(it);

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
        for(int l  = simulation_.domain_.tree()->base_level();
                l < simulation_.domain_.tree()->depth();++l)
        {
            
            for(auto it  = simulation_.domain_.begin(l);
                     it != simulation_.domain_.end(l); ++it)
            {
                auto base = it->data()->descriptor().base();
                auto max  = it->data()->descriptor().max();
                for (auto k = base[2]; k <= max[2]; ++k)
                {
                    for (auto j = base[1]; j <= max[1]; ++j)
                    {
                        for (auto i = base[0]; i <= max[0]; ++i)
                        {
                            it->data()->get<dummy_field>(i,j,k) = it->real_level();
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
            const auto ibase= it_i->data()->descriptor().base();
            
            for (auto it_j  = simulation_.domain_.begin_leafs();
                 it_j != simulation_.domain_.end_leafs(); ++it_j)
            {
                
                const auto jbase   = it_j->data()->descriptor().base();
                const auto jextent = it_j->data()->descriptor().extent();
                const auto shift   = ibase - jbase;
                
                const auto base_lgf   = shift - (jextent - 1);
                const auto extent_lgf = 2 * (jextent) - 1;
                
                lgf_.get_subblock(block_descriptor_t(base_lgf,
                                                     extent_lgf), lgf);
                
                conv.execute(lgf, it_j->data()->get<source>().data());
                block_descriptor_t extractor(jbase, jextent);
                conv.add_solution(extractor,
                                  it_i->data()->get<phi_num>().data(), dx*dx);
            }
        }
        
        //simple_lapace_fd();
        compute_errors();
        pcout << "Writing solution " << std::endl;
        simulation_.write("solution.vtk");
    }


  
    
    /*
     * Interpolate a given field from corser to finer level.
     * Note: maximum jump allowed is one level.
     */
    template<class Block_it>
    void interpolate(const Block_it* _b_parent)
    {
        
        for (int i = 0; i < _b_parent->num_children(); ++i)
        {
            auto _b_child = _b_parent->child(i);
            
            auto ic = _b_child->data()->descriptor().base()[0];
            auto jc = _b_child->data()->descriptor().base()[1];
            auto kc = _b_child->data()->descriptor().base()[2];
            
            // Loops on coordinates
            for (auto kp  = _b_parent->data()->descriptor().base()[2];
                 kp < _b_parent->data()->descriptor().max()[2]; ++kp)
            {
                for (auto jp  = _b_parent->data()->descriptor().base()[1];
                     jp < _b_parent->data()->descriptor().max()[1]; ++jp)
                {
                    for (auto ip  = _b_parent->data()->descriptor().base()[0];
                         ip < _b_parent->data()->descriptor().max()[0]; ++ip)
                    {
                        
                        _b_child->data()->template get<phi_num>(ic,jc,kc) +=
                            _b_parent->data()->template get<phi_num_tmp>(ip,jp,kp);
                        
                        _b_child->data()->template get<phi_num>(ic+1,jc+1,kc+1) +=
                            (_b_parent->data()->template get<phi_num_tmp>(ip+1,jp+1,kp+1) +
                             _b_parent->data()->template get<phi_num_tmp>(ip,jp,kp)) / 2;
                        
                        ic+=2;
                        jc+=2;
                        kc+=2;
                    }
                }
            }
            
        }
    }
    
    
    
    /*
     * Coarsify given field from finer to coarser level.
     * Note: maximum jump allowed is one level.
     */
    template<class Block_it>
    void coarsify(const Block_it& _b)
    {
        auto _b_child  = _b;
        auto _b_parent = _b_child->parent();

        auto ip = _b_parent->data()->descriptor().base()[0];
        auto jp = _b_parent->data()->descriptor().base()[1];
        auto kp = _b_parent->data()->descriptor().base()[2];
        
        // Loops on coordinates
        for (auto kc  = _b_child->data()->descriptor().base()[2];
                  kc <= _b_child->data()->descriptor().max()[2]; kc+=2)
        {
            for (auto jc  = _b_child->data()->descriptor().base()[1];
                      jc <= _b_child->data()->descriptor().max()[1]; jc+=2)
            {
                for (auto ic  = _b_child->data()->descriptor().base()[0];
                          ic <= _b_child->data()->descriptor().max()[0]; ic+=2)
                {
                    _b_parent->data()->template get<source>(ip,jp,kp) =
                        _b_child->data()->template get<source>(ic,jc,kc);
                    
                    pcout << "b_parent = " << _b_parent->data()->template get<source>(ip,jp,kp) << std::endl;
                    pcout << "b_child  = " << _b_child->data()->template get<source>(ic,jc,kc) << "    " <<_b_child->data()->template get<source>(ic+1,jc+1,kc+1) << std::endl;
                    ip+=1;
                    jp+=1;
                    kp+=1;
                }
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

        for (auto it_t  = simulation_.domain_.begin_leafs();
             it_t != simulation_.domain_.end_leafs(); ++it_t)
        {
            if (it_t->is_hanging()) continue;

            for (std::size_t i = 0; i < it_t->data()->nodes().size(); ++i)
            {
               it_t->data()->get<error>().data()[i] = std::abs(
                    it_t->data()->get<phi_num>().data()[i] -
                    it_t->data()->get<phi_exact>().data()[i]);
                    
                it_t->data()->get<error2>().data()[i] =
                    it_t->data()->get<error>().data()[i] *
                    it_t->data()->get<error>().data()[i];
                    
                L2 += it_t->data()->get<error2>().data()[i];
                    
                if ( it_t->data()->get<error>().data()[i] > LInf)
                {
                    LInf = it_t->data()->get<error>().data()[i];
                }
            }
            pcout << "L2   = " << L2/it_t->data()->nodes().size() << std::endl;
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
