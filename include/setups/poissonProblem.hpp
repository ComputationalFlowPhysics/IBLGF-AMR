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

    //Field with buffer:
    make_field_type(bla_field  , float_type, 1, 2)

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
        dummy_field, 
        bla_field     >;

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
            
            auto dx_level =  dx/std::pow(2,it->refinement_level());
            auto scaling =  std::pow(2,it->refinement_level());


            // ijk-way of initializing
            auto base = it->data()->base();
            auto max  = it->data()->max();
            for (auto k = base[2]; k <= max[2]; ++k)
            {
                for (auto j = base[1]; j <= max[1]; ++j)
                {
                    for (auto i = base[0]; i <= max[0]; ++i)
                    {
                        it->data()->get<source>(i,j,k)  = 1.0;
                        it->data()->get<phi_num>(i,j,k) = 0.0;
                        it->data()->get<phi_num_tmp>(i,j,k) = 0.0;
                        
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
                auto base = it->data()->base();
                auto max  = it->data()->max();
                for (auto k = base[2]; k <= max[2]; ++k)
                {
                    for (auto j = base[1]; j <= max[1]; ++j)
                    {
                        for (auto i = base[0]; i <= max[0]; ++i)
                        {
                            it->data()->get<dummy_field>(i,j,k) = it->refinement_level();
                        }
                    }
                }
            }
        }
    }

    void buffer_test()
    {
        //
        for (auto it  = simulation_.domain_.begin_leafs();
                it != simulation_.domain_.end_leafs(); ++it)
        {

            //make a box-overlap check to determine buffers
            auto base = it->data()->base();
            auto max  = it->data()->max();
            for (auto k = base[2]-1; k <= max[2]+1; ++k)
            {
                for (auto j = base[1]-1; j <= max[1]+1; ++j)
                {
                    for (auto i = base[0]-1; i <= max[0]+1; ++i)
                    {
                        it->data()->get<bla_field>(i,j,k) = it->refinement_level();
                    }
                }
            }
        }
    }


    void neighbor_test()
    {

        int count=0;
        for (auto it  = simulation_.domain_.begin_leafs();
                it != simulation_.domain_.end_leafs(); ++it)
        {
            coordinate_t direction(0);
            direction[0]=+1;
            auto nn=it->vertex_neighbor(direction);
            if(nn!=nullptr)
            {

                std::ofstream ofs("point_"+std::to_string(count)+".txt");
                std::ofstream ofs1("nn_"+std::to_string(count)+".txt");
                ofs<<it->global_coordinate()<<std::endl;
                ofs1<<nn->global_coordinate()<<std::endl;
                ++count;
            }
            else
            {
                std::cout<<"could not find neighbors"<<std::endl;
            }
        }
    }

    void neighborhood_test()
    {

        int count=0;
        for (auto it  = simulation_.domain_.begin_leafs();
                it != simulation_.domain_.end_leafs(); ++it)
        {
            coordinate_t lowBuffer(1);
            coordinate_t highBuffer(2);
            auto neighborhood = it->get_neighborhood(lowBuffer, highBuffer); 

            if(neighborhood.size()!=0)
            {
                std::ofstream ofs("point_nh__"+std::to_string(count)+".txt");
                ofs<<it->global_coordinate()<<std::endl;
                std::ofstream ofs1("nh_"+std::to_string(count)+".txt");
                for(auto& e:neighborhood) 
                {
                    ofs1<<e->global_coordinate()<<std::endl;
                }
                count++;
            }
            else
            {
                std::cout<<"empty neighborhood"<<std::endl;
            }
        }
    }

    void simple_lapace_fd()
    {
        //Only in interior for simplicity:
        for (auto it  = simulation_.domain_.begin_leafs();
                  it != simulation_.domain_.end_leafs(); ++it)
        {

            auto base = it->data()->base();
            auto max  = it->data()->max();
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
     *  \brief It solves the Poisson problem with homogeneous boundary conditions
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
        
        // Cross-level interactions (source is finer, target is coarser)
        // This should take care of the cross-level interactions, that
        // are computed as part of the self-interactions, because parents
        // are included.
        for (int lt = simulation_.domain_.tree()->base_level();
                 lt < simulation_.domain_.tree()->depth(); ++lt)
        {
            for (int ls = lt+1;
                     ls < simulation_.domain_.tree()->depth(); ++ls)
            {
                pcout << "--------- CROSS-INTERACTION (SOURCE FINER) --------" << std::endl;
                pcout << "BASE-LEVEL = " << simulation_.domain_.tree()->base_level() << std::endl;
                pcout << "======== TARGET BLOCK LEVEL = " << lt
                << ",        SOURCE BLOCK LEVEL = " << ls
                << std::endl;
                
                for (auto it_s  = simulation_.domain_.begin(ls);
                          it_s != simulation_.domain_.end(ls); ++it_s)
                {
                   this->coarsify(it_s);
                }
            }
        }
        
        // Self-level interactions
        for (int l  = simulation_.domain_.tree()->base_level();
                 l <= simulation_.domain_.tree()->depth(); ++l)
        {
            auto target = 0;
            for (auto it_t  = simulation_.domain_.begin(l);
                      it_t != simulation_.domain_.end(l); ++it_t)
            {
                auto refinement_level = it_t->refinement_level();
                auto dx_level =  dx/std::pow(2,refinement_level);

                pcout << "--------- SELF-INTERACTION --------" << std::endl;
                pcout << "======== TARGET BLOCK = "        << target
                      << ",        TARGET REAL LEVEL = " << refinement_level
                      << std::endl;
                
                for (auto it_s  = simulation_.domain_.begin(l);
                          it_s != simulation_.domain_.end(l); ++it_s)
                {
                    // Get the coordinate of target block
                    const auto t_base = simulation_.domain_.tree()->
                        octant_to_level_coordinate(it_t->tree_coordinate());
                    
                    // Get tree_coordinate of source block
                    const auto s_base = simulation_.domain_.tree()->
                        octant_to_level_coordinate(it_s->tree_coordinate());
                    
                    // Get extent of source region
                    const auto s_extent = it_s->data()->extent();
                    const auto shift    = t_base - s_base;
                    
                    // Calculate the dimensions of the LGF to be allocated
                    const auto base_lgf   = shift - (s_extent - 1);
                    const auto extent_lgf = 2 * (s_extent) - 1;
                    
                    // Calculate the LGF
                    lgf_.get_subblock(block_descriptor_t(base_lgf,
                                                         extent_lgf), lgf);
                    
                    // Perform convolution
                    conv.execute(lgf, it_s->data()->get<source>().data());
                    
                    // Extract the solution
                    block_descriptor_t extractor(s_base, s_extent);
                    conv.add_solution(extractor,
                                      it_t->data()->get_data<phi_num>(),
                                      dx_level*dx_level);
                }
                target++;
            }
        }
        
        // Cross-level interactions (source is coarser, target is finer)
        for (int lt  = simulation_.domain_.tree()->depth()-1;
                 lt >= simulation_.domain_.tree()->base_level(); --lt)
        {
            for (int ls = lt-1; ls >= simulation_.domain_.tree()->base_level();
                 --ls)
            {
                pcout << "--------- CROSS-INTERACTION (SOURCE COARSER) --------" << std::endl;
                pcout << "======== TARGET BLOCK LEVEL = " << lt
                << ",        SOURCE BLOCK LEVEL = " << ls
                << std::endl;
                
                for (auto it_t  = simulation_.domain_.begin(lt);
                          it_t != simulation_.domain_.end(lt); ++it_t)
                {
                    auto refinement_level = it_t->refinement_level();
                    auto dx_level   = dx/std::pow(2,refinement_level-1);
                    
                    for (auto it_s  = simulation_.domain_.begin(ls);
                         it_s != simulation_.domain_.end(ls); ++it_s)
                    {
                        // Get the tree_coordinate of target block
                        const auto t_base_parent = simulation_.domain_.tree()->
                        octant_to_level_coordinate(it_t->parent()->tree_coordinate());
                        
                        // Get tree_coordinate of source block
                        const auto s_base = simulation_.domain_.tree()->
                        octant_to_level_coordinate(it_s->tree_coordinate());
                        
                        // Get extent of source region
                        const auto s_extent = it_s->data()->extent();
                        const auto shift    = t_base_parent - s_base;
                        
                        // Calculate the dimensions of the LGF to be allocated
                        const auto base_lgf   = shift - (s_extent - 1);
                        const auto extent_lgf = 2 * (s_extent) - 1;
                        
                        lgf_.get_subblock(block_descriptor_t(base_lgf,
                                                             extent_lgf), lgf);
                        
                        conv.execute(lgf, it_s->data()->get_data<source>());
                        block_descriptor_t extractor(s_base, s_extent);
                        std::cout<<it_t->parent()->data()->descriptor()<<std::endl;
                        conv.add_solution(
                            extractor,
                            it_t->parent()->data()->get_data<phi_num>(),
                            dx_level*dx_level);
                        
                    }
                    this->interpolate(it_t->parent());
                    std::advance(it_t,8);
                }
            }
        }
        
        //simple_lapace_fd();
        compute_errors();
        pcout << "Writing solution " << std::endl;
        simulation_.write("solution.vtk");
    }

    void solve2()
    {
        neighborhood_test();
        buffer_test();
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

            auto ip = _b_parent->data()->base()[0];
            auto jp = _b_parent->data()->base()[1];
            auto kp = _b_parent->data()->base()[2];

            // Loops on coordinates
            for (auto kc  = _b_child->data()->base()[2];
                       kc < _b_child->data()->max()[2]; ++kc)
            {
                for (auto jc  = _b_child->data()->base()[1];
                        jc < _b_child->data()->max()[1]; ++jc)
                {
                    for (auto ic  = _b_child->data()->base()[0];
                            ic < _b_child->data()->max()[0]; ++ic)
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

        auto ip = _b_parent->data()->base()[0];
        auto jp = _b_parent->data()->base()[1];
        auto kp = _b_parent->data()->base()[2];
        
        // Loops on coordinates
        for (auto kc  = _b_child->data()->base()[2];
                  kc <= _b_child->data()->max()[2]; kc+=2)
        {
            for (auto jc  = _b_child->data()->base()[1];
                      jc <= _b_child->data()->max()[1]; jc+=2)
            {
                for (auto ic  = _b_child->data()->base()[0];
                          ic <= _b_child->data()->max()[0]; ic+=2)
                {
                    _b_parent->data()->template get<source>(ip,jp,kp) =
                        _b_child->data()->template get<source>(ic,jc,kc);
                    ++ip;
                }
                ++jp;
            }
            ++kp;
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

            for (std::size_t i = 0; i < it_t->data()->nodes().size(); ++i)
            {
               it_t->data()->get<error>()[i] = std::abs(
                    it_t->data()->get<phi_num>()[i] -
                    it_t->data()->get<phi_exact>()[i]);
                    
                it_t->data()->get<error2>()[i] =
                    it_t->data()->get<error>()[i] *
                    it_t->data()->get<error>()[i];
                    
                L2 += it_t->data()->get<error2>()[i];
                    
                if ( it_t->data()->get<error>()[i] > LInf)
                {
                    LInf = it_t->data()->get<error>()[i];
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
