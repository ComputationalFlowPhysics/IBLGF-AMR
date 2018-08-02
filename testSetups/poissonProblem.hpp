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
#include <functional>

#include <fftw3.h>

// IBLGF-specific
#include <global.hpp>
#include <simulation.hpp>
#include <domain/domain.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <domain/octree/tree.hpp>
#include <io/parallel_ostream.hpp>
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
    make_field_type(phi_num           , float_type)
    make_field_type(source            , float_type)
    make_field_type(lgf_field_integral, float_type)
    make_field_type(lgf_field_lookup  , float_type)
    make_field_type(phi_exact         , float_type)
    make_field_type(lgf               , float_type)
    make_field_type(error             , float_type)
    make_field_type(error2            , float_type)



    using datablock_t = DataBlock<
        Dim, node,
        phi_num,
        source,
        lgf_field_integral,
        lgf_field_lookup,
        phi_exact,
        error,
        error2
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
        lgf_(lgf_block()), conv(simulation_.domain_.block_extent()+1,
                                simulation_.domain_.block_extent()+1)
    {
        pcout << "\n Setup:  LGF PoissonProblem \n" << std::endl;
        pcout << "Simulation: \n" << simulation_    << std::endl;
        this->initialize();
    }                               

    block_descriptor_t lgf_block()
    {
        base_t bb(-20);
        extent_t ex = -2 * bb + 1;
        return block_descriptor_t(bb, ex);
    }

    
    
    void initialize()
    {
        int count = 0, ocount = 0;
        simulation_.domain_.tree()->determine_hangingOctants();
        
        float_type L = simulation_.dictionary_->
            template get_or<float_type>("refLength", 1);

        const auto center = (simulation_.domain_.bounding_box().max() -
                             simulation_.domain_.bounding_box().min()) / 2.0;
        
        L = L * center[0];
        
        const float_type a  = 1.0 / L;
        const float_type a2 = a*a;
        const auto b = a; const auto b2 = a2;
        const auto c = a; const auto c2 = a2;


        for (auto it  = simulation_.domain_.begin_octants();
                  it != simulation_.domain_.end_octants(); ++it)
        {
            if (it->is_hanging()) continue;
            ++ocount;
            
            //iterate over nodes:
            for (auto& n : it->data()->nodes())
            {
                count++;
                n.get<source>() = count;
                lgf::LGF<lgf::Integrator> lgfsI;
                n.get<lgf_field_integral>() = lgfsI.get(n.level_coordinate());
            }

            //ijk- way of initializing 
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
                        const float_type x = static_cast<float_type>(i-center[0]);
                        const float_type y = static_cast<float_type>(j-center[1]);
                        const float_type z = static_cast<float_type>(k-center[2]);
                        const auto x2 = x*x;
                        const auto y2 = y*y;
                        const auto z2 = z*z;

                        it->data()->get<source>(i,j,k)=
                            4 * a2 * x2 * std::exp(-a*x2 - b*y2 - c*z2) -
                            2 * b  *      std::exp(-a*x2 - b*y2 - c*z2) -
                            2 * c  *      std::exp(-a*x2 - b*y2 - c*z2) -
                            2 * a  *      std::exp(-a*x2 - b*y2 - c*z2) +
                            4 * b2 * y2 * std::exp(-a*x2 - b*y2 - c*z2) +
                            4 * c2 * z2 * std::exp(-a*x2 - b*y2 - c*z2);

                        it->data()->get<phi_exact>(i,j,k) =
                            std::exp((-a*x2 - b*y2 - c*z2));
                    }
                }
            }
        }
    }


    
    void solve()
    {
        // allocate lgf
        std::vector<float_type> lgf, result;
        for (auto it_i  = simulation_.domain_.begin_octants();
                  it_i != simulation_.domain_.end_octants(); ++it_i)
        {
            if (it_i->is_hanging()) continue;
            const auto ibase= it_i->data()->descriptor().base();

            for (auto it_j  = simulation_.domain_.begin_octants();
                      it_j != simulation_.domain_.end_octants(); ++it_j)
            {
                if (it_j->is_hanging()) continue;

                const auto jbase   = it_j->data()->descriptor().base();
                const auto jextent = it_j->data()->descriptor().extent();
                const auto shift   = jbase - ibase;

                const auto base_lgf   = shift - (jextent - 1);
                const auto extent_lgf = 2 * (jextent) - 1;
                
                lgf_.get_subblock(block_descriptor_t(base_lgf, extent_lgf), lgf);
                std::cout<<" size lgf "<<lgf.size()<<std::endl;

                conv.execute(lgf, it_i->data()->get<source>().data());
                block_descriptor_t extractor(jbase, jextent);
                auto result = conv.res(extractor);
                auto L2   = 0.;
                auto LInf = 0.;
                
                for (std::size_t i = 0; i < result.size(); ++i)
                {
                    it_i->data()->get<phi_num>().data()[i] += result[i];
                    
                    it_i->data()->get<error>().data()[i] = std::abs(
                        it_i->data()->get<phi_num>().data()[i] -
                        it_i->data()->get<phi_exact>().data()[i]);
                    
                    it_i->data()->get<error2>().data()[i] =
                        it_i->data()->get<error>().data()[i] *
                        it_i->data()->get<error>().data()[i];
                    
                    L2 += it_i->data()->get<error2>().data()[i];
                    
                    if (i > 0 &&
                        it_i->data()->get<error>().data()[i] >
                        it_i->data()->get<error>().data()[i-1])
                    {
                        LInf = it_i->data()->get<error>().data()[i];
                    }
                }
                pcout << "L2   = " << L2   << std::endl;
                pcout << "LInf = " << LInf << std::endl;
            }
        }
        simulation_.write("solution.vtk");
    }

    void compute_errors()
    {
        auto L2   = 0.;
        auto LInf = 0.;

        for (auto it_i  = simulation_.domain_.begin_octants();
             it_i != simulation_.domain_.end_octants(); ++it_i)
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
                    
                if (i > 0 &&
                    it_i->data()->get<error>().data()[i] >
                    it_i->data()->get<error>().data()[i-1])
                {
                    LInf = it_i->data()->get<error>().data()[i];
                }
            }
            pcout << "L2   = " << L2   << std::endl;
            pcout << "LInf = " << LInf << std::endl;
        }
    }
    
private:

    Simulation<domain_t>              simulation_;
    parallel_ostream::ParallelOstream pcout;
    lgf::LGF<lgf::Integrator>         lgf_;
    Convolution                       conv;
};


#endif // IBLGF_INCLUDED_POISSON_HPP
