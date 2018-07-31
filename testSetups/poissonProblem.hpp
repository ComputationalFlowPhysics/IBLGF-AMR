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
    using vel_type = vector_type<float_type, Dim>;
    using size_v_type = vector_type<int, Dim>;

    //              name                type
    make_field_type(phi_num           , float_type)
    make_field_type(source            , float_type)
    make_field_type(lgf_field_integral, float_type)
    make_field_type(lgf_field_lookup  , float_type)
    make_field_type(phi_exact         , float_type)

    make_field_type(lgf         , float_type)

    using datablock_t = DataBlock<
        Dim, node,
        phi_num,
        source,
        lgf_field_integral,
        lgf_field_lookup,
        phi_exact
    >;

    using datablock_t_2= DataBlock<Dim, node, lgf>;

    using block_descriptor_t = typename datablock_t::block_descriptor_type;
    using tree_t             = Tree<Dim,datablock_t>;
    using octant_t           = typename tree_t::octant_type;
    using coordinate_t       = typename datablock_t::coordinate_type;
    using domain_t           = domain::Domain<Dim,datablock_t>;
    using b_descriptor       = BlockDescriptor<int, Dim>;
    using base_t             = typename b_descriptor::base_t;;
    using extent_t           = typename b_descriptor::extent_t;;

    
    PoissonProblem(Dictionary* _d) 
    : simulation_( _d->get_dictionary("simulation_parameters")),
      lgf_(lgf_block()),
      conv(simulation_.domain_.block_extent(),simulation_.domain_.block_extent())
    {
        pcout << "\n Setup:  LGF PoissonProblem \n" << std::endl;
        pcout << "Simulation: \n" << simulation_   << std::endl;
        this->initialize();
    }                               

    block_descriptor_t lgf_block()
    {
        base_t bb(-20);
        extent_t ex=-2*bb+1;
        return block_descriptor_t(bb, ex);
    }

    void initialize()
    {
        int count = 0, ocount = 0;
        simulation_.domain_.tree()->determine_hangingOctants();


        for(auto it  = simulation_.domain_.begin_octants(); 
                 it != simulation_.domain_.end_octants();++it)
        {
            if(it->is_hanging()) continue;
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
            auto base=it->data()->descriptor().base();
            auto max=it->data()->descriptor().max();
            for(auto  k =base[2];k<=max[2];++k)
            {
                for(auto j =base[1];j<=max[1];++j)
                {

                    for(auto i=base[0]; i<=max[0];++i  )
                    {
                        it->data()->get<source>(i,j,k)=1.0;
                    }
                }
            }
        }
    }


    void solve()
    {
        
        //Allocate lgf
        std::vector<float_type> lgf;
        const auto bextent=simulation_.domain_.block_extent();
        for(auto it_i  = simulation_.domain_.begin_octants();
                it_i != simulation_.domain_.end_octants();++it_i)
        {
            if(it_i->is_hanging()) continue;
            const auto ibase= it_i->data()->descriptor().base();

            for(auto it_j  = simulation_.domain_.begin_octants();
                    it_j != simulation_.domain_.end_octants();++it_j)
            {
                if(it_j->is_hanging()) continue;

                const auto jbase= it_j->data()->descriptor().base();
                const auto shift = jbase-ibase;

                const auto base_lgf=shift-bextent;
                const auto extent_lgf =2*bextent-1;
                //extract lgfs:
                lgf_.get_subblock(block_descriptor_t (base_lgf,extent_lgf), lgf);
                conv.execute(lgf,it_i->data()->get<source>().data());
                
            }
        }
        simulation_.write("bla.vtk");
    }

private:

    Simulation<domain_t> simulation_;
    parallel_ostream::ParallelOstream pcout;
    lgf::LGF<lgf::Integrator> lgf_;
    Convolution conv;
};


#endif // IBLGF_INCLUDED_POISSON_HPP
