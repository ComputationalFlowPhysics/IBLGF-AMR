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


const int Dim = 3;

using namespace domain;
using namespace octree;
using namespace types;
using namespace dictionary;

struct PoissonProblem
{
    using vel_type = vector_type<float_type, Dim>;

    //              name                type
    make_field_type(phi_num           , float_type)
    make_field_type(source            , float_type)
    make_field_type(lgf_field_integral, float_type)
    make_field_type(lgf_field_lookup  , float_type)
    make_field_type(phi_exact         , float_type)

    using datablock_t = DataBlock<
        Dim, node,
        phi_num,
        source,
        lgf_field_integral,
        lgf_field_lookup,
        phi_exact
    >;

    using block_descriptor_t = typename datablock_t::block_descriptor_type;
    using tree_t             = Tree<Dim,datablock_t>;
    using octant_t           = typename tree_t::octant_type;
    using coordinate_t       = typename datablock_t::coordinate_type;
    using domain_t           = domain::Domain<Dim,datablock_t>;
    
    PoissonProblem(Dictionary* _d) : simulation_(
        _d->get_dictionary("simulation_parameters"))
    {
        pcout << "\n Setup:  LGF PoissonProblem \n" << std::endl;
        pcout << "Simulation: \n" << simulation_   << std::endl;
        this->initialize();
    }                                 

    void initialize()
    {
        //Refine:
        int count = 0, ocount = 0;
        for (auto it = simulation_.domain_.begin_octants();
                  it!= simulation_.domain_.end_octants(); ++it)
        {
            if (it->is_hanging())continue;
            if (ocount == 0 || ocount == 3)
            {
                simulation_.domain_.refine(it);
            }
            ++ocount;
        }
        simulation_.domain_.tree()->determine_hangingOctants();


        for(auto it  = simulation_.domain_.begin_octants(); 
                 it != simulation_.domain_.end_octants();++it)
        {
            if(it->is_hanging()) continue;
            ++ocount;
            for (auto& n : it->data()->nodes())
            {
                count++;
                n.get<phi_num>() = count;
                
                lgf::LGF<lgf::Integrator> lgfsI;
                n.get<lgf_field_integral>() = lgfsI.get(n.level_coordinate());
                
                lgf::LGF<lgf::Lookup> lgfsL;
                n.get<lgf_field_lookup>() = lgfsL.get(n.level_coordinate());
            }
        }
        pcout << " Total number of nodes        : " << count  << std::endl;
        pcout << " number of non-hanging octants: " << ocount << std::endl;
    }
    
    
    
    void run()
    {
        simulation_.write("bla.vtk");
    }
    
    
    
private:

    Simulation<domain_t> simulation_;
    parallel_ostream::ParallelOstream pcout;
};


#endif // IBLGF_INCLUDED_POISSON_HPP
