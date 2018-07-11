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


#include "global.hpp"
#include "domain/domain.hpp"
#include "domain/dataFields/dataBlock.hpp"
#include "domain/dataFields/datafield.hpp"
#include "domain/octree/tree.hpp"
#include "simulation.hpp"
#include "io/parallel_ostream.hpp"
#include "lgf/lgf.hpp"


const int Dim=3;

using namespace domain;
using namespace octree;
using namespace types;
using namespace dictionary;

struct PoissonSolver
{
    using vel_type = vector_type<float_type, Dim>;

    //              name               type
    make_field_type(phi,             float_type)
    make_field_type(f,               float_type)
    make_field_type(lgf_field,       float_type)

    using datablock_t = DataBlock<Dim, node, phi, f, lgf_field>;

    using block_descriptor_t = typename datablock_t::block_descriptor_type;

    using tree_t = Tree<Dim,datablock_t>;
    using octant_t = typename tree_t::octant_type;

    using coordinate_t=  typename datablock_t ::coordinate_type;

    using domain_t = domain::Domain<Dim,datablock_t>;
    
    PoissonSolver(Dictionary* _d)
    :simulation_(_d->get_dictionary("simulation_parameters"))
    {
        pcout<<"\n Setup:  LGF PoissonSolver \n"<<std::endl;
        pcout<<"Simulation: \n"<<simulation_<<std::endl;
        this->initialize();
    }                                 

    void initialize()
    {
        //Refine:
        int count=0, ocount=0;
        for(auto it = simulation_.domain_.begin_octants(); 
                 it!= simulation_.domain_.end_octants();++it)
        {
            if(it->is_hanging())continue;
            ++ocount;
            for(auto& n : it->data()->nodes())
            {
                count++;
                n.get<phi>()=count;
                Bessel lgf_lookup;
                n.get<lgf_field>() = lgf_lookup.retrieve(
                        n.level_coordinate().x(),
                        n.level_coordinate().y(), 
                        n.level_coordinate().z()  );
            }
        }
        pcout<<"Total number of nodes: "<<count<<std::endl;
        pcout<<" number of non-hanging octants: "<<ocount<<std::endl;
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
