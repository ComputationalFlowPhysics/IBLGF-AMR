#ifndef IBLGF_INCLUDED_DECOMPOSITION_TEST_HPP
#define IBLGF_INCLUDED_DECOMPOSITION_TEST_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fftw3.h>

// IBLGF-specific
#include <global.hpp>
#include <simulation.hpp>
#include <domain/domain.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <domain/octree/tree.hpp>
#include <IO/parallel_ostream.hpp>
#include <lgf/lgf.hpp>
#include <fmm/fmm.hpp>

#include<utilities/convolution.hpp>
#include<utilities/interpolation.hpp>
#include<solver/poisson/poisson.hpp>


const int Dim = 3;

using namespace domain;
using namespace octree;
using namespace types;
using namespace dictionary;
using namespace fft;

struct DecomposistionTest
{

    static constexpr int Dim = 3;

    //              name            type     lBuffer.  hBuffer
    make_field_type(phi_num         , float_type, 1,       1)
    make_field_type(source          , float_type, 1,       1)
    make_field_type(phi_exact       , float_type, 1,       1)
    make_field_type(error           , float_type, 1,       1)

    // temporarily here for FMM
    make_field_type(fmm_s           , float_type, 1,       1)
    make_field_type(fmm_t           , float_type, 1,       1)
    make_field_type(fmm_tmp         , float_type, 1,       1)
    make_field_type(phi_num_fmm     , float_type, 1,       1)

    using datablock_t = DataBlock<
        Dim, node,
        phi_num,
        source,
        phi_exact,
        error,
        fmm_s,
        fmm_t,
        fmm_tmp,
        phi_num_fmm
        >;


    using coordinate_t       = typename datablock_t::coordinate_type;
    using domain_t           = domain::Domain<Dim,datablock_t>;
    using simulation_type    = Simulation<domain_t>;
    using node_type          = typename datablock_t::node_t;
    using node_field_type    = typename datablock_t::node_field_type;


    DecomposistionTest(Dictionary* _d)
    :simulation_(_d->get_dictionary("simulation_parameters")),
     domain_(simulation_.domain_)
    {
        pcout << "\n Setup:  LGF ViewTest \n" << std::endl;
        pcout << "Simulation: \n" << simulation_    << std::endl;
        this->initialize();
        domain_.distribute();
    }


    /** @brief Initialization of poisson problem.
     *  @detail Testing poisson with manufactured solutions: exp
     */
    void initialize()
    {
        this->pcout<<"Initializing"<<std::endl;
        auto center = (domain_.bounding_box().max() -
                       domain_.bounding_box().min()-1) / 2.0 +
                       domain_.bounding_box().min();

        const int nRef = simulation_.dictionary_->
            template get_or<int>("nLevels",0);
        for(int l=0;l<nRef;++l)
        {
            for (auto it  = domain_.begin_leafs();
                    it != domain_.end_leafs(); ++it)
            {
                auto b=it->data()->descriptor();
                coordinate_t lower(2), upper(2);
                b.grow(lower, upper);
                if(b.is_inside( std::pow(2.0,l)*center )
                   && it->refinement_level()==l
                  )
                {
                    domain_.refine(it);
                }
            }
        }

    }


    /** @brief Run poisson test case, compute errors and write out.  */
    void run()
    {


    }


private:

    Simulation<domain_t>              simulation_;
    domain_t&                         domain_;

    parallel_ostream::ParallelOstream pcout;
    lgf::LGF<lgf::Lookup>             lgf_;
};


#endif // IBLGF_INCLUDED_POISSON_HPP
