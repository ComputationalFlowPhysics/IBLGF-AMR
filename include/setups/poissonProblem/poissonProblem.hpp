#ifndef IBLGF_INCLUDED_POISSON_HPP
#define IBLGF_INCLUDED_POISSON_HPP

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

struct PoissonProblem 
{

    static constexpr int Dim = 3;

    //              name            type     lBuffer.  hBuffer
    make_field_type(phi_num         , float_type, 1,       1)
    make_field_type(source          , float_type, 1,       1)
    make_field_type(phi_exact       , float_type, 1,       1)
    make_field_type(error           , float_type, 1,       1)

    using datablock_t = DataBlock<
        Dim, node,
        phi_num,
        source,
        phi_exact,
        error
        >;


    using coordinate_t       = typename datablock_t::coordinate_type;
    using domain_t           = domain::Domain<Dim,datablock_t>;
    using simulation_type    = Simulation<domain_t>;
    using node_type          = typename datablock_t::node_t; 
    using node_field_type    = typename datablock_t::node_field_type; 


    PoissonProblem(Dictionary* _d)
    :simulation_(_d->get_dictionary("simulation_parameters")),
     domain_(simulation_.domain_)
    {
        pcout << "\n Setup:  LGF ViewTest \n" << std::endl;
        pcout << "Simulation: \n" << simulation_    << std::endl;
        this->initialize();
    }


    /** @brief Initialization of poisson problem.
     *  @detail Testing poisson with manufactured solutions: exp
     */
    void initialize()
    {
        this->pcout<<"Initializing"<<std::endl;
        auto center = (domain_.bounding_box().max() -
                       domain_.bounding_box().min()) / 2.0 +
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

        for (int lt = domain_.tree()->base_level(); 
                 lt < domain_.tree()->depth(); ++lt)
        {
            for (auto it  = domain_.begin(lt);
                      it != domain_.end(lt); ++it)
            {
                if(it->data())
                {
                    for(auto& e: it->data()->get_data<source>())
                        e=0.0;
                    for(auto& e: it->data()->get_data<phi_num>())
                        e=0.0;
                }
            }
        }


        center+=0.5/std::pow(2,nRef);
        const float_type a  = 10.;
        const float_type a2 = a*a;
        const float_type dx_base = domain_.dx_base();

        for (auto it  = domain_.begin_leafs();
                it != domain_.end_leafs(); ++it)
        {

            auto dx_level =  dx_base/std::pow(2,it->refinement_level());
            auto scaling =  std::pow(2,it->refinement_level());

           auto view(it->data()->node_field().domain_view());
           auto& nodes_domain=it->data()->nodes_domain();
           for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
           {
               it2->get<source>()  = 0.0;
               it2->get<phi_num>()= 0.0;
               const auto& coord=it2->level_coordinate();

               // manufactured solution:
               float_type x = static_cast<float_type>
                   (coord[0]-center[0]*scaling+0.5)*dx_level;
               float_type y = static_cast<float_type>
                   (coord[1]-center[1]*scaling+0.5)*dx_level;
               float_type z = static_cast<float_type>
                   (coord[2]-center[2]*scaling+0.5)*dx_level;
               const auto x2 = x*x;
               const auto y2 = y*y;
               const auto z2 = z*z;

               float_type s_tmp=
                   a*std::exp(-a*(x2)-a*(y2)-a*(z2))*(-6.0)+
                   (a2)*(x2)*std::exp(-a*(x2)-a*(y2)-a*(z2))*4.0 +
                   (a2)*(y2)*std::exp(-a*(x2)-a*(y2)-a*(z2))*4.0+
                   (a2)*(z2)*std::exp(-a*(x2)-a*(y2)-a*(z2))*4.0;

               it2->get<source>()=s_tmp;
               it2->get<phi_exact>() = std::exp((-a*x2 - a*y2 - a*z2));
           }
        }
    }


    /** @brief Run poisson test case, compute errors and write out.  */
    void run()
    {
      
        solver::PoissonSolver<simulation_type> psolver(&simulation_);
        psolver.solve<source, phi_num>();
        compute_errors();
        simulation_.write("solution.vtk");
        pcout << "Writing solution " << std::endl;

    }


    /** @brief Compute L2 and LInf errors */
    void compute_errors()
    {
        auto L2   = 0.; auto LInf = -1.0; int count=0;
        for (auto it_t  = domain_.begin_leafs();
             it_t != domain_.end_leafs(); ++it_t)
        {

            auto& nodes_domain=it_t->data()->nodes_domain();
            for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
            {
                const float_type error_tmp = std::fabs(
                        it2->get<phi_num>() - it2->get<phi_exact>());

                it2->get<error>() = error_tmp;
                L2 += error_tmp*error_tmp;

                if ( error_tmp > LInf)
                    LInf = error_tmp;

                ++count;
            }
        }
        pcout << "L2   = " << std::sqrt(L2/count)<< std::endl;
        pcout << "LInf = " << LInf << std::endl;
    }

private:

    Simulation<domain_t>              simulation_;
    domain_t&                         domain_;

    parallel_ostream::ParallelOstream pcout;
    lgf::LGF<lgf::Lookup>             lgf_;
};


#endif // IBLGF_INCLUDED_POISSON_HPP
