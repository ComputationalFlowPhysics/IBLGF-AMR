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

#include"../setup_base.hpp"




/**  @brief Parameters, for the PoissonProblem setup
 *          and aliases for datablock, domain and simulation.
 */
struct ParametersPoissonProblem
{
    static constexpr std::size_t Dim= 3;
    REGISTER_FIELDS
    (
    Dim,
     (
        //name               type     lBuffer.  hBuffer
         (phi_num          , float_type, 1,       1),
         (source           , float_type, 1,       1),
         (phi_exact        , float_type, 1,       1),
         (error            , float_type, 1,       1),
         (amr_lap_source   , float_type, 1,       1),
         (error_lap_source , float_type, 1,       1)
    ))
};


/**  @brief Test-setup to solve the poisson problem in 3d using manufactured
 *          solutions.
 */
struct PoissonProblem: public SetupBase<PoissonProblem,ParametersPoissonProblem>
{

public: //member types
    using super_type =SetupBase<PoissonProblem,ParametersPoissonProblem>;

public: //Ctor
    PoissonProblem(Dictionary* _d)
    :super_type(_d)
    {
        this->initialize();
    }


    /** @brief Initialization of poisson problem.
     *  @detail Testing poisson with manufactured solutions: exp
     */
    void initialize()
    {
        this->pcout<<"Initializing"<<std::endl;
        auto center = (domain_->bounding_box().max() -
                       domain_->bounding_box().min()) / 2.0 +
                       domain_->bounding_box().min();

        const int nRef = this->simulation_.dictionary_->
            template get_or<int>("nLevels",0);

        for(int l=0;l<nRef;++l)
        {
            for (auto it  = domain_->begin_leafs();
                    it != domain_->end_leafs(); ++it)
            {
                auto b=it->data()->descriptor();

                const auto lower((center )/2-2 ), upper((center )/2+2 - b.extent());
                b.grow(lower, upper);
                if(b.is_inside( center * pow(2.0,l))
                   && it->refinement_level()==l
                  )
                {
                    //domain_->refine(it);
                }
            }
        }

        for (int lt = domain_->tree()->base_level();
                 lt < domain_->tree()->depth(); ++lt)
        {
            for (auto it  = domain_->begin(lt);
                      it != domain_->end(lt); ++it)
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
        const float_type dx_base = domain_->dx_base();

        for (auto it  = domain_->begin_leafs();
                it != domain_->end_leafs(); ++it)
        {

            auto dx_level =  dx_base/std::pow(2,it->refinement_level());
            auto scaling =  std::pow(2,it->refinement_level());

           auto view(it->data()->node_field().domain_view());
           auto& nodes_domain=it->data()->nodes_domain();
           for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
           {
               it2->get<source>() = 0.0;
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

        poisson_solver_t psolver(&this->simulation_);
        psolver.solve<source, phi_num>();
        psolver.laplace_diff<phi_num,amr_lap_source>();

        //compute_errors();
        this->simulation_.write("solution.vtk");
        this->pcout << "Writing solution " << std::endl;

    }


    /** @brief Compute L2 and LInf errors */
    void compute_errors()
    {

        const float_type dx_base=domain_->dx_base();

        auto L2   = 0.; auto LInf = -1.0; int count=0;
        for (auto it_t  = domain_->begin_leafs();
             it_t != domain_->end_leafs(); ++it_t)
        {

            int refinement_level = it_t->refinement_level();
            double dx = dx_base/std::pow(2,refinement_level);

            auto& nodes_domain=it_t->data()->nodes_domain();
            for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
            {
                const float_type error_tmp = (
                        it2->get<phi_num>() - it2->get<phi_exact>());


                it2->get<error>() = error_tmp;
                L2 += error_tmp*error_tmp * (dx*dx*dx);

                if ( abs(error_tmp) > LInf)
                    LInf = abs(error_tmp);

                ++count;
            }
        }
        pcout << "L2   = " << std::sqrt(L2)<< std::endl;
        pcout << "LInf = " << LInf << std::endl;

        auto L2_source   = 0.; auto LInf_source = -1.0; count=0;
        for (auto it_t  = domain_->begin_leafs();
             it_t != domain_->end_leafs(); ++it_t)
        {

            int refinement_level = it_t->refinement_level();
            double dx = dx_base/std::pow(2,refinement_level);

            auto& nodes_domain=it_t->data()->nodes_domain();
            for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
            {
                const float_type error_tmp = (
                        it2->get<amr_lap_source>() - it2->get<source>());

                it2->get<error_lap_source>() = error_tmp;
                L2_source += error_tmp*error_tmp * (dx*dx*dx);

                if ( abs(error_tmp) > LInf_source)
                    LInf_source = abs(error_tmp);

                ++count;
            }
        }
        pcout << "L2_source   = " << std::sqrt(L2_source)<< std::endl;
        pcout << "LInf_source = " << LInf_source << std::endl;
    }

};


#endif // IBLGF_INCLUDED_POISSON_HPP
