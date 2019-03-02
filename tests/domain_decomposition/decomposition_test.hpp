#ifndef IBLGF_INCLUDED_DECOMPOSITION_TEST_HPP
#define IBLGF_INCLUDED_DECOMPOSITION_TEST_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

// IBLGF-specific
#include <global.hpp>
#include <simulation.hpp>
#include <domain/domain.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <domain/octree/tree.hpp>
#include <chrono>
#include <IO/parallel_ostream.hpp>
#include <lgf/lgf.hpp>
#include <fmm/fmm.hpp>

#include<utilities/convolution.hpp>
#include<utilities/interpolation.hpp>
#include<solver/poisson/poisson.hpp>

#include"../../setups/setup_base.hpp"

//#define FMM_TIMING(  )


const int Dim = 3;

struct parameters
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

struct DecomposistionTest:public SetupBase<DecomposistionTest,parameters>
{

    using super_type =SetupBase<DecomposistionTest,parameters>;


    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

    DecomposistionTest(Dictionary* _d)
    :super_type(_d)
    {

        if(domain_.is_client())client_comm_=client_comm_.split(1);
        else client_comm_=client_comm_.split(0);

        pcout << "\n Setup:  Test - Domain decomposition \n" << std::endl;
        pcout << "Simulation: \n" << simulation_    << std::endl;
//        this->refine();
        domain_.distribute();
        this->initialize();
    }


    void run()
    {
        if(domain_.is_client())
        {
            poisson_solver_t psolver(&this->simulation_);

            mDuration_type solve_duration(0);
            TIME_CODE( solve_duration, SINGLE_ARG(
                    psolver.solve<source, phi_num>();
            ))
            pcout_c<<"Total Psolve time: " <<solve_duration.count()<<std::endl;
        }
        this->compute_errors();
        simulation_.write2("mesh.hdf5");
    }

//    void refine()
//    {
//        boost::mpi::communicator world;
//        std::cout<<"Refining on rank:"<<world.rank()<<std::endl;
//
//        auto center = (domain_.bounding_box().max() -
//                       domain_.bounding_box().min()) / 2.0 +
//                       domain_.bounding_box().min();
//
//        const int nRef = simulation_.dictionary_->
//            template get_or<int>("nLevels",0);
//
//        for(int l=0;l<nRef;++l)
//        {
//            for (auto it  = domain_.begin_leafs();
//                    it != domain_.end_leafs(); ++it)
//            {
//                auto b=it->data()->descriptor();
//
//                const auto lower((center )/2-2 ), upper((center )/2+2 - b.extent());
//                b.grow(lower, upper);
//                if(b.is_inside( center * pow(2.0,l))
//                        && it->refinement_level()==l )
//                {
//                    domain_.refine(it);
//                }
//            }
//        }
//
//    }

    /** @brief Initialization of poisson problem.
     *  @detail Testing poisson with manufactured solutions: exp
     */
    void initialize()
    {
        boost::mpi::communicator world;
        if(domain_.is_server()) return ;
        std::cout<<"Initializing on rank:"<<world.rank()<<std::endl;
        auto center = (domain_.bounding_box().max() -
                       domain_.bounding_box().min()) / 2.0 +
                       domain_.bounding_box().min();

        const int nRef = simulation_.dictionary_->
            template get_or<int>("nLevels",0);


        // Adapt center to always have peak value in a cell-center
        center+=0.5/std::pow(2,nRef);
        const float_type a  = 10.;
        const float_type a2 = a*a;
        const float_type dx_base = domain_.dx_base();

        // Loop through leaves and assign values
        for (auto it  = domain_.begin_leafs();
                  it != domain_.end_leafs(); ++it)
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

        //std::cout<<"Initialization on rank "<<world.rank()<<" done."<<std::endl;
        if(world.size()==2)
            simulation_.write("solution.vtk");
    }

    /** @brief Compute L2 and LInf errors */
    void compute_errors()
    {


        const float_type dx_base=domain_.dx_base();
        auto L2   = 0.; auto LInf = -1.0; int count=0;
        if(domain_.is_server())  return;
        for (auto it_t  = domain_.begin_leafs();
                it_t != domain_.end_leafs(); ++it_t)
        {
            if(!it_t->locally_owned())continue;

            int refinement_level = it_t->refinement_level();
            double dx = dx_base/std::pow(2,refinement_level);

            auto& nodes_domain=it_t->data()->nodes_domain();
            for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
            {

                const float_type error_tmp = (
                        it2->get<phi_num>() - it2->get<phi_exact>());

                it2->get<error>() = error_tmp;
                L2 += error_tmp*error_tmp * (dx*dx*dx);

                if ( std::fabs(error_tmp) > LInf)
                    LInf = std::fabs(error_tmp);

                ++count;
            }
        }

        float_type L2_global(0.0);
        float_type LInf_global(0.0);
        boost::mpi::all_reduce(client_comm_,L2, L2_global, std::plus<float_type>());
        boost::mpi::all_reduce(client_comm_,LInf, LInf_global,[&](const auto& v0,
        const auto& v1){return v0>v1? v0  :v1;} );
        pcout_c << "L2  = " << std::sqrt(L2_global)<< std::endl;
        pcout_c << "LInf_global = " << LInf_global << std::endl;




    }
    private:

        boost::mpi::communicator client_comm_;
};


#endif // IBLGF_INCLUDED_POISSON_HPP
