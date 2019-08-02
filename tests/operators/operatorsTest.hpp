#ifndef IBLGF_INCLUDED_OPERATORTEST_HPP
#define IBLGF_INCLUDED_OPERATORTEST_HPP

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <vector>
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
#include<operators/operators.hpp>



const int Dim = 3;

struct parameters
{
    static constexpr std::size_t Dim= 3;
    REGISTER_FIELDS
    (
    Dim,
     (
        //name               type        Dim   lBuffer  hBuffer, storage type
         (phi_num          , float_type, 1,    1,       1,     cell),
         (source           , float_type, 1,    1,       1,     cell),
         (phi_exact        , float_type, 1,    1,       1,     cell),
         (error            , float_type, 1,    1,       1,     cell),
         (amr_lap_source   , float_type, 1,    1,       1,     cell),
         (amr_div_source   , float_type, 1,    1,       1,     cell),
         (error_lap_source , float_type, 1,    1,       1,     cell)
    ))
};


struct OperatorTest:public SetupBase<OperatorTest,parameters>
{

    using super_type =SetupBase<OperatorTest,parameters>;


    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

    OperatorTest(Dictionary* _d)
    :super_type(_d,
            [this](auto _d, auto _domain){
                return this->initialize_domain(_d, _domain); })
    {

        if(domain_->is_client())client_comm_=client_comm_.split(1);
        else client_comm_=client_comm_.split(0);


        global_refinement_=simulation_.dictionary_->
            template get_or<int>("global_refinement",0);

        pcout << "\n Setup:  Test - Vortex ring \n" << std::endl;
        pcout << "Number of refinement levels: "<<nLevels_<<std::endl;

        domain_->register_refinement_condition()=
            [this](auto octant, int diff_level){return this->refinement(octant, diff_level);};
        domain_->init_refine(_d->get_dictionary("simulation_parameters")
                ->template get_or<int>("nLevels",0), global_refinement_);
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        this->initialize();


        boost::mpi::communicator world;
        if(world.rank()==0)
            std::cout << "on Simulation: \n" << simulation_ << std::endl;
    }


    void run()
    {
        boost::mpi::communicator world;
        if(domain_->is_client())
        {
            
            const float_type dx_base = domain_->dx_base();
            //std::cout<<"BLA"<<std::endl; 

            //mDuration_type solve_duration(0);
            //TIME_CODE( solve_duration, SINGLE_ARG(
            //        psolver.solve<source, phi_num>();
            //))
            //pcout_c<<"Total Psolve time: "
            //      <<solve_duration.count()<<" on "<<world.size()<<std::endl;

            //Bufffer exchange of some fields 
            auto client=domain_->decomposition().client();
            client->buffer_exchange<phi_num>();
            client->buffer_exchange<face_aux>();

            mDuration_type lap_duration(0);
            TIME_CODE( lap_duration, SINGLE_ARG(
            for (auto it  = domain_->begin_leafs();
                      it != domain_->end_leafs(); ++it)
            {
                if(!it->locally_owned() || !it->data())continue;

                auto dx_level =  dx_base/std::pow(2,it->refinement_level());
                //domain::Operator::laplace<phi_num, amr_lap_source>( *(it->data()),dx_level);
                //domain::Operator::divergence<face_aux, amr_div_source>( *(it->data()),dx_level);
            }
            ))
            pcout_c<<"Total Laplace time: "
                  <<lap_duration.count()<<" on "<<world.size()<<std::endl;
        }
        this->compute_errors<phi_num,phi_exact,error>();
        this->compute_errors<amr_lap_source,source,error_lap_source>("Lap");

        simulation_.write2("mesh.hdf5");
    }


    /** @brief Initialization of poisson problem.
     *  @detail Testing poisson with manufactured solutions: exp
     */
    void initialize()
    {
        boost::mpi::communicator world;
        if(domain_->is_server()) return ;
        auto center = (domain_->bounding_box().max() -
                       domain_->bounding_box().min()) / 2.0 +
                       domain_->bounding_box().min();

        // Adapt center to always have peak value in a cell-center
        //center+=0.5/std::pow(2,nRef);
        const float_type dx_base = domain_->dx_base();

        for (auto it  = domain_->begin_leafs();
                  it != domain_->end_leafs(); ++it)
        {
            if(!it->locally_owned()) continue;
            if (!(*it && it->data())) continue;
            auto dx_level =  dx_base/std::pow(2,it->refinement_level());
            auto scaling =  std::pow(2,it->refinement_level());

           auto view(it->data()->node_field().domain_view());
           auto& nodes_domain=it->data()->nodes_domain();
           for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
           {
               const auto& coord=it2->level_coordinate();
               std::cout<<"coord  "<<coord<<std::endl;
               it2->get<source>() = coord[0];
               it2->get<face_aux>(0)= coord[0];
               it2->get<face_aux>(1)= coord[1];
               it2->get<face_aux>(2)= coord[2];
           }
        }
    }



    /** @brief Compute L2 and LInf errors */
    template<class Numeric, class Exact, class Error>
    void compute_errors(std::string _output_prefix="")
    {
        const float_type dx_base=domain_->dx_base();
        float_type L2   = 0.; float_type LInf = -1.0; int count=0;
        float_type L2_exact = 0; float_type LInf_exact = -1.0;

        std::vector<float_type> L2_perLevel(nLevels_+1+global_refinement_,0.0);
        std::vector<float_type> L2_exact_perLevel(nLevels_+1+global_refinement_,0.0);
        std::vector<float_type> LInf_perLevel(nLevels_+1+global_refinement_,0.0);
        std::vector<float_type> LInf_exact_perLevel(nLevels_+1+global_refinement_,0.0);

        std::vector<int> counts(nLevels_+1+global_refinement_,0);

        if(domain_->is_server())  return;

        for (auto it_t  = domain_->begin_leafs();
                it_t != domain_->end_leafs(); ++it_t)
        {
            if(!it_t->locally_owned() || !it_t->data())continue;

            int refinement_level = it_t->refinement_level();
            double dx = dx_base/std::pow(2.0,refinement_level);

            auto& nodes_domain=it_t->data()->nodes_domain();
            for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
            {
                float_type tmp_exact = it2->template get<Exact>();
                float_type tmp_num   = it2->template get<Numeric>();

                float_type error_tmp = tmp_num - tmp_exact;

                it2->template get<Error>() = error_tmp;

                L2 += error_tmp*error_tmp * (dx*dx*dx);
                L2_exact += tmp_exact*tmp_exact*(dx*dx*dx);

                L2_perLevel[refinement_level]+=error_tmp*error_tmp* (dx*dx*dx);
                L2_exact_perLevel[refinement_level]+=tmp_exact*tmp_exact*(dx*dx*dx);
                ++counts[refinement_level];

                if ( std::fabs(tmp_exact) > LInf_exact)
                    LInf_exact = std::fabs(tmp_exact);

                if ( std::fabs(error_tmp) > LInf)
                    LInf = std::fabs(error_tmp);

                if ( std::fabs(error_tmp) > LInf_perLevel[refinement_level] )
                    LInf_perLevel[refinement_level]=std::fabs(error_tmp);

                if ( std::fabs(tmp_exact) > LInf_exact_perLevel[refinement_level] )
                    LInf_exact_perLevel[refinement_level]=std::fabs(tmp_exact);

                ++count;
            }
        }

        float_type L2_global(0.0);
        float_type LInf_global(0.0);

        float_type L2_exact_global(0.0);
        float_type LInf_exact_global(0.0);

        boost::mpi::all_reduce(client_comm_,L2, L2_global, std::plus<float_type>());
        boost::mpi::all_reduce(client_comm_,L2_exact, L2_exact_global, std::plus<float_type>());

        boost::mpi::all_reduce(client_comm_,LInf, LInf_global,[&](const auto& v0,
                               const auto& v1){return v0>v1? v0  :v1;} );
        boost::mpi::all_reduce(client_comm_,LInf_exact, LInf_exact_global,[&](const auto& v0,
                               const auto& v1){return v0>v1? v0  :v1;} );

        pcout_c << "Glabal "<<_output_prefix<<"L2_exact = " << std::sqrt(L2_exact_global)<< std::endl;
        pcout_c << "Global "<<_output_prefix<<"LInf_exact = " << LInf_exact_global << std::endl;

        pcout_c << "Glabal "<<_output_prefix<<"L2 = " << std::sqrt(L2_global)<< std::endl;
        pcout_c << "Global "<<_output_prefix<<"LInf = " << LInf_global << std::endl;

        //Level wise errros
        std::vector<float_type> L2_perLevel_global(nLevels_+1+global_refinement_,0.0);
        std::vector<float_type> LInf_perLevel_global(nLevels_+1+global_refinement_,0.0);

        std::vector<float_type> L2_exact_perLevel_global(nLevels_+1+global_refinement_,0.0);
        std::vector<float_type> LInf_exact_perLevel_global(nLevels_+1+global_refinement_,0.0);

        std::vector<int> counts_global(nLevels_+1+global_refinement_,0);
        for(std::size_t i=0;i<LInf_perLevel_global.size();++i)
        {
            boost::mpi::all_reduce(client_comm_,counts[i],
                                   counts_global[i], std::plus<float_type>());
            boost::mpi::all_reduce(client_comm_,L2_perLevel[i],
                                   L2_perLevel_global[i], std::plus<float_type>());
            boost::mpi::all_reduce(client_comm_,LInf_perLevel[i],
                                   LInf_perLevel_global[i],[&](const auto& v0,
                                       const auto& v1){return v0>v1? v0  :v1;});

            boost::mpi::all_reduce(client_comm_,L2_exact_perLevel[i],
                                   L2_exact_perLevel_global[i], std::plus<float_type>());
            boost::mpi::all_reduce(client_comm_,LInf_exact_perLevel[i],
                                   LInf_exact_perLevel_global[i],[&](const auto& v0,
                                       const auto& v1){return v0>v1? v0  :v1;});

            pcout_c<<_output_prefix<<"L2_"<<i<<" "<<std::sqrt(L2_perLevel_global[i])<<std::endl;
            pcout_c<<_output_prefix<<"LInf_"<<i<<" "<<LInf_perLevel_global[i]<<std::endl;
            pcout_c<<"count_"<<i<<" "<<counts_global[i]<<std::endl;
        }


    }


    /** @brief  Refienment conditon for octants.  */
    template<class OctantType>
    bool refinement(OctantType* it, int diff_level,
            bool use_all=false) const noexcept
    {
        return false;
    }


    /** @brief  Initialization of the domain blocks. This is registered in the
     *          domain through the base setup class, passing it to the domain ctor.
     */
    std::vector<extent_t> initialize_domain( Dictionary* _d, domain_t* _domain )
    {
        return _domain-> construct_basemesh_blocks(_d, _domain->block_extent());
    }


    private:
    boost::mpi::communicator client_comm_;

    float_type eps_grad_=1.0e6;;
    int nLevels_=0;
    int global_refinement_;
    fcoord_t offset_;
};


#endif // IBLGF_INCLUDED_POISSON_HPP
