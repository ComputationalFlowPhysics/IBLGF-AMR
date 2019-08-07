#ifndef IBLGF_INCLUDED_IFHERK_HEAT_HPP
#define IBLGF_INCLUDED_IFHERK_HEAT_HPP


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
#include<solver/time_integration/ifherk.hpp>

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
         (error            , float_type, 1,    1,       1,     cell),
         (amr_lap_source   , float_type, 1,    1,       1,     cell),
         (amr_div_source   , float_type, 1,    1,       1,     cell),
         (error_lap_source , float_type, 1,    1,       1,     cell),
         (decomposition    , float_type, 1,    1,       1,     cell),
        //IF-HERK
         (u_0_exact        , float_type, 1,    1,       1,     cell),
         (u                , float_type, 3,    1,       1,     face),
         (w                , float_type, 3,    1,       1,     face),
         (p                , float_type, 1,    1,       1,     cell)
    ))
};


struct IfherkHeat:public SetupBase<IfherkHeat,parameters>
{

    using super_type =SetupBase<IfherkHeat,parameters>;


    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

    IfherkHeat(Dictionary* _d)
    :super_type(_d,
            [this](auto _d, auto _domain){
                return this->initialize_domain(_d, _domain); })
    {

        if(domain_->is_client())client_comm_=client_comm_.split(1);
        else client_comm_=client_comm_.split(0);

        dx_  = domain_->dx_base();
        cfl_ = simulation_.dictionary()->template get_or<float_type>("cfl",0.5);
        dt_  = simulation_.dictionary()->template get_or<float_type>("dt",-1.0);
        tot_steps_ = simulation_.dictionary()->template get<int>("nTimeSteps");
        Re_        = simulation_.dictionary()->template get<float_type>("Re");

        if (dt_<0)
            dt_=dx_*cfl_;

        source_max_=simulation_.dictionary_->
            template get_or<float_type>("source_max",1.0);

        nLevelRefinement_=simulation_.dictionary_->
            template get_or<int>("nLevels",0);

        pcout << "\n Setup:  Test - Simple IC \n" << std::endl;
        pcout << "Number of refinement levels: "<<nLevelRefinement_<<std::endl;

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
        //simulation_.write2("ifherk_begin.hdf5");

        simulation_.write2("ifherk_0.hdf5");
        if(domain_->is_client())
        {
            time_integration_t ifherk(&this->simulation_);

            mDuration_type ifherk_duration(0);
            TIME_CODE( ifherk_duration, SINGLE_ARG(
                ifherk.time_march();
            ))
            pcout_c<<"Time to solution [ms] "<<ifherk_duration.count()<<std::endl;

            for (auto it  = domain_->begin_leafs();
                    it != domain_->end_leafs(); ++it)
                if (it->locally_owned())
                {
                    it->data()->template get_linalg<phi_num>().get()->
                    cube_noalias_view() =
                    it->data()->template get_linalg_data<u>(0);
                }
            this->compute_errors<phi_num,u_0_exact,error>();
        }
        simulation_.write2("ifherk_1.hdf5");
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

           float_type T = dt_*tot_steps_;
           for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
           {
               it2->get<source>() = 0.0;
               it2->get<phi_num>()= 0.0;

               const auto& coord=it2->level_coordinate();

               float_type x = static_cast<float_type>
                   (coord[0]-center[0]*scaling+0.5)*dx_level;
               float_type y = static_cast<float_type>
                   (coord[1]-center[1]*scaling+0.5)*dx_level;
               float_type z = static_cast<float_type>
                   (coord[2]-center[2]*scaling+0.5)*dx_level;

               const float_type r=std::sqrt(x*x+y*y+z*z) ;
               /***********************************************************/

               float_type r_2 = r*r;
               it2->template get<u>(0)=std::exp(-a_*r_2);
               it2->template get<u>(1)=10.0;
               it2->template get<u>(2)=0;

               it2->template get<u_0_exact>() =
               std::exp(-(a_*Re_*r_2)/(Re_ + 4*a_*T))/
                (std::pow((1 + a_*4*T/Re_),1.5));

               it2->template get<decomposition>()=world.rank();
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

        std::vector<float_type> L2_perLevel(nLevelRefinement_+1+global_refinement_,0.0);
        std::vector<float_type> L2_exact_perLevel(nLevelRefinement_+1+global_refinement_,0.0);
        std::vector<float_type> LInf_perLevel(nLevelRefinement_+1+global_refinement_,0.0);
        std::vector<float_type> LInf_exact_perLevel(nLevelRefinement_+1+global_refinement_,0.0);

        std::vector<int> counts(nLevelRefinement_+1+global_refinement_,0);

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

                //if(std::isnan(tmp_num))
                //    std::cout<<"this is nan at level = " << it_t->level()<<std::endl;

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
        std::vector<float_type> L2_perLevel_global(nLevelRefinement_+1+global_refinement_,0.0);
        std::vector<float_type> LInf_perLevel_global(nLevelRefinement_+1+global_refinement_,0.0);

        std::vector<float_type> L2_exact_perLevel_global(nLevelRefinement_+1+global_refinement_,0.0);
        std::vector<float_type> LInf_exact_perLevel_global(nLevelRefinement_+1+global_refinement_,0.0);

        std::vector<int> counts_global(nLevelRefinement_+1+global_refinement_,0);
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
        auto b=it->data()->descriptor();
        b.level()=it->refinement_level();
        const float_type dx_base = domain_->dx_base();

        auto center = (domain_->bounding_box().max() -
                       domain_->bounding_box().min()) / 2.0 +
                       domain_->bounding_box().min();

        auto scaling =  std::pow(2,b.level());
        center*=scaling;
        auto dx_level =  dx_base/std::pow(2,b.level());

        b.grow(2,2);
        auto corners= b.get_corners();



        //auto lower_t = (b.base()-center);
        //auto upper_t= (b.max()+1 -center);

        for(int i=b.base()[0];i<=b.max()[0];++i)
        {
            for(int j=b.base()[1];j<=b.max()[1];++j)
            {
                for(int k=b.base()[2];k<=b.max()[2];++k)
                {

                    const float_type x = static_cast<float_type>
                        (i-center[0]+0.5)*dx_level;
                    const float_type y = static_cast<float_type>
                        (j-center[1]+0.5)*dx_level;
                    const float_type z = static_cast<float_type>
                        (k-center[2]+0.5)*dx_level;

                    const float_type r=std::sqrt(x*x+y*y+z*z) ;

                    float_type r_2 = r*r;
                    float_type u=std::exp(-a_*r_2);

                    if(u > 1.0*pow(0.25, diff_level))
                    {
                        return true;
                    }
                }
            }
        }

        return false;

    }



    /** @brief  Initialization of the domain blocks. This is registered in the
     *          domain through the base setup class, passing it to the domain ctor.
     */
    std::vector<extent_t> initialize_domain( Dictionary* _d, domain_t* _domain )
    {
        auto res=_domain-> construct_basemesh_blocks(_d, _domain->block_extent());
        domain_->read_parameters(_d);

        return res;
    }


    private:
    boost::mpi::communicator client_comm_;
    float_type R_;
    float_type source_max_;

    float_type rmin_ref_;
    float_type rmax_ref_;
    float_type rz_ref_;
    float_type c1=0;
    float_type c2=0;
    float_type eps_grad_=1.0e6;;
    int nLevelRefinement_=0;
    int global_refinement_=0;
    fcoord_t offset_;

    float_type a_ = 10.0;

    float_type dt_,dx_;
    float_type cfl_;
    float_type Re_;
    int tot_steps_;
};


#endif
