#ifndef IBLGF_INCLUDED_VORTEXRING_HPP
#define IBLGF_INCLUDED_VORTEXRING_HPP

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
         (error_lap_source , float_type, 1,       1),
         (decomposition    , float_type, 1,       1)
    ))
};


struct VortexRingTest:public SetupBase<VortexRingTest,parameters>
{

    using super_type =SetupBase<VortexRingTest,parameters>;


    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

    VortexRingTest(Dictionary* _d)
    :super_type(_d, 
            [this](auto _d, auto _domain){
                return this->initialize_domain(_d, _domain); })
    {

        if(domain_->is_client())client_comm_=client_comm_.split(1);
        else client_comm_=client_comm_.split(0);

        R_ = simulation_.dictionary_->
            template get_or<float_type>("R",1);

        rmin_ref_ = simulation_.dictionary_->
            template get_or<float_type>("Rmin_ref",R_);

        rmax_ref_ = simulation_.dictionary_->
            template get_or<float_type>("Rmax_ref",R_);

        rz_ref_ = simulation_.dictionary_->
            template get_or<float_type>("Rz_ref",R_);

        c1 = simulation_.dictionary_->
            template get_or<float_type>("c1",1);
        c2 = simulation_.dictionary_->
            template get_or<float_type>("c2",1);

        nLevels_=simulation_.dictionary_->
            template get_or<int>("nLevels",0);

        pcout << "\n Setup:  Test - Vortex ring \n" << std::endl;
        pcout << "Number of refinement levels: "<<nLevels_<<std::endl;
        pcout << "Simulation: \n" << simulation_ << std::endl;
        domain_->register_refinement_codtion()=
            [this](auto octant){return this->refinement(octant);};
        domain_->init_refine(_d->get_dictionary("simulation_parameters") 
                ->template get_or<int>("nLevels",0));
        domain_->distribute();
        this->initialize();
    }


    void run()
    {
        boost::mpi::communicator world;
        if(domain_->is_client())
        //if(false)
        {
            poisson_solver_t psolver(&this->simulation_);

            mDuration_type solve_duration(0);
            TIME_CODE( solve_duration, SINGLE_ARG(
                    psolver.solve<source, phi_num>();
            ))
            pcout_c<<"Total Psolve time: " 
                  <<solve_duration.count()<<" on "<<world.size()<<std::endl;
        }
        this->compute_errors();
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

        const float_type alpha = simulation_.dictionary_->
            template get_or<float_type>("alpha",1);
        const float_type a = simulation_.dictionary_->
            template get_or<float_type>("a",1);



        R_ = simulation_.dictionary_->
            template get_or<float_type>("R",1);
        float_type R=R_;

        // Adapt center to always have peak value in a cell-center
        //center+=0.5/std::pow(2,nRef);
        const float_type dx_base = domain_->dx_base();

        // Loop through leaves and assign values
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
               //const auto& coord=it2->global_coordinate();

               const auto& coord=it2->level_coordinate();


               // manufactured solution:
               float_type x = static_cast<float_type>
                   (coord[0]-center[0]*scaling+0.5)*dx_level;
               float_type y = static_cast<float_type>
                   (coord[1]-center[1]*scaling+0.5)*dx_level;
               float_type z = static_cast<float_type>
                   (coord[2]-center[2]*scaling+0.5)*dx_level;

               const float_type r=std::sqrt(x*x+y*y+z*z) ;

               /***********************************************************/
               // Sharp interface sphere
               const float_type alphas=alpha;
               float_type source_tmp= 1.0/(R*R)*(a*a)*(alphas*alphas)*
                                      std::exp(-a*std::pow(r/R,alphas))*
                                      std::pow(r/R,alphas*2.0-2.0)-1.0/(R*R)*
                                      a*alphas*std::exp(-a*std::pow(r/R,alphas))*
                                      std::pow(r/R,alphas-2.0)*(alphas-1.0);

               float_type exact = std::exp(-a*std::pow(r/R,alpha)) ;
               it2->get<source>()=source_tmp;
               it2->get<phi_exact>() = exact;
               it2->get<decomposition>()=world.rank();
               /***********************************************************/
               // Vortex Ring

               float_type t=std::sqrt( (r-R)*(r-R) +z*z )/R;
               float_type t0=0.0;
               float_type t1=0.0;
               if(std::fabs(t)<1.0)
               {

                    t0 = -1.0/(R*R)*c1*c2*std::exp(c2/(1.0/(R*R)*
                    (std::pow(R-r,2.0)+z*z)-1.0))
                    *1.0/std::pow(1.0/(R*R)*(std::pow(R-r,2.0)+z*z)-1.0,2.0)*
                    ((R-r)*1.0/r*-2.0+((z*z)*2.0)/(x*x+y*y+z*z)+(z*z)*(R-r)*
                    1.0/std::pow(x*x+y*y+z*z,3.0/2.0)*2.0+2.0)+1.0/(R*R*R*R)*
                    c1*c2*std::exp(c2/(1.0/(R*R)*(std::pow(R-r,2.0)+z*z)-1.0))*
                    1.0/std::pow(1.0/(R*R)*(std::pow(R-r,2.0)+z*z)-1.0,3.0)*
                    std::pow(z*2.0-z*(R-r)*1.0/r*2.0,2.0)*2.0+1.0/(R*R*R*R)*
                    c1*(c2*c2)*std::exp(c2/(1.0/(R*R)*(std::pow(R-r,2.0)+z*z)-1.0))*
                    1.0/std::pow(1.0/(R*R)*(std::pow(R-r,2.0)+z*z)-1.0,4.0)*
                    std::pow(z*2.0-z*(R-r)*1.0/r*2.0,2.0)+1.0/(R*R)*c1*c2*
                    std::exp(c2/(1.0/(R*R)*(std::pow(R-r,2.0)+z*z)-1.0))*
                    (R-r)*1.0/std::pow(1.0/(R*R)*(std::pow(R-r,2.0)+z*z)-1.0,2.0)*
                    1.0/r*4.0-(1.0/(R*R)*c1*c2*(x*x)*std::exp(c2/(1.0/(R*R)*
                    (std::pow(R-r,2.0)+z*z)-1.0))*1.0/std::pow(1.0/(R*R)*
                    (std::pow(R-r,2.0)+z*z)-1.0,2.0)*2.0)/(x*x+y*y+z*z)-(1.0/(R*R)*
                    c1*c2*(y*y)*std::exp(c2/(1.0/(R*R)*(std::pow(R-r,2.0)+z*z)-1.0))*
                    1.0/std::pow(1.0/(R*R)*(std::pow(R-r,2.0)+z*z)-1.0,2.0)*2.0)/
                    (x*x+y*y+z*z)+(1.0/(R*R*R*R)*c1*c2*(x*x)*std::exp(c2/(1.0/(R*R)*
                    (std::pow(R-r,2.0)+z*z)-1.0))*std::pow(R-r,2.0)*
                    1.0/std::pow(1.0/(R*R)* (std::pow(R-r,2.0)+z*z)-1.0,3.0)*8.0)/
                    (x*x+y*y+z*z)+(1.0/(R*R*R*R)*c1*c2*(y*y)* 
                    std::exp(c2/(1.0/(R*R)*(std::pow(R-r,2.0)+z*z)-1.0))*
                    std::pow(R-r,2.0)* 1.0/std::pow(1.0/(R*R)*
                    (std::pow(R-r,2.0)+z*z)-1.0,3.0)*8.0)/(x*x+y*y+z*z)+
                    (1.0/(R*R*R*R)*c1*(c2*c2)*(x*x)*std::exp(c2/(1.0/(R*R)*
                    (std::pow(R-r,2.0)+z*z)-1.0))*std::pow(R-r,2.0)*
                    1.0/std::pow(1.0/(R*R)*
                    (std::pow(R-r,2.0)+z*z)-1.0,4.0)*4.0)/(x*x+y*y+z*z)+
                    (1.0/(R*R*R*R)*c1*(c2*c2)*(y*y)*std::exp(c2/(1.0/(R*R)*
                    (std::pow(R-r,2.0)+z*z)-1.0))*std::pow(R-r,2.0)*
                    1.0/std::pow(1.0/(R*R)*(std::pow(R-r,2.0)+z*z)-1.0,4.0)*4.0)/
                    (x*x+y*y+z*z)-1.0/(R*R)*c1*c2*(x*x)*std::exp(c2/(1.0/(R*R)*
                    (std::pow(R-r,2.0)+z*z)-1.0))*(R-r)*1.0/std::pow(1.0/(R*R)*
                    (std::pow(R-r,2.0)+z*z)-1.0,2.0)*1.0/std::pow(x*x+y*y+z*z,3.0/2.0)*
                    2.0-1.0/(R*R)*c1*c2*(y*y)*std::exp(c2/(1.0/(R*R)*
                    (std::pow(R-r,2.0)+z*z)-1.0))*(R-r)*1.0/std::pow(1.0/
                    (R*R)*(std::pow(R-r,2.0)+z*z)-1.0,2.0)*1.0/
                    std::pow(x*x+y*y+z*z,3.0/2.0)*2.0;

                   t1 = c1* std::exp(- c2/ (1-t*t) );
               }
               it2->get<source>()=t0;
               it2->get<phi_exact>() = t1;
               /***********************************************************/
           }
        }
    }

    /** @brief Compute L2 and LInf errors */
    void compute_errors()
    {
        const float_type dx_base=domain_->dx_base();
        auto L2   = 0.; auto LInf = -1.0; int count=0;
        std::vector<float_type> L2_perLevel(nLevels_+1,0.0);
        std::vector<float_type> LInf_perLevel(nLevels_+1,0.0);

        if(domain_->is_server())  return;

        for (auto it_t  = domain_->begin_leafs();
                it_t != domain_->end_leafs(); ++it_t)
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

                L2_perLevel[refinement_level]+=(error_tmp*error_tmp * (dx*dx*dx));

                if ( std::fabs(error_tmp) > LInf)
                    LInf = std::fabs(error_tmp);

                if ( std::fabs(error_tmp) > LInf_perLevel[refinement_level] )
                    LInf_perLevel[refinement_level]=std::fabs(error_tmp);

                ++count;
            }
        }

        float_type L2_global(0.0);
        float_type LInf_global(0.0);

        boost::mpi::all_reduce(client_comm_,L2, L2_global, std::plus<float_type>());
        boost::mpi::all_reduce(client_comm_,LInf, LInf_global,[&](const auto& v0,
                               const auto& v1){return v0>v1? v0  :v1;} );

        //Level wise errros
        std::vector<float_type> L2_perLevel_global(nLevels_+1,0.0);
        std::vector<float_type> LInf_perLevel_global(nLevels_+1,0.0);
        for(std::size_t i=0;i<L2_perLevel.size();++i)
        {

            boost::mpi::all_reduce(client_comm_,L2_perLevel[i], 
                                   L2_perLevel_global[i], std::plus<float_type>());
            boost::mpi::all_reduce(client_comm_,LInf_perLevel[i], 
                                   LInf_perLevel[i],[&](const auto& v0,
                                       const auto& v1){return v0>v1? v0  :v1;});
            std::cout<<"L2_"<<i<<" "<<L2_perLevel_global[2]<<std::endl;
            std::cout<<"LInf_"<<i<<" "<<LInf_perLevel_global[2]<<std::endl;
        }

        pcout_c << "L2 = " << std::sqrt(L2_global)<< std::endl;
        pcout_c << "LInf_global = " << LInf_global << std::endl;

    }


    /** @brief  Refienment conditon for octants.  */
    template<class OctantType>
    bool refinement(OctantType* it) const noexcept
    {
        auto b=it->data()->descriptor();
        b.level()=it->refinement_level();
        const float_type dx_base = domain_->dx_base();
        return refinement(b,R_, rmin_ref_,rmax_ref_,rz_ref_,dx_base);
    }

    /** @brief  Refienment conditon for blocks.  */
    bool refinement(block_descriptor_t b, 
                    float_type _R, 
                    float_type _rmin_ref, 
                    float_type  _rmax_ref, 
                    float_type _rz_ref,
                    float_type dx_base ) const noexcept
    {

        auto center = (domain_->bounding_box().max() -
                       domain_->bounding_box().min()) / 2.0 +
                       domain_->bounding_box().min();

        auto scaling =  std::pow(2,b.level());
        center*=scaling;

        b.grow(2,2);
        auto corners= b.get_corners();


        float_type rscale=_R*scaling/dx_base;

        auto lower_t = (b.base()-center); 
        auto upper_t= (b.max()+1 -center);

        bool outside=false;
        bool inside=false;
        float_type rcmin=std::numeric_limits<float_type>::max();
        float_type rcmax=std::numeric_limits<float_type>::lowest();
        float_type zcmin=std::numeric_limits<float_type>::max();
        for(auto& c : corners)
        {
            float_type r_c =  std::sqrt( c.x()*c.x() + c.y()*c.y()  );
            if(r_c < rcmin) rcmin=r_c;
            if(r_c > rcmax) rcmax=r_c;

            float_type cz=std::fabs(static_cast<float_type>(c.z()));
            if(cz<zcmin) zcmin=cz;

            if( r_c<=rscale ) inside=true;
            if( r_c>=rscale ) outside=true;
        }
        float_type rz =_rz_ref*rscale;
        bool z_cond = (zcmin<=rz) || (lower_t.z() <=0 && upper_t.z()>=0);
        if( z_cond  && outside &&inside)
        {
            return true;
        }

        float_type rmin =_rmin_ref*rscale;
        float_type rmax =_rmax_ref*rscale;
        float_type rcmid=0.5*(rcmax+rcmin);
        if( z_cond && (rcmid>=rmin && rcmid<=rmax) )  
        {
            return true;
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

        float_type rmin = _d-> template get<float_type>("Rmin");
        float_type rmax = _d-> template get<float_type>("Rmax");
        float_type rz   = _d-> template get<float_type>("Rz");
        float_type R0   = _d-> template get<float_type>("R");

        const float_type dx_base = _domain->dx_base();

        auto it=res.begin();
        while(it!=res.end())
        {
            block_descriptor_t b(*it*_domain->block_extent(), _domain->block_extent());
            if(refinement(b,R0,rmin,rmax,rz,dx_base))
                ++it;
            else
                it=res.erase(it);
        }
        return res;
    }


    private:
    boost::mpi::communicator client_comm_;
    float_type R_;

    float_type rmin_ref_;
    float_type rmax_ref_;
    float_type rz_ref_;
    float_type c1=0;
    float_type c2=0;
    int nLevels_=0;



};


#endif // IBLGF_INCLUDED_POISSON_HPP
