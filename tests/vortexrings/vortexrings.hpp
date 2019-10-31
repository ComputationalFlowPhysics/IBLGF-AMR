#ifndef IBLGF_INCLUDED_VORTEXRINGS_HPP
#define IBLGF_INCLUDED_VORTEXRINGS_HPP

#define POISSON_TIMINGS

#include <iostream>
#include <chrono>

// IBLGF-specific
#include"../../setups/setup_base.hpp"


const int Dim = 3;

struct parameters
{
    static constexpr std::size_t Dim= 3;
    REGISTER_FIELDS
    (
    Dim,
     (
         //name              type       Dim   lBuffer.  hBuffer
         (phi_num          ,float_type, 1,    1,       1, cell),
         (source           ,float_type, 1,    1,       1, cell),
         (phi_exact        ,float_type, 1,    1,       1, cell),
         (error            ,float_type, 1,    1,       1, cell),
         (amr_lap_source   ,float_type, 1,    1,       1, cell),
         (error_lap_source ,float_type, 1,    1,       1, cell),
         (decomposition    ,float_type, 1,    1,       1, cell)
    ))
};


struct vortex_ring_wrong
{
    float_type vorticity(float_type x, float_type y, float_type z) const noexcept
    {
        x-=center[0];
        y-=center[1];
        z-=center[2];

        const float_type r=std::sqrt(x*x+y*y+z*z) ;
        const float_type t=std::sqrt( (r-R)*(r-R) +z*z )/R;

        if(std::fabs(t)>=1.0) return 0.0;

        const float_type t3 = z*z;
        const float_type t5 = x*x;
        const float_type t6 = y*y;
        const float_type t7 = t3+t5+t6;
        const float_type t18 = std::sqrt(t7);
        const float_type t2 = R-t18;
        const float_type t4 = R*R;
        const float_type t8 = std::pow(t7,9.0/2.0);
        const float_type t9 = t3*t3;
        const float_type t10 = t9*t9;
        const float_type t11 = t7*t7;
        const float_type t12 = t11*t11;
        const float_type t13 = std::pow(t7,5.0/2.0);
        const float_type t14 = std::pow(t7,3.0/2.0);
        const float_type t15 = t5*t5;
        const float_type t16 = t6*t6;
        const float_type t17 = std::pow(t7,7.0/2.0);
        const float_type t19 = c2*t4;
        const float_type t20 = t4*4.0;
        const float_type t21 = t19+t20;
        const float_type t22 = t4*1.2E1;
        const float_type t23 = t19+t22;
        const float_type t24 = t4*t4;

        float_type res= (c1*c2*t4*std::exp(c2/(1.0/(R*R)*(t3+t2*t2)-1.0))*(t17*-2.0+t4*t13*
        8.0-t9*t14*8.0+t14*t15*2.0+t14*t16*2.0-t3*t14*(t20+c2*t4*4.0)+R*t3*t9*
        1.3E1+R*t5*t9*2.3E1+R*t6*t9*2.3E1+R*t3*t15*1.0E1-R*t7*t11+R*t3*t16*
        1.0E1-t4*t5*t14*2.0-t4*t6*t14*2.0+t5*t6*t14*4.0+t3*t9*t18*2.0-t4*
        t9*t18*8.0+t5*t9*t18*2.0+t6*t9*t18*2.0-t15*t18*t21-t16*t18*t21+R*t3*
        t5*t6*2.0E1-c2*t3*t18*t24-c2*t5*t18*t24-c2*t6*t18*t24-t5*t6*t18*
        (t4*8.0+c2*t4*2.0)-t3*t5*t18*t23-t3*t6*t18*t23+R*c2*t4*t9*4.0+R*c2*
        t4*t15*2.0+R*c2*t4*t16*2.0+R*c2*t3*t4*t5*6.0+R*c2*t3*t4*t6*6.0+R*c2*
        t4*t5*t6*4.0)*-4.0)/(t3*t8*5.0+t5*t8+t6*t8+t10*t14*1.1E1+t17*t24*
        1.6E1-R*t3*t10*8.0-R*t5*t10*1.6E1-R*t6*t10*1.6E1-R*t7*t12*8.0+t4*t9*
        t13*9.6E1+t4*t13*t15*2.4E1+t4*t13*t16*2.4E1+t9*t14*t15*6.0+t9*t14*
        t16*6.0-R*t3*t9*t15*8.0-R*t7*t9*t11*4.8E1-R*t3*t9*t16*8.0+t3*t4*t5*
        t13*9.6E1+t3*t4*t6*t13*9.6E1+t4*t5*t6*t13*4.8E1+t3*t5*t9*t14*1.6E1+t3*
        t6*t9*t14*1.6E1+t5*t6*t9*t14*1.2E1-R*t3*t5*t6*t9*1.6E1-R*t3*t4*t7*t11*
        6.4E1-R*t3*t5*t7*t11*2.4E1-R*t3*t6*t7*t11*2.4E1-R*t4*t5*t7*
        t11*3.2E1-R*t4*t6*t7*t11*3.2E1);

        if(std::isnan(res)) return 0.0;
        return res;
    }

    float_type psi(float_type x, float_type y, float_type z) const noexcept
    {

        x-=center[0];
        y-=center[1];
        z-=center[2];
        const float_type r=std::sqrt(x*x+y*y+z*z);
        const float_type t=std::sqrt( (r-R)*(r-R) +z*z )/R;
        if(std::fabs(t)>=1.0) return 0.0;
        return  c1* std::exp(- c2/ (1-t*t) );
    }

public:
    coordinate_type<float_type,Dim> center;
    float_type R;
    float_type c1;
    float_type c2;
};

struct vortex_ring
{
    float_type vorticity(float_type x, float_type y, float_type z) const noexcept
    {
        x-=center[0];
        y-=center[1];
        z-=center[2];

        const float_type r=std::sqrt(x*x+y*y) ;
        const float_type t=std::sqrt( (r-R)*(r-R) +z*z )/R;

        if(std::fabs(t)>=1.0) return 0.0;

        const float_type t3 = x*x;
        const float_type t4 = y*y;
        const float_type t5 = t3+t4;
        const float_type t19 = std::sqrt(t5);
        const float_type t2 = R-t19;
        const float_type t6 = t5*t5;
        const float_type t7 = z*z;
        const float_type t8 = R*R;
        const float_type t9 = std::pow(t5,3.0/2.0);
        const float_type t10 = t3*t3;
        const float_type t11 = t4*t4;
        const float_type t12 = R*c2*t8*4.0;
        const float_type t13 = R*t7*2.0;
        const float_type t14 = R*t8*4.0;
        const float_type t15 = t12+t13+t14;
        const float_type t16 = t7*2.0;
        const float_type t24 = t8*8.0;
        const float_type t17 = -t16+t24;
        const float_type t18 = t7*t7;
        const float_type t20 = t8*t8;
        const float_type t21 = c2*t20*2.0;
        const float_type t22 = t7*t8*8.0;
        const float_type t23 = t21+t22;
        const float_type t25 = c2*t8*2.0;
        const float_type t26 = t7*4.0;
        const float_type t27 = t24+t25+t26;
        const float_type t28 = t6*t6;
        const float_type t29 = std::pow(t5,7.0/2.0);
        const float_type t30 = std::pow(t5,9.0/2.0);
        const float_type t31 = std::pow(t5,5.0/2.0);
        const float_type t32 = R*t7*2.4E1;
        const float_type t33 = R*t8*3.2E1;
        const float_type t34 = t32+t33;

        const float_type res =
        (c1*c2*t8*std::exp(c2/(1.0/(R*R)*(t7+t2*t2)-1.0))*
        (t29*4.0-t9*t10*3.0-t9*t11*3.0-t10*t15-t11*t15-t8*t31*1.2E1+
        t9*(t18-t7*t8*8.0+c2*t7*t8*2.0)-t3*t4*(R*t7*4.0+R*t8*8.0+R*c2*t8*8.0)-
        R*t5*t6*3.0+R*t3*t18+R*t4*t18-t3*t4*t9*6.0+t3*t9*t17+t4*t9*t17+
        t3*t19*t23+t4*t19*t23+t10*t19*t27+t11*t19*t27+
        t3*t4*t19*(t7*8.0+t8*1.6E1+c2*t8*4.0))*2.0)/
        (t3*t30+t4*t30+t7*t30*4.0+t20*t29*1.6E1+t9*(t18*t18)-t5*t6*
        (R*t18*2.4E1+R*t7*t8*3.2E1)-R*t5*t28*8.0+t9*t10*t18*6.0+t9*t11*t18*6.0+
        t8*t10*t31*2.4E1+t8*t11*t31*2.4E1+t8*t18*t31*2.4E1-R*t7*t10*t18*8.0-
        R*t7*t11*t18*8.0+t3*t4*t9*t18*1.2E1+t3*t7*t9*t18*4.0+t4*t7*t9*t18*4.0+
        t3*t4*t8*t31*4.8E1-t3*t5*t6*t34+t3*t7*t8*t31*4.8E1-t4*t5*t6*t34+
        t4*t7*t8*t31*4.8E1-R*t3*t4*t7*t18*1.6E1);


        if(std::isnan(res)) return 0.0;
        return res;
    }

    float_type psi(float_type x, float_type y, float_type z) const noexcept
    {

        x-=center[0];
        y-=center[1];
        z-=center[2];
        const float_type r=std::sqrt(x*x+y*y);
        const float_type t=std::sqrt( (r-R)*(r-R) +z*z )/R;
        if(std::fabs(t)>=1.0) return 0.0;
        return  c1* std::exp(- c2/ (1-t*t) );
    }

public:
    coordinate_type<float_type,Dim> center;
    float_type R;
    float_type c1;
    float_type c2;
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


        vrings_=this->read_vrings(simulation_.dictionary_.get());
        float_type max_vort=0.0;
        for(auto& vr : vrings_)
        {
            const auto center=vr.center;
            max_vort=std::max(max_vort,
                std::fabs(vr.vorticity(center[0]+vr.R,center[1],center[2])));
            pcout<<"vor max:"<<std::fabs(vr.vorticity(center[0]+vr.R,center[1],center[2]))<<std::endl;
        }

        vorticity_max_ =simulation_.dictionary_->
            template get_or<float_type>("source_max",max_vort);
        pcout<<"source_max "<<vorticity_max_<<std::endl;

        refinement_factor_ =simulation_.dictionary_->
            template get<float_type>("refinement_factor");
        pcout<<"Refienment factor "<<refinement_factor_<<std::endl;
        use_correction_ =simulation_.dictionary_->
            template get_or<bool>("correction", true);
        pcout<<"Using correction "<<std::boolalpha<<use_correction_<<std::endl;


        pcout << "\n Setup:  Test - Vortex rings \n" << std::endl;
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

    template<class Dict>
    std::vector<vortex_ring> read_vrings(Dict* _dict)
    {
        std::vector<vortex_ring> vrings;
        auto dicts=_dict->get_all_dictionaries("vortex");
        for(auto& d :  dicts)
        {
            vortex_ring v_tmp;
            v_tmp.center =d->template get<float_type,3>("center");
            v_tmp.c1 =d->template get<float_type>("c1");
            v_tmp.c2 =d->template get<float_type>("c2");
            v_tmp.R  =d->template get<float_type>("R");
            vrings.push_back(v_tmp);
        }
        return vrings;
    }

    void run()
    {

        std::ofstream ofs,ofs_level, ofs_timings;
        parallel_ostream::ParallelOstream
            pofs(io::output().dir()+"/"+"global_timings.txt",1,ofs),
            pofs_level(io::output().dir()+"/"+"level_timings.txt",1,ofs_level);


        boost::mpi::communicator world;
        //simulation_.write2("mesh.hdf5");

        auto pts=domain_->get_nPoints();

        if(domain_->is_client())
        {
            auto pts=domain_->get_nPoints();
            poisson_solver_t psolver(&this->simulation_);

            psolver.use_correction()=use_correction_;

            mDuration_type solve_duration(0);
            client_comm_.barrier();

            pcout_c<<"Poisson equation ---------------------------------" << std::endl;
            TIME_CODE( solve_duration, SINGLE_ARG(
                psolver.solve<source, phi_num>();
                client_comm_.barrier();
            ))

            pcout_c<<"Elapsed time "<< solve_duration.count()/1.0e3 <<" Rate "<< pts.back()/(solve_duration.count()/1.0e3)<< std::endl;

#ifdef POISSON_TIMINGS
            psolver.print_timings(pofs, pofs_level);
#endif
            psolver.apply_laplace<phi_num,amr_lap_source>() ;
        }

        this->compute_errors<phi_num,phi_exact,error>();
        this->compute_errors<amr_lap_source,source,error_lap_source>("laplace_");

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

        // Loop through leaves and assign values
        int nLocally_owned=0;
        int nGhost=0;
        int nAllocated=0;
        for (auto it  = domain_->begin();it != domain_->end(); ++it)
        {
            if(it.ptr())
            {
                if(it->locally_owned() && it->data()) { ++nLocally_owned; }
                else if (it->data())
                {
                    ++nGhost;
                    if(it->data()->is_allocated()) ++nAllocated;
                }
            }
        }
        //std::cout <<"rank: "<<world.rank()
        //          <<", owned: "<<nLocally_owned
        //          <<", ghosts: "<<nGhost
        //          <<", allocated ghosts: "<<nAllocated<<std::endl;

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

               it2->template get<source>() = vorticity(x,y,z);
               it2->template get<phi_exact>() = psi(x,y,z);
               /***********************************************************/
           }
        }
    }

    float_type vorticity(float_type x, float_type y, float_type z) const noexcept
    {
        float_type vort=0.0;
        for(auto& vr : vrings_)
        {
            vort+=vr.vorticity(x,y,z);
        }
        return vort;
    }

    float_type psi(float_type x, float_type y, float_type z) const noexcept
    {
        float_type psi=0.0;
        for(auto& vr : vrings_)
        {
            psi+=vr.psi(x,y,z);
        }
        return psi;
    }

    int get_nPoints() const noexcept
    {
        if(!domain_->is_client()) return 0;

        int nPts=0;
        int nPts_global=0;
        for (auto it  = domain_->begin_leafs();
                  it != domain_->end_leafs(); ++it)
        {
            if(it->data()) nPts+=it->data()->node_field().size();
        }
        boost::mpi::all_reduce(client_comm_,nPts, nPts_global, std::plus<int>());
        return nPts_global;
    }


    /** @brief  Refienment conditon for octants.  */
    template<class OctantType>
    bool refinement(OctantType* it, int diff_level,
            bool use_all=false) const noexcept
    {
        auto b=it->data()->descriptor();
        b.level()=it->refinement_level();
        const float_type dx_base = domain_->dx_base();

        return refinement(b, dx_base, vorticity_max_, diff_level,use_all);
    }

    /** @brief  Refienment conditon for blocks.  */
    bool refinement(block_descriptor_t b,
                    float_type dx_base,
                    float_type vorticity_max,
                    int diff_level,
                    bool use_all=false) const noexcept
    {
        auto center = (domain_->bounding_box().max() -
                       domain_->bounding_box().min()) / 2.0 +
                       domain_->bounding_box().min();

        auto scaling =  std::pow(2,b.level());
        center*=scaling;
        auto dx_level =  dx_base/std::pow(2,b.level());

        b.grow(2,2);
        auto corners= b.get_corners();
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

                    const auto vort=vorticity(x,y,z);
                    if(std::fabs(vort) >
                            //vorticity_max_*pow(0.25*0.25*0.5 , diff_level)
                            //vorticity_max_*pow(0.25*0.5 , diff_level)
                            //vorticity_max_*pow(0.25 , diff_level)
                            vorticity_max_*pow(refinement_factor_ , diff_level)
                      )
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /** @brief  Initialization of the domain blocks. This is registered in the
     *          domain through the base setup class, passing it to the domain ctor. */
    std::vector<extent_t> initialize_domain( Dictionary* _d, domain_t* _domain )
    {
        auto res=_domain-> construct_basemesh_blocks(_d, _domain->block_extent());
        domain_->read_parameters(_d);
        return res;
    }


private:
    float_type vorticity_max_;
    std::vector<vortex_ring> vrings_;
    float_type refinement_factor_=1./8;
    bool use_correction_=true;
};


#endif // IBLGF_INCLUDED_POISSON_HPP
