//      ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
//      ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌
//          ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄
//          ▐░▌     ▐░░░░░░░░░░▌ ▐░▌          ▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌
//          ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌
//      ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌
//      ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀

#ifndef IBLGF_INCLUDED_NS_AMR_LGF_HPP
#define IBLGF_INCLUDED_NS_AMR_LGF_HPP

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <vector>
#include <math.h>
#include <chrono>
#include <fftw3.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/domain/dataFields/dataBlock.hpp>
#include <iblgf/domain/dataFields/datafield.hpp>
#include <iblgf/domain/octree/tree.hpp>
#include <iblgf/IO/parallel_ostream.hpp>
#include <iblgf/lgf/lgf.hpp>
#include <iblgf/fmm/fmm.hpp>

#include <iblgf/utilities/convolution.hpp>
#include <iblgf/interpolation/interpolation.hpp>
#include <iblgf/solver/poisson/poisson.hpp>
#include <iblgf/solver/time_integration/ifherk.hpp>

#include "../../setups/setup_base.hpp"
#include <iblgf/operators/operators.hpp>

namespace iblgf
{
const int Dim = 3;

struct parameters
{
    static constexpr std::size_t Dim = 3;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
     (
        //name               type        Dim   lBuffer  hBuffer, storage type
         (error_u          , float_type, 3,    1,       1,     face,false ),
         (error_p          , float_type, 1,    1,       1,     cell,false ),
         (decomposition    , float_type, 1,    1,       1,     cell,false ),
        //IF-HERK
         (u                , float_type, 3,    1,       1,     face,true ),
         (u_ref            , float_type, 3,    1,       1,     face,false ),
         (p_ref            , float_type, 1,    1,       1,     cell,false ),
         (p                , float_type, 1,    1,       1,     cell,true )
    ))
    // clang-format on
};

struct NS_AMR_LGF : public SetupBase<NS_AMR_LGF, parameters>
{
    using super_type =SetupBase<NS_AMR_LGF,parameters>;
    using vr_fct_t = std::function<float_type(float_type x, float_type y, float_type z, int field_idx, float_type perturbation)>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

    NS_AMR_LGF(Dictionary* _d)
    : super_type(_d, [this](auto _d, auto _domain) {
        return this->initialize_domain(_d, _domain);
    })
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);

        dx_ = domain_->dx_base();
        cfl_ =
            simulation_.dictionary()->template get_or<float_type>("cfl", 0.2);
        dt_ = simulation_.dictionary()->template get_or<float_type>("dt", -1.0);

        tot_steps_      = simulation_.dictionary()->template get<int>("nBaseLevelTimeSteps");
        Re_             = simulation_.dictionary()->template get<float_type>("Re");
        R_              = simulation_.dictionary()->template get<float_type>("R");
        d2v_            = simulation_.dictionary()->template get_or<float_type>("DistanceOfVortexRings", R_);
        v_delta_        = simulation_.dictionary()->template get_or<float_type>("vDelta", 0.2*R_);
        single_ring_    = simulation_.dictionary()->template get_or<bool>("single_ring", true);
        perturbation_   = simulation_.dictionary()->template get_or<float_type>("perturbation", 0.0);

        bool use_fat_ring = simulation_.dictionary()->template get_or<bool>("fat_ring", false);
        if (use_fat_ring)
            vr_fct_=
            [this](float_type x, float_type y, float_type z, int field_idx, float_type perturbation){return this->vortex_ring_vor_fat_ic(x,y,z,field_idx, perturbation);};
        else
            vr_fct_=
            [this](float_type x, float_type y, float_type z, int field_idx, float_type perturbation){return this->vortex_ring_vor_ic(x,y,z,field_idx, perturbation);};

        ic_filename_ = simulation_.dictionary_->template get_or<std::string>(
            "hdf5_ic_name", "null");
        ref_filename_ = simulation_.dictionary_->template get_or<std::string>(
            "hdf5_ref_name", "null");

        source_max_ = simulation_.dictionary_->template get_or<float_type>(
            "source_max", 1.0);

        refinement_factor_ = simulation_.dictionary_->template get<float_type>(
            "refinement_factor");

        base_threshold_ = simulation_.dictionary()->
            template get_or<float_type>("base_level_threshold", 1e-4);

        nLevelRefinement_=simulation_.dictionary_->
            template get_or<int>("nLevels",0);

        hard_max_level_ = simulation_.dictionary()->template get_or<int>("hard_max_level", nLevelRefinement_);

        global_refinement_ = simulation_.dictionary_->template get_or<int>(
            "global_refinement", 0);

        if (dt_ < 0) dt_ = dx_ * cfl_;

        dt_ /= pow(2.0, nLevelRefinement_);
        tot_steps_ *= pow(2, nLevelRefinement_);

        pcout << "\n Setup:  Test - Simple IC \n" << std::endl;
        pcout << "Number of refinement levels: " << nLevelRefinement_
              << std::endl;

        domain_->register_adapt_condition()=
            [this]( std::vector<float_type> source_max, auto& octs, std::vector<int>& level_change )
                {return this->template adapt_level_change(source_max, octs, level_change);};

        domain_->register_refinement_condition() = [this](auto octant,
                                                       int diff_level) {
            return this->refinement(octant, diff_level);
        };

        if (!use_restart())
        {
            domain_->init_refine(_d->get_dictionary("simulation_parameters")
                                     ->template get_or<int>("nLevels", 0),
                global_refinement_);
        }
        else
        {
            domain_->restart_list_construct();
        }

        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        if (!use_restart()) { this->initialize(); }
        else
        {
            simulation_.template read_h5<u_type>(simulation_.restart_field_dir(),"u");
        }

        boost::mpi::communicator world;
        if (world.rank() == 0)
            std::cout << "on Simulation: \n" << simulation_ << std::endl;
    }

    float_type run()
    {
        boost::mpi::communicator world;

        time_integration_t ifherk(&this->simulation_);

        if (ic_filename_!="null")
            simulation_.template read_h5<u_type>(ic_filename_,"u");

        mDuration_type ifherk_duration(0);
        TIME_CODE( ifherk_duration, SINGLE_ARG(
                    ifherk.time_march(use_restart());
                    ))
        pcout_c<<"Time to solution [ms] "<<ifherk_duration.count()<<std::endl;

        if (ref_filename_!="null")
        {
            simulation_.template read_h5<u_ref_type>(ref_filename_, "u");
            simulation_.template read_h5<p_ref_type>(ref_filename_, "p");

            auto center = (domain_->bounding_box().max() -
                    domain_->bounding_box().min()+1) / 2.0 +
                    domain_->bounding_box().min();


            for (auto it  = domain_->begin_leaves();
                    it != domain_->end_leaves(); ++it)
            {
                if(!it->locally_owned()) continue;

                auto dx_level =  domain_->dx_base()/std::pow(2,it->refinement_level());
                auto scaling =  std::pow(2,it->refinement_level());

                for (auto& node : it->data())
                {
                    const auto& coord = node.level_coordinate();
                    float_type x = static_cast<float_type>
                        (coord[0]-center[0]*scaling)*dx_level;
                    float_type y = static_cast<float_type>
                        (coord[1]-center[1]*scaling)*dx_level;
                    float_type z = static_cast<float_type>
                        (coord[2]-center[2]*scaling)*dx_level;

                    float_type r2 = x*x+y*y;
                    if (std::fabs(z)>R_ || r2>4*R_*R_)
                    {
                        node(u_ref, 0)=0.0;
                        node(u_ref, 1)=0.0;
                        node(u_ref, 2)=0.0;
                    }
                }

            }
        }

        ifherk.clean_leaf_correction_boundary<u_type>(domain_->tree()->base_level(),true,1);

        this->compute_errors<u_type, u_ref_type, error_u_type>(
                std::string("u1_"), 0);
        this->compute_errors<u_type, u_ref_type, error_u_type>(
                std::string("u2_"), 1);
        float_type u3_linf=this->compute_errors<u_type, u_ref_type, error_u_type>(
                std::string("u3_"), 2);

        simulation_.write("final.hdf5");
        return u3_linf;
    }

    template< class key_t >
    void adapt_level_change(std::vector<float_type> source_max,
                            std::vector<key_t>& octs,
                            std::vector<int>&   level_change )
    {
        octs.clear();
        level_change.clear();
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {

            if (!it->locally_owned()) continue;
            if (!it->is_leaf() && !it->is_correction()) continue;
            if (it->is_leaf() && it->is_correction()) continue;

            int l1=-1;
            int l2=-1;
            int l3=-1;

            if (!it->is_correction() && it->is_leaf())
                l1=this->template adapt_levle_change_for_field<cell_aux_type>(it, source_max[0], true);

            if (it->is_correction() && !it->is_leaf())
                l2=this->template adapt_levle_change_for_field<correction_tmp_type>(it, source_max[0], false);

            if (!it->is_correction() && it->is_leaf())
                l3=this->template adapt_levle_change_for_field<edge_aux_type>(it, source_max[1], false);

            int l=std::max(std::max(l1,l2),l3);

            if( l!=0)
            {
                if (it->is_leaf()&&!it->is_correction())
                {
                    octs.emplace_back(it->key());
                    level_change.emplace_back(l);
                }
                else if (it->is_correction() && !it->is_leaf() && l2>0)
                {
                    octs.emplace_back(it->key().parent());
                    level_change.emplace_back(l2);
                }
            }
        }
    }

    template<class Field, class OctantType>
    int adapt_levle_change_for_field(OctantType it, float_type source_max, bool use_base_level_threshold)
    {
        source_max *=1.05;

        // ----------------------------------------------------------------

        float_type field_max=
            domain::Operator::blockRootMeanSquare<Field>(it->data());

        if (field_max<1e-10) return -1;

        // to refine and harder to delete
        // This prevent rapid change of level refinement
        float_type deletion_factor=0.7;

        int l_aim = static_cast<int>( ceil(nLevelRefinement_-log(field_max/source_max) / log(refinement_factor_)));
        int l_delete_aim = static_cast<int>( ceil(nLevelRefinement_-((log(field_max/source_max) - log(deletion_factor)) / log(refinement_factor_))));

        if (l_aim>hard_max_level_)
            l_aim=hard_max_level_;

        if (it->refinement_level()==0 && use_base_level_threshold)
        {
            if (field_max>source_max*base_threshold_)
                l_aim = std::max(l_aim,0);

            if (field_max>source_max*base_threshold_*deletion_factor)
                l_delete_aim = std::max(l_delete_aim,0);
        }

        int l_change = l_aim - it->refinement_level();
        if (l_change>0)
            return 1;

        l_change = l_delete_aim - it->refinement_level();
        if (l_change<0) return -1;

        return 0;
    }


    /** @brief Initialization of poisson problem.
     *  @detail Testing poisson with manufactured solutions: exp
     */
    void initialize()
    {
        poisson_solver_t psolver(&this->simulation_);

        boost::mpi::communicator world;
        if(domain_->is_server()) return ;
        auto center = (domain_->bounding_box().max() -
                       domain_->bounding_box().min()+1) / 2.0 +
                       domain_->bounding_box().min();

        // Adapt center to always have peak value in a cell-center
        //center+=0.5/std::pow(2,nRef);
        const float_type dx_base = domain_->dx_base();

        if (ic_filename_ != "null") return;

        // Voriticity IC
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if(!it->locally_owned()) continue;

            auto dx_level =  dx_base/std::pow(2,it->refinement_level());
            auto scaling =  std::pow(2,it->refinement_level());

            for (auto& node : it->data())
            {

                const auto& coord = node.level_coordinate();

                float_type x = static_cast<float_type>
                (coord[0]-center[0]*scaling+0.5)*dx_level;
                float_type y = static_cast<float_type>
                (coord[1]-center[1]*scaling)*dx_level;
                float_type z = static_cast<float_type>
                (coord[2]-center[2]*scaling)*dx_level;

                node(edge_aux,0) = vor(x,y,z,0);

                /***********************************************************/
                x = static_cast<float_type>
                (coord[0]-center[0]*scaling)*dx_level;
                y = static_cast<float_type>
                (coord[1]-center[1]*scaling+0.5)*dx_level;
                z = static_cast<float_type>
                (coord[2]-center[2]*scaling)*dx_level;

                node(edge_aux,1) = vor(x,y,z,1);

                /***********************************************************/
                x = static_cast<float_type>
                (coord[0]-center[0]*scaling)*dx_level;
                y = static_cast<float_type>
                (coord[1]-center[1]*scaling)*dx_level;
                z = static_cast<float_type>
                (coord[2]-center[2]*scaling+0.5)*dx_level;

                node(edge_aux,2) = vor(x,y,z,2);
            }

        }

        psolver.template apply_lgf<edge_aux_type, stream_f_type>();
        auto client = domain_->decomposition().client();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            //client->template buffer_exchange<stream_f_type>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                const auto dx_level =
                    dx_base / std::pow(2, it->refinement_level());
                domain::Operator::curl_transpose<stream_f_type, u_type>(
                    it->data(), dx_level, -1.0);
            }
            client->template buffer_exchange<u_type>(l);
        }
    }


    float_type vortex_ring_vor_fat_ic(float_type x, float_type y, float_type z, int field_idx, float_type perturbation)
    {
        const float_type alpha = 0.54857674;
        float_type       R2 = R_ * R_;

        float_type r2 = x * x + y * y;
        float_type r = sqrt(r2);
        float_type s2 = z * z + (r - R_) * (r - R_);

        float_type theta = std::atan2(y, x);
        float_type w_theta = alpha * 1.0 / R2 * std::exp(-4.0 * s2 / (R2 - s2));

        float_type rd = (static_cast <float_type> (rand()) / static_cast <float_type> (RAND_MAX))-0.5;
        rd *= perturbation;

        if (s2>=R2) return 0.0;

        if (field_idx==0)
            return -w_theta*std::sin(theta)*(1+rd);
        else if (field_idx==1)
            return w_theta*std::cos(theta)*(1+rd);
        else
            return 0.0;
    }

    float_type vor(float_type x, float_type y, float_type z, int field_idx) const
    {
        if (single_ring_)
            return vr_fct_(x,y,z,field_idx,perturbation_);
        else
            return -vr_fct_(x,y,z-d2v_/2,field_idx,perturbation_)+vr_fct_(x,y,z+d2v_/2,field_idx,perturbation_);
    }

    float_type noise(float_type theta, float_type perturbation, int leftring)
    {
        std::vector<std::vector<float_type>> rd_phase{
            {0.23952, 0.93281, 0.792195, 0.743415, 0.809997, 0.25604, 0.934906, \
            0.887225, 0.73549, 0.458176, 0.457471, 0.742074, 0.514981, 0.679815, \
            0.543974, 0.886245, 0.972392, 0.774492, 0.43179, 0.864585, 0.600469, \
            0.0403256, 0.146207, 0.48862, 0.690213, 0.925106, 0.17203, 0.370254, \
            0.065236, 0.898992, 0.189733, 0.736777, 0.664988, 0.152577, 0.144798, \
            0.846399, 0.648571, 0.63376, 0.110466, 0.672082, 0.536401},
            {0.422702, 0.936246, 0.881116, 0.694646, 0.668566, 0.574301, \
            0.557115, 0.865498, 0.126846, 0.541819, 0.491133, 0.893088, 0.831923, \
            0.834021, 0.0966956, 0.80156, 0.244035, 0.121746, 0.932934, \
            0.0572397, 0.116674, 0.737242, 0.585802, 0.466478, 0.735447, \
            0.231485, 0.980104, 0.295989, 0.916545, 0.408343, 0.0579712, \
            0.0496912, 0.39271, 0.285154, 0.390967, 0.851305, 0.966866, 0.854457, \
            0.299213, 0.97338, 0.186061, 0.564841}
            };


        float_type tmp=0.0;
        int N=32;
        for (int i=1;i<=N;++i)
            tmp+=  std::cos(theta*i+rd_phase[leftring][i-1]*2*M_PI);

        return perturbation*tmp;
    }

    float_type vortex_ring_vor_ic(float_type x, float_type y, float_type z, int field_idx, float_type perturbation)
    {
        float_type delta_2 = v_delta_ * v_delta_;

        float_type r2 = x * x + y * y;
        float_type r = sqrt(r2);
        float_type theta = std::atan2(y, x);
        int leftring=(z>0)? 1:0;
        float_type s2 = z * z + (r - R_*(1 + noise(theta, perturbation, leftring))) * (r - R_*(1 + noise(theta, perturbation, leftring)));

        float_type w_theta = 1.0 / M_PI / delta_2 * std::exp(-s2 / delta_2);

        //float_type rd = (static_cast <float_type> (rand()) / static_cast <float_type> (RAND_MAX));
        //rd *= perturbation;

        if (field_idx == 0) return -w_theta * std::sin(theta);
        else if (field_idx == 1)
            return w_theta * std::cos(theta);
        else
            return 0;
    }

    /** @brief  Refienment conditon for octants.  */
    template<class OctantType>
    bool refinement(OctantType* it, int diff_level, bool use_all = false) const
        noexcept
    {
        auto b = it->data().descriptor();
        b.level() = it->refinement_level();
        const float_type dx_base = domain_->dx_base();

        auto center = (domain_->bounding_box().max() -
                       domain_->bounding_box().min()+1) / 2.0 +
                       domain_->bounding_box().min();

        auto scaling = std::pow(2, b.level());
        center *= scaling;
        auto dx_level = dx_base / std::pow(2, b.level());

        b.grow(2, 2);
        auto corners = b.get_corners();

        float_type w_max = std::abs(vr_fct_(float_type(R_),float_type(0.0),float_type(0.0),1,perturbation_));

        for (int i = b.base()[0]; i <= b.max()[0]; ++i)
        {
            for (int j = b.base()[1]; j <= b.max()[1]; ++j)
            {
                for (int k = b.base()[2]; k <= b.max()[2]; ++k)
                {
                    float_type x = static_cast<float_type>
                    (i-center[0]+0.5)*dx_level;
                    float_type y = static_cast<float_type>
                    (j-center[1])*dx_level;
                    float_type z = static_cast<float_type>
                    (k-center[2])*dx_level;

                    float_type tmp_w = vor(x,y,z,0);

                    if(std::fabs(tmp_w) > w_max*pow(refinement_factor_, diff_level))
                        return true;

                    x = static_cast<float_type>(i - center[0]) * dx_level;
                    y = static_cast<float_type>(j - center[1] + 0.5) * dx_level;
                    z = static_cast<float_type>(k - center[2]) * dx_level;

                    tmp_w = vor(x,y,z,1);

                    if(std::fabs(tmp_w) > w_max*pow(refinement_factor_, diff_level))
                        return true;

                    x = static_cast<float_type>(i - center[0]) * dx_level;
                    y = static_cast<float_type>(j - center[1]) * dx_level;
                    z = static_cast<float_type>(k - center[2] + 0.5) * dx_level;

                    tmp_w = vor(x,y,z,2);

                    if(std::fabs(tmp_w) > w_max*pow(refinement_factor_, diff_level))
                        return true;
                }
            }
        }

        return false;
    }

    /** @brief  Initialization of the domain blocks. This is registered in the
     *          domain through the base setup class, passing it to the domain ctor.
     */
    std::vector<coordinate_t> initialize_domain(
        Dictionary* _d, domain_t* _domain)
    {
        auto res =
            _domain->construct_basemesh_blocks(_d, _domain->block_extent());
        domain_->read_parameters(_d);

        return res;
    }

    private:

    boost::mpi::communicator client_comm_;

    bool single_ring_=true;
    float_type perturbation_;

    float_type R_;
    float_type v_delta_;
    float_type d2v_;
    float_type source_max_;

    float_type rmin_ref_;
    float_type rmax_ref_;
    float_type rz_ref_;
    float_type c1=0;
    float_type c2=0;
    float_type eps_grad_=1.0e6;;
    int nLevelRefinement_=0;
    int hard_max_level_;
    int global_refinement_=0;
    fcoord_t offset_;

    float_type a_ = 10.0;

    float_type dt_,dx_;
    float_type cfl_;
    float_type Re_;
    int tot_steps_;
    float_type refinement_factor_=1./8;
    float_type base_threshold_=1e-4;

    vr_fct_t vr_fct_;

    std::string ic_filename_, ref_filename_;
};

double vortex_run(std::string input, int argc = 0, char** argv = nullptr);

} // namespace iblgf

#endif
