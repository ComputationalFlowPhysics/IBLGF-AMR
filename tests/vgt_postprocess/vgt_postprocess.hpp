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

#ifndef IBLGF_INCLUDED_VGT_POSTPROCESS_HPP
#define IBLGF_INCLUDED_VGT_POSTPROCESS_HPP

#include <iostream>
#include <cmath>
// #include <xtensor/xarray.hpp>
// #include <xtensor-blas/xlinalg.hpp>
#include "../../setups/setup_base.hpp"
#include <iblgf/operators/operators.hpp>
// #include 
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
         (test             , float_type, 1,    1,       1,     cell,false ),
        //IF-HERK
         (u                , float_type, 3,    1,       1,     face,true ),
         (p                , float_type, 1,    1,       1,     cell,true ),
         (grad              ,float_type, 9,1,1, cell, true),
         (omega_r              ,float_type, 1,1,1, cell, true),
         (qq              ,float_type, 1,1,1, cell, true),
         (A_norm              ,float_type, 1,1,1, cell, true)
    ))
    // clang-format on
};
struct VGT_PostProcess : public SetupBase<VGT_PostProcess, parameters>
{
    using super_type = SetupBase<VGT_PostProcess, parameters>;
    using vr_fct_t =
        std::function<float_type(float_type x, float_type y, float_type z, int field_idx, float_type perturbation)>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

    // VGT_PostProcess(Dictionary* _d)
    // : super_type(_d, [this](auto _d, auto _domain) { return this->initialize_domain(_d, _domain); })
    // {
    //     if (domain_->is_client()) client_comm_ = client_comm_.split(1);
    //     else
    //         client_comm_ = client_comm_.split(0);
    //     // ------------------------------------------------------------------
    //     // ref frame velocity

    //     U_.resize(domain_->dimension());
    //     U_[0] = simulation_.dictionary()->template get_or<float_type>("Ux", 0.0);
    //     U_[1] = simulation_.dictionary()->template get_or<float_type>("Uy", 0.0);
    //     if (domain_->dimension() > 2) U_[2] = simulation_.dictionary()->template get_or<float_type>("Uz", 0.0);

      
    //     // // ------------------------------------------------------------------
    //     dx_ = domain_->dx_base();
    //     cfl_ = simulation_.dictionary()->template get_or<float_type>("cfl", 0.2);
    //     dt_ = simulation_.dictionary()->template get_or<float_type>("dt", -1.0);

    //     tot_steps_ = simulation_.dictionary()->template get<int>("nBaseLevelTimeSteps");
    //     Re_ = simulation_.dictionary()->template get<float_type>("Re");
        
    //     nLevelRefinement_ = simulation_.dictionary_->template get_or<int>("nLevels", 0);

    //     hard_max_level_ = simulation_.dictionary()->template get_or<int>("hard_max_level", nLevelRefinement_);

    //     global_refinement_ = simulation_.dictionary_->template get_or<int>("global_refinement", 0);

    //     //if (global_refinement_ == 0) global_refinement_ = nLevelRefinement_;

    //     if (dt_ < 0) dt_ = dx_ * cfl_;

    //     dt_ /= pow(2.0, nLevelRefinement_);
    //     tot_steps_ *= pow(2, nLevelRefinement_);

    //     pcout << "\n Setup:  Test - Simple IC \n" << std::endl;
    //     pcout << "Number of refinement levels: " << nLevelRefinement_ << std::endl;

    //     nIB_add_level_ = _d->get_dictionary("simulation_parameters")->template get_or<int>("nIB_add_level", 0);

    //     domain_->ib().init(_d->get_dictionary("simulation_parameters"), domain_->dx_base(),
    //         nLevelRefinement_ + nIB_add_level_, Re_);

    //     // if (!use_restart()) { domain_->init_refine(nLevelRefinement_, global_refinement_, nIB_add_level_); }
    //     // else { domain_->restart_list_construct(); }
    //     domain_->restart_list_construct();
    //     domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
    //     // if (!use_restart()) { this->initialize(); }
    //     // else { simulation_.template read_h5<u_type>(simulation_.restart_field_dir(), "u"); }
    //     simulation_.template read_h5<u_type>(simulation_.restart_field_dir(), "u");

    //     boost::mpi::communicator world;
    //     if (world.rank() == 0) std::cout << "on Simulation: \n" << simulation_ << std::endl;
    // }
    VGT_PostProcess(Dictionary* _d, std::string restart_tree_dir, std::string restart_field_dir)
    : super_type(_d,
        [this](auto _d, auto _domain) { return this->initialize_domain(_d, _domain); },
        restart_tree_dir)
    , restart_tree_dir_(restart_tree_dir)
    , restart_field_dir_(restart_field_dir)
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);
        boost::mpi::communicator world;
        nLevelRefinement_ = simulation_.dictionary_->template get_or<int>("nLevels", 0);
        // std::cout << "Number of refinement levels: " << nLevelRefinement_ << std::endl;
        // std::cout << "Restarting list construction..." << std::endl;
        // domain_->register_adapt_condition()=
        //     [this]( std::vector<float_type> source_max, auto& octs, std::vector<int>& level_change )
        //         {return this->template adapt_level_change(source_max, octs, level_change);};

        domain_->register_refinement_condition() = [this](auto octant,
                                                    int diff_level) {
            return false;
        };
        // domain_->init_refine(nLevelRefinement_, 0, 0);
        if(world.rank() == 0)
        {
            std::cout<<"Restarting list construction..."<<std::endl;
        }
        domain_->restart_list_construct();
        if(world.rank() == 0)
        {
            std::cout<<"Restarting list constructed."<<std::endl;
        }
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        if(world.rank() == 0)
        {
            std::cout<<"Domain distributed."<<std::endl;
        }
        simulation_.template read_h5<u_type>(restart_field_dir,"u");
        output_suffix_=simulation_.dictionary()->template get_or<std::string>("output_suffix","x");
        // simulation_.read(restart_tree_dir,"tree");
        // simulation_.read(restart_field_dir,"fields");
        

        if (world.rank() == 0) std::cout << "on Simulation: \n" << simulation_ << std::endl;
        
    }
    void run()
    {
        std::string output_name="postProc_"+output_suffix_;
        boost::mpi::communicator world;
        computeVGT();
        decompVGT();
        simulation_.write(output_name);
        return;

    }
    void run_batch(int idx_cur)
    {
        std::string output_name="postProc_"+std::to_string(idx_cur);
        boost::mpi::communicator world;
        computeVGT();
        decompVGT();
        simulation_.write(output_name);
        return;

    }
    void computeVGT()
    {
        boost::mpi::communicator world;
        time_integration_t ifherk(&this->simulation_);
        ifherk.clean<omega_r_type>();
        ifherk.clean<qq_type>();
        ifherk.clean<grad_type>();
        auto client = domain_->decomposition().client();
        if(domain_->is_client())
        {
            ifherk.pad_access<u_type, u_type>(true);
            ifherk.template up_and_down<u_type>();
            const float_type dx_base = domain_->dx_base();
            ifherk.curl<u_type>();
            
            for (int l= domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
            {
                client->template buffer_exchange<u_type>(l);
                
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned() || !it->has_data()) continue;
                    // if(!it->is_leaf()) continue;
                    auto dx_level =  dx_base/std::pow(2,it->refinement_level());
                    auto scaling =  std::pow(2,it->refinement_level());
                    for (auto &n : it->data())
                    {
                      //cell center dudx
                      n(grad,0)=(n.at_offset(u,1,0,0,0)-n(u,0))/dx_level;
                      // dudy
                      n(grad,1)=0.5*((n.at_offset(u,0,1,0,0)-n.at_offset(u,0,-1,0,0))+(n.at_offset(u,1,1,0,0)-n.at_offset(u,1,-1,0,0)))/dx_level;
                      //dudz
                    //   if(it->is_extended_ib())
                    //   {
                    //     std::cout<<(n.at_offset(u,0,1,0,0)-n.at_offset(u,0,-1,0,0))+(n.at_offset(u,1,1,0,0)-n.at_offset(u,1,-1,0,0))<<std::endl;
                    //     // std::cout<<1/2*((n.at_offset(u,0,1,0,0)-n.at_offset(u,0,-1,0,0))+(n.at_offset(u,1,1,0,0)-n.at_offset(u,1,-1,0,0)))<<std::endl;
                    //     }
                    n(grad,2)=0.5*((n.at_offset(u,0,0,1,0)-n.at_offset(u,0,0,-1,0))+(n.at_offset(u,1,0,1,0)-n.at_offset(u,1,0,-1,0)))/dx_level;

                    //dvdx
                    n(grad,3)=0.5*((n.at_offset(u,1,0,0,1)-n.at_offset(u,-1,0,0,1))+(n.at_offset(u,1,1,0,1)-n.at_offset(u,-1,1,0,1)))/dx_level;
                    //dvdy==
                    n(grad,4)=(n.at_offset(u,0,1,0,1)-n(u,1))/dx_level;
                    //dvdz
                    n(grad,5)=0.5*((n.at_offset(u,0,0,1,1)-n.at_offset(u,0,0,-1,1))+(n.at_offset(u,0,1,1,1)-n.at_offset(u,0,1,-1,1)))/dx_level;
                    //dwdx
                    n(grad,6)=0.5*((n.at_offset(u,1,0,0,2)-n.at_offset(u,-1,0,0,2))+(n.at_offset(u,1,0,1,2)-n.at_offset(u,-1,0,1,2)))/dx_level;  
                    //dwdy
                    n(grad,7)=0.5*((n.at_offset(u,0,1,0,2)-n.at_offset(u,0,-1,0,2))+(n.at_offset(u,0,1,1,2)-n.at_offset(u,0,-1,1,2)))/dx_level;
                    //dwdz
                    n(grad,8)=(n.at_offset(u,0,0,1,2)-n(u,2))/dx_level;
                    // n(grad)/= dx_level;
                    }


                }
                client->template buffer_exchange<grad_type>(l);

                // ifherk.clean_leaf_correction_boundary<grad_type>(l, false, 2);
            }
        }
        return;
    }
    void decompVGT()
    {
        boost::mpi::communicator world;
        time_integration_t ifherk(&this->simulation_);
        ifherk.clean<omega_r_type>();
        ifherk.clean<qq_type>();
        ifherk.clean<A_norm_type>();
        // ifherk.clean<grad_type>(true,2);
        auto                     client = domain_->decomposition().client();
        if (domain_->is_client())
        {
            for (int l = domain_->tree()->base_level(); l < domain_->tree()->depth(); ++l)
            {

                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned() || !it->has_data()) continue;
                    // if (!it->is_leaf()||it->is_correction()) continue;
                    for (auto& n : it->data())
                    {
                        float_type             orRR = 0.0;
                        xt::xarray<float_type> A = {{n(grad, 0), n(grad, 1), n(grad, 2)},
                            {n(grad, 3), n(grad, 4), n(grad, 5)}, {n(grad, 6), n(grad, 7), n(grad, 8)}};

                        // float_type trA = A(0, 0) + A(1, 1) + A(2, 2);
                        // A(0, 0) -= trA / 3.0;
                        // A(1, 1) -= trA / 3.0;
                        // A(2, 2) -= trA / 3.0;
                        // std::cout<<n(grad,1)<<std::endl;
                        // std::cout<<A<<std::endl;
                        
                        auto S= 1/2.0*(A+xt::transpose(A));
                        auto W= 1/2.0*(A-xt::transpose(A));
                        auto norm_W2 = xt::linalg::norm(W);
                        auto norm_S2 = xt::linalg::norm(S);
                        auto Qcrit = 0.5 * (norm_W2 * norm_W2 - norm_S2 * norm_S2);
                        auto norm_A= xt::linalg::norm(A)+1e-8;
                        // auto Qcrit=1/2.0*(xt::linalg::norm(W)-xt::linalg::norm(S));
                        A/=norm_A;
                        auto d = xt::linalg::eig(A);
                        // check if any eigenvalues are real
                        auto evals = std::get<0>(d);
                        auto evecs = std::get<1>(d);

                        int nReal{0};
                        int idx_real{0};
                        for (int i = 0; i < 3; ++i)
                        {
                            if (std::abs(std::imag(evals(i))) < 1e-3)
                            {
                                nReal++;
                                idx_real = i;
                            }
                        }
                        if (nReal == 1)
                        {
                            xt::xarray<float_type> vort = {A(2, 1) - A(1, 2), A(0, 2) - A(2, 0), A(1, 0) - A(0, 1)};
                            auto                   rot_axis = xt::real(xt::view(evecs, xt::all(), idx_real));

                            auto beta = xt::real(xt::linalg::dot(rot_axis,
                                vort))(); // maginutde of vorticity in the direction of the rotation axis
                            if (beta < 0.0)
                            {
                                beta = -beta;
                                rot_axis = -rot_axis;
                            }
                            auto Lci = xt::amax(xt::abs(xt::imag(evals)))();
                            // if (Lci > 0.5 * beta) beta = 2 * Lci;
                            orRR = std::pow(beta, 2) / (2 * std::pow(beta, 2) - 4 * std::pow(Lci, 2) +
                                                           2 * 1e-5); // Calculate the rotation rate
                        }
                        // if (nReal>1){std::cout << "No complex eigenvalues" << std::endl;}

                        //std::cout<<VGT<<std::endl;
                        // n(test) = VGT(0,0);
                        n(omega_r) = orRR;
                        n(qq) = Qcrit;
                        n(A_norm) = norm_A;
                    }
                }
                // ifherk.clean_leaf_correction_boundary<omega_r_type>(l, false, 2);
            }
        }
        return;
    }

    std::vector<coordinate_t> initialize_domain(
        Dictionary* _d, domain_t* _domain)
    {
        auto res =
            _domain->construct_basemesh_blocks(_d, _domain->block_extent());
        domain_->read_parameters(_d);

        return res;
    }
    private:
    std::vector<float_type> U_;
    boost::mpi::communicator client_comm_;
    float_type dt_,dx_;
    float_type cfl_;
    float_type Re_;
    int tot_steps_;
    float_type refinement_factor_=1./8;
    float_type base_threshold_=1e-4;
    int nLevelRefinement_=0;
    int hard_max_level_;
    int global_refinement_=0;
    int nIB_add_level_=0;
    std::string restart_tree_dir_;
    std::string restart_field_dir_;
    std::string output_suffix_;
};

} // namespace iblgf
#endif