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

#ifndef IBLGF_INCLUDED_RSVDDT_HPP
#define IBLGF_INCLUDED_RSVDDT_HPP


#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <array>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/IO/parallel_ostream.hpp>
#include <iblgf/solver/poisson/poisson.hpp>
#include <iblgf/solver/linsys/linsys.hpp>
#include <iblgf/operators/operators.hpp>
#include <iblgf/utilities/misc_math_functions.hpp>
#include <slepceps.h>
#include <slepcsys.h>
#include <random>
namespace iblgf
{
namespace solver
{
using namespace domain;

template<class Setup>
class RSVD_DT
{

    public: //member types
    using simulation_type = typename Setup::simulation_t;
    using domain_type = typename simulation_type::domain_type;
    using datablock_type = typename domain_type::datablock_t;
    using tree_t = typename domain_type::tree_t;
    using octant_t = typename tree_t::octant_type;
    using MASK_TYPE = typename octant_t::MASK_TYPE;
    using block_type = typename datablock_type::block_descriptor_type;
    using real_coordinate_type = typename domain_type::real_coordinate_type;
    using coordinate_type = typename domain_type::coordinate_type;
    using poisson_solver_t = typename Setup::poisson_solver_t;
    using linsys_solver_t = typename Setup::linsys_solver_t;
    using time_integration_t = typename Setup::time_integration_t;

    using ib_t = typename domain_type::ib_t;
    using force_type = typename ib_t::force_type;

    //FMM
    using Fmm_t = typename Setup::Fmm_t;

    using test_type = typename Setup::test_type;
    using idx_u_type = typename Setup::idx_u_type;
    using idx_u_g_type = typename Setup::idx_u_g_type;
    using u_hat_re_type = typename Setup::u_hat_re_type;
    using u_hat_im_type = typename Setup::u_hat_im_type;
    using f_hat_re_type = typename Setup::f_hat_re_type;
    using f_hat_im_type = typename Setup::f_hat_im_type;
    using u_base_type = typename Setup::u_base_type;
    using edge_aux_type= typename Setup::edge_aux_type;
    using u_type = typename Setup::u_type;
    using stream_f_type = typename Setup::stream_f_type;
    using cell_aux_type = typename Setup::cell_aux_type;
    using cell_aux_tmp_type = typename Setup::cell_aux_tmp_type;
    using face_aux_tmp_type = typename Setup::face_aux_tmp_type;
    using face_aux_type = typename Setup::face_aux_type;
    static constexpr int lBuffer = 1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer = 1; ///< Lower left buffer for interpolation
    static constexpr int Nf = Setup::Nf; ///< Number of frequencies

    static constexpr int Dim = Setup::Dim; ///< Number of dimensions
    RSVD_DT(simulation_type* _simulation)
    : simulation_(_simulation)
    , domain_(_simulation->domain_.get())
    , ifherk_(_simulation)
    ,psolver_(_simulation)
    {
        // u_base already set before this constructor
        // ifherk initialized. ifherk contains psolver and linsys_solver
        dx_base_ = domain_->dx_base();
        max_ref_level_ =
            _simulation->dictionary()->template get<float_type>("nLevels");
        cfl_ =
            _simulation->dictionary()->template get_or<float_type>("cfl", 0.2);
        dt_base_ =
            _simulation->dictionary()->template get_or<float_type>("dt", -1.0);

        Re_ = _simulation->dictionary()->template get<float_type>("Re");

        n_per_N_=_simulation->dictionary()->template get<int>("n_per_N");
        seed_=_simulation->dictionary()->template get_or<int>("seed",2025);
        forcing_flow_name_ = _simulation->dictionary()->template get_or<std::string>("forcing_flow_name", "null");
        N_period_ = _simulation->dictionary()->template get_or<int>("N_period", 4);
        if (dt_base_ < 0) dt_base_ = dx_base_ * cfl_;

        nLevelRefinement_ = domain_->tree()->depth()-domain_->tree()->base_level()-1;
        dt_               = dt_base_/math::pow2(nLevelRefinement_);

        pcout << "dt_base_: " << dt_base_ << std::endl;
        // get frequency vector
        f_vec_.resize(Nf);
        NT=(Nf-1)*2; //number of snapshots per period
        int nt=NT*n_per_N_; //number of timesteps per period

        float_type df=1.0/(dt_*nt);
        for (int i=0; i<Nf; i++)
        {
            f_vec_[i]=i*df;
        }
        pcout << "dt_: " << dt_ << std::endl;
        //print frequencies
        pcout<< "frequencies: " << std::endl;
        for (int i=0; i<Nf; i++)
        {
            pcout<< f_vec_[i] << " ";
        }



    }

    float_type run()
    {
        boost::mpi::communicator world;
        std::cout<<"here"<<std::endl;
        // initialize forcing field
        // for each f, set a voritcity vector to random numbers with bc of 0 and then solve vector potential equation to generate forcing field
        // init_w_random<edge_aux_type, u_type>();
        if(forcing_flow_name_=="null")
        {
            init_forcing_vectors();
            pcout << "init forcing vectors" << std::endl;
            clean_idx<f_hat_re_type>(0);
            clean_idx<f_hat_im_type>(0);
            ifherk_.template normalize_field_complex_by_freq<f_hat_re_type, f_hat_im_type>();
            std::vector<float_type> norm_real = ifherk_.template compute_norm_by_freq<f_hat_re_type>();
            auto norm_imag = ifherk_.template compute_norm_by_freq<f_hat_im_type>();
            std::vector<float_type> norm_total(Nf,0.0);

            for(auto ff=0; ff<Nf;ff++)
            {
                norm_total[ff]=std::sqrt(norm_real[ff]*norm_real[ff]+norm_imag[ff]*norm_imag[ff]);
            }
            for(auto ff=0; ff<Nf;ff++)
            {
                pcout<< "norm total " << ff << ": " << norm_total[ff] << std::endl;
            }
        }
        else if (forcing_flow_name_=="homo")
        {
            init_w_random<edge_aux_type, u_type>(0);
            pcout << "init u(0)" << std::endl;
            clean<f_hat_re_type>();
            clean<f_hat_im_type>();
            ifherk_.template normalize_field<u_type>();
            auto norm = ifherk_.template compute_norm<u_type>();
            pcout << "norm u(0): " << norm << std::endl;
        }
        else
        {
            simulation_->template read_h5<f_hat_re_type>(forcing_flow_name_, "u_hat_re");
            simulation_->template read_h5<f_hat_im_type>(forcing_flow_name_, "u_hat_im");
            pcout << "read forcing vectors" << std::endl;
            up_and_down_vec<f_hat_re_type>();
            up_and_down_vec<f_hat_im_type>();
            clean_idx<f_hat_re_type>(0);
            clean_idx<f_hat_im_type>(0);
            ifherk_.template normalize_field_complex_by_freq<f_hat_re_type, f_hat_im_type>();
            std::vector<float_type> norm_real = ifherk_.template compute_norm_by_freq<f_hat_re_type>();
            auto norm_imag = ifherk_.template compute_norm_by_freq<f_hat_im_type>();
            std::vector<float_type> norm_total(Nf,0.0);

            for(auto ff=0; ff<Nf;ff++)
            {
                norm_total[ff]=std::sqrt(norm_real[ff]*norm_real[ff]+norm_imag[ff]*norm_imag[ff]);
            }
            for(auto ff=0; ff<Nf;ff++)
            {
                pcout<< "norm total " << ff << ": " << norm_total[ff] << std::endl;
            }
        }

        
        //assumming no restart for now
        ifherk_.init_single_step(); //resets time and adapt count/ refresh time to 0
        for (std::size_t i=0; i<N_period_; i++)
        {
            ifherk_.time_march_rsvd_dt(); //march 1 period
            simulation_->write(p_fname(i)); // save data
            clean<u_hat_re_type>(); // clean fft fields
            clean<u_hat_im_type>();
        }


        

        // clean<cell_aux_type>();
        // // clean<cell_aux_type>();
        // if (domain_->is_client())
        // {
        //     divergence<u_type,cell_aux_type>();
        //     up_and_down<cell_aux_type>();
        // }
        // auto error0=ifherk_.template compute_norm<u_type>();
        // auto error1=ifherk_.template compute_norm<cell_aux_type>();
        // std::cout<<"error0: "<<error0<<std::endl;
        // std::cout<<"error1: "<<error1<<std::endl;
        return 0.0;

    }
    std::string p_fname(int i){
        return "flow_T_" + std::to_string(i);
    }

    void init_forcing_vectors()
    {
        boost::mpi::communicator world;
        if (!domain_->is_client()) return;
        clean<f_hat_re_type>();
        clean<f_hat_im_type>();
        for (std::size_t i = 0; i < Nf; ++i)
        {
            clean<edge_aux_type>();
            clean<face_aux_tmp_type>();
            init_w_random<edge_aux_type, face_aux_tmp_type>(i);
            up_and_down<face_aux_tmp_type>();
            copy_to_vec<face_aux_tmp_type, f_hat_re_type>(i);
            clean<edge_aux_type>();
            clean<face_aux_tmp_type>();
            init_w_random<edge_aux_type, face_aux_tmp_type>(2*i); 
            up_and_down<face_aux_tmp_type>();
            copy_to_vec<face_aux_tmp_type, f_hat_im_type>(i);

        }
    }
    template<typename face>
    void init_u_random(int offset=0)
    {
        boost::mpi::communicator world;
        if (!domain_->is_client()) return;
        clean<face>();
        auto client = domain_->decomposition().client();
        std::mt19937 gen(seed_ + world.rank()*Nf*2+offset);
        std::normal_distribution<float_type> dist(0.0, std::sqrt(1.0 / 2.0));
        auto center =
            (domain_->bounding_box().max() - domain_->bounding_box().min() + 1) / 2.0 + domain_->bounding_box().min();
        for (int l = domain_->tree()->base_level(); l < domain_->tree()->depth(); ++l)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned()) continue;
                if (!it->is_leaf()) continue;
                // if (it->is_correction()) continue; // vorticity of boundary points are 0 //for u we want velocity at boundary 
                const auto dx_base = domain_->dx_base();
                auto       dx_level = dx_base / std::pow(2, it->refinement_level());
                auto       scaling = std::pow(2, it->refinement_level());
                for (auto& n : it->data())
                {
                    for (std::size_t field_idx = 0; field_idx < face::nFields(); ++field_idx)
                    {
                        n(face::tag(), field_idx) = dist(gen);
                    }
                }
            }
            // client->template buffer_exchange<face>(l);
        }
        up_and_down<face>();
        clean<edge_aux_type>();
        clean<stream_f_type>();

        for (int l = domain_->tree()->base_level(); l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<face>(l);
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned()|| it->is_correction()) continue;

                const auto dx_level = dx_base_ / math::pow2(it->refinement_level());
                domain::Operator::curl<face,edge_aux_type>(it->data(), dx_level);

            }
            
        }
        clean<face>();
        psolver_.template apply_lgf<edge_aux_type, stream_f_type>();
        for (int l = domain_->tree()->base_level(); l < domain_->tree()->depth(); ++l)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned()) continue;

                const auto dx_level = dx_base_ / math::pow2(it->refinement_level());
                domain::Operator::curl_transpose<stream_f_type, face>(it->data(), dx_level, -1.0);
            }
            client->template buffer_exchange<face>(l);
        }

    }
    template<typename edge, typename face>
    void init_w_random(int offset = 0)
    {
        boost::mpi::communicator world;
        if (!domain_->is_client()) return;
        clean<edge>();
        clean<face>();
        auto                                 client = domain_->decomposition().client();
        std::mt19937                         gen(seed_ + world.rank()*Nf*2+offset);
        std::normal_distribution<float_type> dist(0.0, std::sqrt(1.0 / 2.0));
        auto                                 center =
            (domain_->bounding_box().max() - domain_->bounding_box().min() + 1) / 2.0 + domain_->bounding_box().min();
        for (int l = domain_->tree()->base_level(); l < domain_->tree()->depth(); ++l)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned()) continue;
                if (!it->is_leaf()) continue;
                if (it->is_correction()) continue; // vorticity of boundary points are 0
                const auto dx_base = domain_->dx_base();
                auto       dx_level = dx_base / std::pow(2, it->refinement_level());
                auto       scaling = std::pow(2, it->refinement_level());
                for (auto& n : it->data())
                {
                    for (std::size_t field_idx = 0; field_idx < edge::nFields(); ++field_idx)
                    {
                        n(edge::tag(), field_idx) = dist(gen)*scaling;
                    }
                }
            }
            // client->template buffer_exchange<edge>(l);
        }
        up_and_down<edge>();
        clean<face>();
        clean<stream_f_type>();
        psolver_.template apply_lgf<edge, stream_f_type>();
        for (int l = domain_->tree()->base_level(); l < domain_->tree()->depth(); ++l)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned()) continue;

                const auto dx_level = dx_base_ / math::pow2(it->refinement_level());
                domain::Operator::curl_transpose<stream_f_type, face>(it->data(), dx_level, -1.0);
            }
            client->template buffer_exchange<face>(l);
        }
    }

    template<typename face>
    void init_pressure_proj(int offset=0)
    {
        boost::mpi::communicator world;
        if (!domain_->is_client()) return;
        clean<face>();
        auto client = domain_->decomposition().client();
        std::mt19937 gen(seed_ + world.rank()*Nf*2+offset);
        std::normal_distribution<float_type> dist(0.0, std::sqrt(1.0 / 2.0));
        auto center =
            (domain_->bounding_box().max() - domain_->bounding_box().min() + 1) / 2.0 + domain_->bounding_box().min();
        for (int l = domain_->tree()->base_level(); l < domain_->tree()->depth(); ++l)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned()) continue;
                if (!it->is_leaf()) continue;
                //if (it->is_correction()) continue; // vorticity of boundary points are 0 //for u we want velocity at boundary 
                const auto dx_base = domain_->dx_base();
                auto       dx_level = dx_base / std::pow(2, it->refinement_level());
                auto       scaling = std::pow(2, it->refinement_level());
                for (auto& n : it->data())
                {
                    for (std::size_t field_idx = 0; field_idx < face::nFields(); ++field_idx)
                    {
                        n(face::tag(), field_idx) = dist(gen);
                    }
                }
            }
            client->template buffer_exchange<face>(l);
        }
        // up_and_down<face>();
        clean<face_aux_type>();
        copy<face, face_aux_type>();
        up_and_down<face_aux_type>();
        domain_->client_communicator().barrier();
        clean<cell_aux_type>();
        clean<cell_aux_tmp_type>();
        clean<face_aux_tmp_type>();
        divergence<face_aux_type, cell_aux_type>();
        domain_->client_communicator().barrier();
        psolver_.template apply_lgf<cell_aux_type, cell_aux_tmp_type>();
        domain_->client_communicator().barrier();
        gradient<cell_aux_tmp_type,face_aux_tmp_type>();
        domain_->client_communicator().barrier();
        add<face_aux_tmp_type, face>(-1.0);

    }
    template<class Field>
    void up_and_down()
    {
        //clean non leafs
        clean<Field>(true);
        this->up<Field>();
        this->down_to_correction<Field>();
    }
    

    template<class Field>
    void up(bool leaf_boundary_only = false)
    {
        //Coarsification:
        for (std::size_t _field_idx = 0; _field_idx < Field::nFields(); ++_field_idx)
            psolver_.template source_coarsify<Field, Field>(_field_idx, _field_idx, Field::mesh_type(), false, false,
                false, leaf_boundary_only);
    }
    template<class Field>
    void down_to_correction()
    {
        // Interpolate to correction buffer
        for (std::size_t _field_idx = 0; _field_idx < Field::nFields(); ++_field_idx)
            psolver_.template intrp_to_correction_buffer<Field, Field>(_field_idx, _field_idx, Field::mesh_type(), true,
                false);
    }
    template<class Source, class Target>
    void divergence() noexcept
    {
        auto client = domain_->decomposition().client();

        up_and_down<Source>();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::divergence<Source, Target>(
                    it->data(), dx_level);
            }

            // client->template buffer_exchange<Target>(l);
            // //client->template buffer_exchange<Target>(l);
            clean_leaf_correction_boundary<Target>(l, true, 2);
            //clean_leaf_correction_boundary<Target>(l, false,4+stage_idx_);
           
        }
    }
    template<class Source, class Target>
    void gradient(float_type _scale = 1.0) noexcept
    {
        //up_and_down<Source>();
        domain::Operator::domainClean<Target>(domain_);

        // up_and_down<Source>();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            auto client = domain_->decomposition().client();
            client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::gradient<Source, Target>(it->data(),
                    dx_level);
                for (std::size_t field_idx = 0; field_idx < Target::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(Target::tag(), field_idx).linalg_data();

                    lin_data *= _scale;
                }
            }
            client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true, 2);
        }

        //clean_leaf_correction_boundary<Target>(domain_->tree()->base_level(), true,2);
    }
    template <typename F>
    void clean_leaf_correction_boundary(int l, bool leaf_only_boundary=false, int clean_width=1) noexcept
    {
        for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
        {
            if (!it->locally_owned())
            {
                if (!it->has_data() || !it->data().is_allocated()) continue;
                for (std::size_t field_idx = 0; field_idx < F::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(F::tag(), field_idx).linalg_data();
                    std::fill(lin_data.begin(), lin_data.end(), 0.0);
                }
            }
        }

        for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->has_data() || !it->data().is_allocated()) continue;

            if (leaf_only_boundary && (it->is_correction() || it->is_old_correction() ))
            {
                for (std::size_t field_idx = 0; field_idx < F::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(F::tag(), field_idx).linalg_data();
                    std::fill(lin_data.begin(), lin_data.end(), 0.0);
                }
            }
        }


        //---------------
        if (l==domain_->tree()->base_level())
        for (auto it  = domain_->begin(l);
                it != domain_->end(l); ++it)
        {
            if(!it->locally_owned()) continue;
            if(!it->has_data() || !it->data().is_allocated()) continue;
            //std::cout<<it->key()<<std::endl;

            for(std::size_t i=0;i< it->num_neighbors();++i)
            {
                auto it2=it->neighbor(i);
                if ((!it2 || !it2->has_data()) || (leaf_only_boundary && (it2->is_correction() || it2->is_old_correction() )))
                {
                    for (std::size_t field_idx=0; field_idx<F::nFields(); ++field_idx)
                    {
                        domain::Operator::smooth2zero<F>( it->data(), i);
                    }
                }
            }
        }
    }

    template<typename F>
    void clean(bool non_leaf_only = false, int clean_width = 1) noexcept
    {
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->has_data()) continue;
            if (!it->data().is_allocated()) continue;

            for (std::size_t field_idx = 0; field_idx < F::nFields(); ++field_idx)
            {
                auto& lin_data = it->data_r(F::tag(), field_idx).linalg_data();

                if (non_leaf_only && it->is_leaf() && it->locally_owned())
                {
                    int N = it->data().descriptor().extent()[0];
                    if (domain_->dimension() == 3)
                    {
                        view(lin_data, xt::all(), xt::all(), xt::range(0, clean_width)) *= 0.0;
                        view(lin_data, xt::all(), xt::range(0, clean_width), xt::all()) *= 0.0;
                        view(lin_data, xt::range(0, clean_width), xt::all(), xt::all()) *= 0.0;
                        view(lin_data, xt::range(N + 2 - clean_width, N + 3), xt::all(), xt::all()) *= 0.0;
                        view(lin_data, xt::all(), xt::range(N + 2 - clean_width, N + 3), xt::all()) *= 0.0;
                        view(lin_data, xt::all(), xt::all(), xt::range(N + 2 - clean_width, N + 3)) *= 0.0;
                    }
                    else
                    {
                        view(lin_data, xt::all(), xt::range(0, clean_width)) *= 0.0;
                        view(lin_data, xt::range(0, clean_width), xt::all()) *= 0.0;
                        view(lin_data, xt::range(N + 2 - clean_width, N + 3), xt::all()) *= 0.0;
                        view(lin_data, xt::all(), xt::range(N + 2 - clean_width, N + 3)) *= 0.0;
                    }
                }
                else
                {
                    //TODO whether to clean base_level correction?
                    std::fill(lin_data.begin(), lin_data.end(), 0.0);
                }
            }
        }
    }

    template<typename Face_from, typename Face_to_vec>
    void copy_to_vec(int idx_f,float_type scale =1.0)
    {
        //if field is frequnecy than x-compoenents are (1,NF) y is (NF+1,2NF)
        static_assert(Face_to_vec::nFields() == Nf * Face_from::nFields(),
              "Face_to_vec must hold Nf copies of Face_from fields");
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if(!it->locally_owned()||!it->has_data()) continue;
            for(std::size_t field_idx=0;field_idx<Face_from::nFields();++field_idx)
            {
               for (auto &n:it->data())
               {
                   n(Face_to_vec::tag(), field_idx*Nf+idx_f)=n(Face_from::tag(),field_idx)*scale;
               }
            }
        }
    }
    template<typename Face_from_vec, typename Face_to>
    void copy_from_vec(int idx_f,float_type scale =1.0)
    {
        //if field is frequnecy than x-compoenents are (1,NF) y is (NF+1,2NF)
        static_assert(Face_to::nFields() *Nf ==  Face_from_vec::nFields(),
              "Face_to_vec must hold Nf copies of Face_from fields");
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if(!it->locally_owned()||!it->has_data()) continue;
            for(std::size_t field_idx=0;field_idx<Face_to::nFields();++field_idx)
            {
               for (auto &n:it->data())
               {
                   n(Face_to::tag(), field_idx)=n(Face_from_vec::tag(),field_idx*Nf+idx_f)*scale;
               }
            }
        }
    }

    template<typename face_vec>
    void up_and_down_vec()
    {
        clean<face_aux_tmp_type>();
        for(std::size_t i=0; i<Nf;++i)
        {
            copy_from_vec<face_vec, face_aux_tmp_type>(i);
            up_and_down<face_aux_tmp_type>();
            clean_idx<face_vec>(i);
            copy_to_vec<face_aux_tmp_type, face_vec>(i);
        }
    }
    template<typename face>
    void clean_idx(int idx_f=0)
    {
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            for (std::size_t dd = 0; dd<domain_->dimension(); ++dd)
            {
                for(auto& n:it->data())
                {
                    n(face::tag(), dd*Nf+idx_f) = 0.0;
                }
            }
        }
    }

    template<typename From, typename To>
    void add(float_type scale = 1.0) noexcept
    {
        static_assert(From::nFields() == To::nFields(),
            "number of fields doesn't match when add");
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            for (std::size_t field_idx = 0; field_idx < From::nFields();
                 ++field_idx)
            {
                it->data_r(To::tag(), field_idx)
                    .linalg()
                    .get()
                    ->cube_noalias_view() +=
                    it->data_r(From::tag(), field_idx).linalg_data() * scale;
            }
        }
    }
    template<typename From, typename To>
    void copy(float_type scale = 1.0) noexcept
    {
        static_assert(From::nFields() == To::nFields(),
            "number of fields doesn't match when copy");

        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            for (std::size_t field_idx = 0; field_idx < From::nFields();
                 ++field_idx)
            {
                for (auto& n:it->data().node_field())
                    n(To::tag(), field_idx) = n(From::tag(), field_idx) * scale;
            }
        }
    }
    float_type run_mat_test()
    {
        boost::mpi::communicator world;
        std::cout << "RSVD_DT::run()" << std::endl;
        this->init_idx<idx_u_type, idx_u_g_type>();
        
        world.barrier();
        PetscMPIInt rank;
        rank= world.rank();
        PetscInt m_local, M;
        m_local=max_local_idx; // since 1 based 
        if(rank == 0)
            m_local = 0;
        
        boost::mpi::all_reduce(world, m_local, M, std::plus<int>());
        
        Vec x, b;
        // Mat A;

        PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
        PetscCall(VecSetSizes(x, m_local, M));
        PetscCall(VecSetFromOptions(x));
        PetscCall(VecSet(x,2.0));
        PetscCall(VecAssemblyBegin(x));
        PetscCall(VecAssemblyEnd(x));
        // vec2grid<idx_u_type, u_hat_re_type, u_hat_im_type>(x); //vec_check is x
        grid2vec<idx_u_type, u_type, u_base_type>(x);
        PetscCall(VecConjugate(x));
        vec2grid<idx_u_type, u_hat_re_type, u_hat_im_type>(x); //vec_check is x
        //get forcing from file
        //get baseflow from file 
        //clean u 
        //time step:
            // forcing from ifft of f_hat
            // u_hat from fft of u
    }

    template<class Field_idx, class Field_re, class Field_im>
    float_type vec2grid(Vec x)
    {
        //loop through points
        const PetscComplex* x_array;
        PetscCall(VecGetArrayRead(x, &x_array));
        int base_level = domain_->tree()->base_level();
        for (int l = base_level; l < domain_->tree()->depth(); l++)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                // if (!it->data().is_allocated()) continue;
                if (it->is_leaf() )
                {
                    for (std::size_t field_idx = 0; field_idx < Dim; ++field_idx)
                    {
                        for (auto& n : it->data())
                        {
                            int i_local = n(Field_idx::tag(), field_idx) - 1; //since used to be 1 based
                            if (i_local < 0) continue;
                            // int i_global =
                            //     (rank - 1) * m_local + i_local - 1;
                            PetscComplex value;
                            // PetscCall(MatGetValuesLocal(B, 1, &i_local, 1, &j, &value));
                            value = x_array[i_local];
                            n(Field_re::tag(), field_idx) = PetscRealPart(value);
                            n(Field_im::tag(), field_idx) = PetscImaginaryPart(value);
                            // n(Field_re::tag(), field_idx * N_tv + idx) = PetscRealPart(value) / w_1_2;
                            // n(Field_im::tag(), field_idx * N_tv + idx) = PetscImaginaryPart(value) / w_1_2;
                        }
                    }
                }
            }
        }
        VecRestoreArrayRead(x, &x_array);
        return 0.0;
    }

    template<class Field_idx, class Field_re, class Field_im>
    float_type grid2vec(Vec x)
    {   
        PetscComplex* x_array;
        PetscCall(VecGetArray(x, &x_array));
        for (int i = 0; i < max_local_idx; ++i)
        {
            x_array[i] = 0.0;
        }
        if (domain_->is_client())
        {
            int base_level = domain_->tree()->base_level();
            for (int l = base_level; l < domain_->tree()->depth(); l++)
            {
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned() || !it->has_data()) continue;
                    // if (!it->data().is_allocated()) continue;
                    if (it->is_leaf())
                    {
                        for (std::size_t field_idx = 0; field_idx < Dim; ++field_idx)
                        {
                            for (auto& n : it->data())
                            {
                                int i_local = n(Field_idx::tag(), field_idx) - 1; //since used to be 1 based
                                if (i_local < 0) continue;
                                PetscScalar value_r= n(Field_re::tag(), field_idx);
                                PetscScalar value_i= n(Field_im::tag(), field_idx);
                                PetscComplex value= value_r + value_i*PETSC_i;
                                x_array[i_local]=value;

                            }
                        }
                    }
                }
            }
        }
    
        PetscCall(VecAssemblyBegin(x));
        PetscCall(VecAssemblyEnd(x));
        PetscCall(VecRestoreArray(x, &x_array));
        return 0.0;


    }

    template<typename F, typename F_g>
    void init_idx(bool assign_base_correction = true)
    {
        boost::mpi::communicator world;
        ifherk_.template clean<F>();
        ifherk_.template clean<F_g>();
        int local_count = 0; //local count of the number of indices
        max_local_idx = -1;
        int max_idx_from_prev_prc = -1;
        if (domain_->is_server()) return; // server has no data

        int base_level = domain_->tree()->base_level();
        for (int l = base_level; l < domain_->tree()->depth(); l++) //other levels found from up down
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf()) continue;
                if (it->is_correction() && !assign_base_correction) continue;
                for (std::size_t field_idx = 0; field_idx < F::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        local_count++;
                        n(F::tag(), field_idx) = local_count; //0  means not part of matrix so make 1 based
                    }
                }
            }
        }
        max_local_idx = local_count;
        domain_->client_communicator().barrier();
        boost::mpi::scan(domain_->client_communicator(), max_local_idx, max_idx_from_prev_prc, std::plus<float_type>());
        max_idx_from_prev_prc-=max_local_idx; //max idx from previous processor is the sum of all local counts from previous processors
                for (int i = 1; i < world.size();i++) {
            if (world.rank() == i) std::cout << "rank " << world.rank() <<" counter is " << local_count << " counter + max idx is " << (local_count + max_idx_from_prev_prc) << " max idx from prev prc " << max_idx_from_prev_prc << std::endl;
            domain_->client_communicator().barrier();
        }

        int offset_to_global= max_idx_from_prev_prc; //if max from last index is 1 (1 based, so 1 data point) then offset is 1
        int min_local_g_idx=max_idx_from_prev_prc+1; //min global index on current processor is max idx from previous + 1 since next point
        int max_local_g_idx=offset_to_global+max_local_idx; //max global index on current processor is offset + local count
        for (int l = base_level; l < domain_->tree()->depth(); l++)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf()) continue;
                if (it->is_correction() && !assign_base_correction) continue;
                for (std::size_t field_idx = 0; field_idx < F::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        if(n(F::tag(), field_idx) > 0)
                        {
                            n(F_g::tag(), field_idx) =n(F::tag(), field_idx)+ offset_to_global;
                        }
                    }
                }
            }
        }
        if(local_count != max_local_g_idx-min_local_g_idx+1)
        {
            std::cout << "local count is " << local_count << " max local g idx is " << max_local_g_idx << " min local g idx is " << min_local_g_idx << std::endl;
            std::cout << "local count is not equal to max local g idx - min local g idx + 1" << std::endl;
        }

        return;
    }

  
    

    private:
        simulation_type* simulation_;
        domain_type*     domain_; ///< domain

        time_integration_t ifherk_;
        poisson_solver_t psolver_;
        std::string forcing_flow_name_;
        float_type dt_base_, dt_, dx_base_;
        float_type Re_;
        float_type cfl_max_, cfl_;
        int max_ref_level_=0;
        int max_local_idx = -1;
        int min_local_idx = -1;
        int n_per_N_;
        int seed_;
        int nLevelRefinement_;
        int NT;
        int N_period_;
        std::vector<float_type> f_vec_;
        parallel_ostream::ParallelOstream pcout =
        parallel_ostream::ParallelOstream(1);

};
}
}

#endif // IBLGF_INCLUDED_RSVDDT_HPP