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

#ifndef IBLGF_INCLUDED_SOLVER_POISSON_HPP
#define IBLGF_INCLUDED_SOLVER_POISSON_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/IO/parallel_ostream.hpp>

#include <iblgf/interpolation/interpolation.hpp>
#include <iblgf/operators/operators.hpp>

#include <iblgf/lgf/lgf_gl.hpp>
#include <iblgf/lgf/lgf_ge.hpp>
#include <iblgf/lgf/helmholtz.hpp>

#include <iblgf/linalg/linalg.hpp>

namespace iblgf
{
namespace solver
{
using namespace domain;

/** @brief Poisson solver using lattice Green's functions. Convolutions
 *         are computed using FMM and near field interaction are computed
 *         using blockwise fft-convolutions.
 */
template<class Setup>
class PoissonSolver
{
  public: //member types
    static constexpr std::size_t Dim = Setup::Dim;

    using simulation_type = typename Setup::simulation_t;
    using domain_type = typename simulation_type::domain_type;
    using interpolation_type = typename interpolation::cell_center_nli<domain_type>;
    using datablock_type = typename domain_type::datablock_t;
    using tree_t = typename domain_type::tree_t;
    using octant_t = typename tree_t::octant_type;
    using block_type = typename datablock_type::block_descriptor_type;
    using real_coordinate_type = typename domain_type::real_coordinate_type;
    using coordinate_type = typename domain_type::coordinate_type;
    using MASK_TYPE = typename octant_t::MASK_TYPE;

    //Fields
    using source_tmp_type = typename Setup::source_tmp_type;
    using target_tmp_type = typename Setup::target_tmp_type;
    using correction_tmp_type = typename Setup::correction_tmp_type;

    static constexpr auto source_tmp = Setup::source_tmp;
    static constexpr auto target_tmp = Setup::target_tmp;
    static constexpr auto correction_tmp = Setup::correction_tmp;

    //FMM
    using Fmm_t = typename Setup::Fmm_t;
    using lgf_lap_t = typename lgf::LGF_GL<Dim>;
    using lgf_if_t = typename lgf::LGF_GE<Dim>;
    using helm_t   = typename lgf::Helmholtz<Dim>;

    static constexpr int lBuffer = 1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer = 1; ///< Lower left buffer for interpolation

    PoissonSolver(simulation_type* _simulation, int _N = 0)
    :
    domain_(_simulation->domain_.get()),
    fmm_(domain_,domain_->block_extent()[0]+lBuffer+rBuffer),
    c_cntr_nli_(domain_->block_extent()[0]+lBuffer+rBuffer, _simulation->intrp_order()),
    N_fourier_modes(_N)
    {
        //initializing vector of helmholtz solvers
        //_N is the number of Fourier modes computed at finest level, but c is the wave number at the base level. 
        //Need to change the computation from 2pi*n/N to 2*pi*n*2^(levels)/N
        const int l_max = domain_->tree()->depth();
        const int l_min = domain_->tree()->base_level();
        const int nLevels = l_max - l_min;

        const float_type dx_base = domain_->dx_base();

        c_z_ = _simulation->dictionary()->template get_or<float_type>("L_z", 1.0);

        for (int i = 0; i < N_fourier_modes; i++) {
            float_type c = (static_cast<float_type>(i)+1.0) * 2.0 * M_PI * dx_base/c_z_;
            lgf_helm_vec.emplace_back(helm_t(c));
        }

        /*for (int i = 0; i < N_fourier_modes; i++) {
            float_type c = (static_cast<float_type>(i)+1.0)/static_cast<float_type>(N_fourier_modes + 1) * 2.0 * M_PI * std::pow(2.0, nLevels - 1);
            lgf_helm_vec.emplace_back(helm_t(c));
        }*/


        //setting if only advect a subset of modes, the total simulated modes are additional_modes + 1
        additional_modes = _simulation->dictionary()->template get_or<int>("add_modes", N_fourier_modes);
        //std::cout << "Number of Fourier modes are " << N_fourier_modes <<std::endl;
        if (additional_modes > N_fourier_modes) throw std::runtime_error("Additional modes cannot be higher than the number of Fourier modes");
    }

  public:
    template<class Source, class Target>
    void apply_lgf(int fmm_type = MASK_TYPE::AMR2AMR)
    {
        for (std::size_t entry = 0; entry < Source::nFields(); ++entry)
            this->apply_lgf<Source, Target>(&lgf_lap_, entry, fmm_type);
    }

    template<class Source, class Target>
    void apply_helm(int n, int fmm_type = MASK_TYPE::AMR2AMR)
    {
        if (n >= N_fourier_modes) throw std::runtime_error("Fourier mode number too high");
        for (std::size_t entry = 0; entry < Source::nFields(); ++entry)
            this->apply_lgf<Source, Target>(&lgf_helm_vec[n], entry, fmm_type);
    }

    /*template<class Source, class Target>
    void apply_lgf_and_helm(int N_modes, int NComp = 1,
        int fmm_type = MASK_TYPE::AMR2AMR)
    {
        if (N_modes != (N_fourier_modes + 1))
            throw std::runtime_error(
                "Fourier modes do not match in helmholtz solver");
        if (Source::nFields() != N_modes * 2 * NComp)
            throw std::runtime_error(
                "Fourier modes number elements do not match in helmholtz solver");
        for (int i = 0; i < NComp; i++)
        {
            int add_num = i * N_modes*2;
            for (std::size_t entry = 0; entry < 2; ++entry)
                this->apply_lgf<Source, Target>(&lgf_lap_, (add_num + entry), fmm_type);
            for (std::size_t idx = 0; idx < N_fourier_modes; ++idx)
            {
                //int entry = idx*2 + NComp*2;
                for (std::size_t addentry = 0; addentry < 2; addentry++)
                {
                    int entry = addentry + idx * 2 + 2 + add_num;
                    this->apply_lgf<Source, Target>(&lgf_helm_vec[idx], entry,
                        fmm_type);
                }
            }
        }
    }*/

    template<class Source, class Target>
    void apply_lgf_and_helm(int N_modes, int NComp = 1,
        int fmm_type = MASK_TYPE::AMR2AMR)
    {
        if (N_modes != (N_fourier_modes + 1))
            throw std::runtime_error(
                "Fourier modes do not match in helmholtz solver");
        if (Source::nFields() != N_modes * 2 * NComp)
            throw std::runtime_error(
                "Fourier modes number elements do not match in helmholtz solver");
        for (int i = 0; i < NComp; i++)
        {
            int add_num = i * N_modes*2;
            for (std::size_t entry = 0; entry < 2; ++entry) {
                this->apply_lgf<Source, Target>(&lgf_lap_, (add_num + entry), fmm_type);
                domain_->client_communicator().barrier();
            }
            for (std::size_t idx = 0; idx < additional_modes; ++idx)
            {
                //int entry = idx*2 + NComp*2;
                for (std::size_t addentry = 0; addentry < 2; addentry++)
                {
                    int entry = addentry + idx * 2 + 2 + add_num;
                    this->apply_lgf<Source, Target>(&lgf_helm_vec[idx], entry,
                        fmm_type);
                    domain_->client_communicator().barrier();
                }
            }
        }
    }



    /*template<class Source, class Target>
    void apply_helm_if(float_type _alpha_base, int N_modes, float_type L_z, int NComp = 3, 
        int fmm_type = MASK_TYPE::AMR2AMR)
    {
        lgf_if_.alpha_base_level() = _alpha_base;
        if (N_modes != (N_fourier_modes + 1))
            throw std::runtime_error(
                "Fourier modes do not match in helmholtz solver");
        if (Source::nFields() != N_modes * 2 * NComp)
            throw std::runtime_error(
                "Fourier modes number elements do not match in helmholtz solver");
        for (int i = 0; i < NComp; i++)
        {
            int add_num = i * N_modes*2;
            for (std::size_t entry = 0; entry < 2; ++entry) {
                this->apply_if<Source, Target>(&lgf_if_, (add_num + entry), fmm_type);
            }
                //this->apply_lgf<Source, Target>(&lgf_if_, (add_num + entry), fmm_type);
            for (std::size_t idx = 0; idx < N_fourier_modes; ++idx)
            {
                //int entry = idx*2 + NComp*2;
                float_type omega = static_cast<float_type>(idx + 1) * 2.0 * M_PI / L_z;
                for (std::size_t addentry = 0; addentry < 2; addentry++)
                {
                    int entry = addentry + idx * 2 + 2 + add_num;
                    this->apply_if_helm<Source, Target>(&lgf_if_, omega, entry,
                        fmm_type);
                }
            }
        }
    }*/

    template<class Source, class Target>
    void apply_helm_if(float_type _alpha_base, int N_modes, float_type L_z, int NComp = 3, 
        int fmm_type = MASK_TYPE::AMR2AMR)
    {
        lgf_if_.alpha_base_level() = _alpha_base;
        if (N_modes != (N_fourier_modes + 1))
            throw std::runtime_error(
                "Fourier modes do not match in helmholtz solver");
        if (Source::nFields() != N_modes * 2 * NComp)
            throw std::runtime_error(
                "Fourier modes number elements do not match in helmholtz solver");
        for (int i = 0; i < NComp; i++)
        {
            int add_num = i * N_modes*2;
            for (std::size_t entry = 0; entry < 2; ++entry) {
                this->apply_if<Source, Target>(&lgf_if_, (add_num + entry), fmm_type);
                domain_->client_communicator().barrier();
            }
                //this->apply_lgf<Source, Target>(&lgf_if_, (add_num + entry), fmm_type);
            for (std::size_t idx = 0; idx < additional_modes; ++idx)
            {
                //int entry = idx*2 + NComp*2;
                float_type omega = static_cast<float_type>(idx + 1) * 2.0 * M_PI / L_z;
                for (std::size_t addentry = 0; addentry < 2; addentry++)
                {
                    int entry = addentry + idx * 2 + 2 + add_num;
                    this->apply_if_helm<Source, Target>(&lgf_if_, omega, entry,
                        fmm_type);
                    domain_->client_communicator().barrier();
                }
            }
        }
    }

    template<class Source, class Target, class Kernel>
    void apply_helm(int n, Kernel* _kernel, int fmm_type = MASK_TYPE::AMR2AMR)
    {
        if (n >= N_fourier_modes) throw std::runtime_error("Fourier mode number too high");
        for (std::size_t entry = 0; entry < Source::nFields(); ++entry)
            this->apply_lgf<Source, Target>(_kernel, entry, fmm_type);
    }



    template<class Source, class Target>
    void apply_lgf_IF(float_type _alpha_base, int fmm_type = MASK_TYPE::AMR2AMR)
    {
        lgf_if_.alpha_base_level() = _alpha_base;
        for (std::size_t entry = 0; entry < Source::nFields(); ++entry)
            this->apply_if<Source, Target>(&lgf_if_, entry, fmm_type);
    }

    template<class Source, class Target, class Kernel>
    void apply_if(Kernel* _kernel, std::size_t _field_idx = 0, int fmm_type = MASK_TYPE::AMR2AMR)
    {
        auto client = domain_->decomposition().client();
        if (!client) return;

        // Cleaning
        clean_field<source_tmp_type>();
        clean_field<target_tmp_type>();

        // Copy source
        if (fmm_type == MASK_TYPE::AMR2AMR)
            copy_leaf<Source, source_tmp_type>(_field_idx, 0, true);
        else if (fmm_type == MASK_TYPE::STREAM)
            copy_level<Source, source_tmp_type>(domain_->tree()->base_level(), _field_idx, 0, false);
        else if (fmm_type == MASK_TYPE::IB2xIB || fmm_type == MASK_TYPE::xIB2IB)
            copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, _field_idx, 0, false);
        else if (fmm_type == MASK_TYPE::IB2AMR)
            copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, _field_idx, 0, false);

        if (fmm_type != MASK_TYPE::STREAM && fmm_type != MASK_TYPE::IB2xIB && fmm_type != MASK_TYPE::xIB2IB)
        {
            // Coarsify
            source_coarsify<source_tmp_type, source_tmp_type>(_field_idx, 0, Source::mesh_type());

            // Interpolate to correction buffer
            intrp_to_correction_buffer<source_tmp_type, source_tmp_type>(
                    _field_idx, 0, Source::mesh_type());
        }

        const int l_max = (fmm_type != MASK_TYPE::STREAM) ?
                    domain_->tree()->depth() : domain_->tree()->base_level()+1;

        const int l_min = (fmm_type !=  MASK_TYPE::IB2xIB && fmm_type !=  MASK_TYPE::xIB2IB) ?
                    domain_->tree()->base_level() : domain_->tree()->depth()-1;

        for (int l = l_min; l < l_max; ++l)
        {

            for (auto it_s = domain_->begin(l); it_s != domain_->end(l); ++it_s)
                if (it_s->has_data() && !it_s->locally_owned())
                {
                    if (!it_s->data().is_allocated()) continue;
                    auto& cp2 = it_s->data_r(source_tmp);
                    std::fill(cp2.begin(), cp2.end(), 0.0);
                }

            _kernel->change_level(l - domain_->tree()->base_level());

            if (fmm_type == MASK_TYPE::AMR2AMR)
                fmm_.template apply<source_tmp_type, target_tmp_type>(
                        domain_, _kernel, l, false, 1.0, fmm_type);

            if (!subtract_non_leaf_)
                fmm_.template apply<source_tmp_type, target_tmp_type>(
                    domain_, _kernel, l, true, 1.0, fmm_type);

            copy_level<target_tmp_type, Target>(l, 0, _field_idx, true);
        }
    }


    template<class Source, class Target, class Kernel>
    void apply_if_helm(Kernel* _kernel, float_type omega, std::size_t _field_idx = 0, int fmm_type = MASK_TYPE::AMR2AMR)
    {
        auto client = domain_->decomposition().client();
        if (!client) return;

        // Cleaning
        clean_field<source_tmp_type>();
        clean_field<target_tmp_type>();

        const float_type dx_base = domain_->dx_base();

        // Copy source
        if (fmm_type == MASK_TYPE::AMR2AMR)
            copy_leaf<Source, source_tmp_type>(_field_idx, 0, true);
        else if (fmm_type == MASK_TYPE::STREAM)
            copy_level<Source, source_tmp_type>(domain_->tree()->base_level(), _field_idx, 0, false);
        else if (fmm_type == MASK_TYPE::IB2xIB || fmm_type == MASK_TYPE::xIB2IB)
            copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, _field_idx, 0, false);
        else if (fmm_type == MASK_TYPE::IB2AMR)
            copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, _field_idx, 0, false);

        if (fmm_type != MASK_TYPE::STREAM && fmm_type != MASK_TYPE::IB2xIB && fmm_type != MASK_TYPE::xIB2IB)
        {
            // Coarsify
            source_coarsify<source_tmp_type, source_tmp_type>(_field_idx, 0, Source::mesh_type());

            // Interpolate to correction buffer
            intrp_to_correction_buffer<source_tmp_type, source_tmp_type>(
                    _field_idx, 0, Source::mesh_type());
        }

        const int l_max = (fmm_type != MASK_TYPE::STREAM) ?
                    domain_->tree()->depth() : domain_->tree()->base_level()+1;

        const int l_min = (fmm_type !=  MASK_TYPE::IB2xIB && fmm_type !=  MASK_TYPE::xIB2IB) ?
                    domain_->tree()->base_level() : domain_->tree()->depth()-1;

        for (int l = l_min; l < l_max; ++l)
        {
            

            for (auto it_s = domain_->begin(l); it_s != domain_->end(l); ++it_s)
                if (it_s->has_data() && !it_s->locally_owned())
                {
                    if (!it_s->data().is_allocated()) continue;
                    auto& cp2 = it_s->data_r(source_tmp);
                    std::fill(cp2.begin(), cp2.end(), 0.0);
                }

            _kernel->change_level(l - domain_->tree()->base_level());

            double dx = dx_base / std::pow(2, l - domain_->tree()->base_level());

            //compute the add scale
            float_type alpha_level = _kernel->alpha_;
            float_type helm_weights = std::exp(-omega*omega*alpha_level * dx * dx);

            if (fmm_type == MASK_TYPE::AMR2AMR)
                fmm_.template apply<source_tmp_type, target_tmp_type>(
                        domain_, _kernel, l, false, helm_weights, fmm_type);

            if (!subtract_non_leaf_)
                fmm_.template apply<source_tmp_type, target_tmp_type>(
                    domain_, _kernel, l, true, helm_weights, fmm_type);

            copy_level<target_tmp_type, Target>(l, 0, _field_idx, true);
        }
    }

    /** @brief Solve the poisson equation using lattice Green's functions on
     *         a block-refined mesh for a given Source and Target field.
     *
     *  @detail Lattice Green's functions are used for solving the poisson
     *  equation. FMM is used for the level convolution's and the near field
     *  convolutions are computed using FFT.
     *  Second order interpolation and coarsification operators are used
    *  to project the solutions to fine and coarse meshes, respectively.
    */
    template<class Source, class Target, class Kernel>
    void apply_lgf(Kernel* _kernel, const std::size_t _field_idx,
        const int fmm_type)
    {
        auto client = domain_->decomposition().client();
        if (!client) return;

        const float_type dx_base = domain_->dx_base();

        // Cleaning
        //clean_field<corr_lap_tmp>();
        clean_field<source_tmp_type>();
        clean_field<target_tmp_type>();
        clean_field<correction_tmp_type>();

        // Copy source
        if (fmm_type == MASK_TYPE::AMR2AMR)
            copy_leaf<Source, source_tmp_type>(_field_idx, 0, true);
        else if (fmm_type == MASK_TYPE::STREAM)
            copy_level<Source, source_tmp_type>(domain_->tree()->base_level(), _field_idx, 0, false);
        else if (fmm_type == MASK_TYPE::IB2xIB || fmm_type == MASK_TYPE::xIB2IB)
            copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, _field_idx, 0, false);
        else if (fmm_type == MASK_TYPE::IB2AMR)
            copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, _field_idx, 0, false);


#ifdef POISSON_TIMINGS
        timings_ = Timings();
        const int nLevels = domain_->nLevels();
        timings_.level.resize(nLevels);
        timings_.fmm_level.resize(nLevels);
        timings_.fmm_level_nl.resize(nLevels);

        auto t0_all = clock_type::now();
        auto t0_coarsify = clock_type::now();
#endif

        if (fmm_type != MASK_TYPE::STREAM && fmm_type != MASK_TYPE::IB2xIB && fmm_type != MASK_TYPE::xIB2IB)
            source_coarsify<source_tmp_type, source_tmp_type>(_field_idx, 0, Source::mesh_type());

#ifdef POISSON_TIMINGS
        auto t1_coarsify = clock_type::now();
        timings_.coarsification = t1_coarsify - t0_coarsify;
        const auto t0_level_interaction = clock_type::now();
#endif

        //Level-Interactions

        const int l_max = (fmm_type != MASK_TYPE::STREAM) ?
                    domain_->tree()->depth() : domain_->tree()->base_level()+1;

        const int l_min = (fmm_type !=  MASK_TYPE::IB2xIB && fmm_type !=  MASK_TYPE::xIB2IB) ?
                    domain_->tree()->base_level() : domain_->tree()->depth()-1;

        for (int l = l_min; l < l_max; ++l)
        {
#ifdef POISSON_TIMINGS
            const auto t0_level = clock_type::now();
#endif
            _kernel->change_level(l - domain_->tree()->base_level());

            for (auto it_s = domain_->begin(l); it_s != domain_->end(l); ++it_s)
                if (it_s->has_data() && !it_s->locally_owned())
                {
                    if (!it_s->data().is_allocated()) continue;
                    auto& cp2 = it_s->data_r(source_tmp).linalg_data();
                    cp2 *= 0.0;
                }

            if (subtract_non_leaf_)
            {
                //fmm_.template apply<source_tmp_type, target_tmp_type>(
                //    domain_, _kernel, l, false, 1.0, fmm_type);
                //// Copy to Target
                //for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                //    if (it->locally_owned() && it->is_leaf())
                //    {
                //        it->data_r(Target::tag(), _field_idx)
                //            .linalg()
                //            .get()
                //            ->cube_noalias_view() =
                //            it->data_r(target_tmp).linalg_data();
                //    }
                //fmm_.template apply<source_tmp_type, target_tmp_type>(
                //    domain_, _kernel, l, true, -1.0);
#ifdef POISSON_T//IMINGS
                //timings_.fmm_level_nl[l - domain_->tree()->base_level()] =
                //    fmm_.timings();
#endif

#ifdef POISSON_T//IMINGS
                //const auto t2 = clock_type::now();
#endif
                //// Interpolate
                //domain_->decomposition()
                //    .client()
                //    ->template communicate_updownward_assign<target_tmp_type,
                //        target_tmp_type>(l, false, false, -1);

                //for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                //{
                //    if (!it->has_data() || !it->data().is_allocated())
                //        continue;
                //    c_cntr_nli_
                //        .nli_intrp_node<target_tmp_type, target_tmp_type>(
                //            it, Source::mesh_type(), _field_idx, 0, false, false);
                //}

#ifdef POISSON_T//IMINGS
                //const auto t3 = clock_type::now();
                //timings_.interpolation += (t3 - t2);
#endif
            }
            else
            {
                if (fmm_type == MASK_TYPE::AMR2AMR)
                {
                    // outside to everywhere
                    fmm_.template apply<source_tmp_type, target_tmp_type>(
                        domain_, _kernel, l, true, 1.0, fmm_type);
#ifdef POISSON_TIMINGS
                    timings_.fmm_level_nl[l - domain_->tree()->base_level()] =
                        fmm_.timings();
#endif

#ifdef POISSON_TIMINGS
                    const auto t2 = clock_type::now();
#endif
                    // Interpolate
                    domain_->decomposition()
                        .client()
                        ->template communicate_updownward_assign<
                            target_tmp_type, target_tmp_type>(
                            l, false, false, -1);

                    for (auto it = domain_->begin(l); it != domain_->end(l);
                         ++it)
                    {
                        if (!it->has_data() || !it->data().is_allocated())
                            continue;
                        c_cntr_nli_.template nli_intrp_node<target_tmp_type, target_tmp_type>(
                                it, Source::mesh_type(), _field_idx, 0, false,
                                false);
                    }

#ifdef POISSON_TIMINGS
                    const auto t3 = clock_type::now();
                    timings_.interpolation += (t3 - t2);
#endif
                }

                // Inside to outside
                fmm_.template apply<source_tmp_type, target_tmp_type>(
                    domain_, _kernel, l, false, 1.0, fmm_type);

                // Copy to Target
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                    if (it->locally_owned() && it->is_leaf())
                    {
                        it->data_r(Target::tag(), _field_idx)
                            .linalg()
                            .get()
                            ->cube_noalias_view() =
                            it->data_r(target_tmp).linalg_data();
                    }
            }

#ifdef POISSON_TIMINGS
            timings_.fmm_level[l - domain_->tree()->base_level()] =
                fmm_.timings();
#endif

            if (fmm_type!=MASK_TYPE::AMR2AMR) continue;
            if (use_correction_)
            {
                if (l == domain_->tree()->depth() - 1) continue;

                // Correction for LGF
                // Calculate PL numerically from target_tmp_type instead of assume PL
                // L^-1 S gives back S exactly.  This improves the accuracy

                //for (auto it  = domain_->begin(l);
                //        it != domain_->end(l); ++it)
                //{
                //    int refinement_level = it->refinement_level();
                //    double dx_level = dx_base/std::pow(2,refinement_level);

                //    if (!it->has_data() || !it->data().is_allocated()) continue;
                //    domain::Operator::laplace<target_tmp_type, corr_lap_tmp>
                //    ( *(it->has_data()),dx_level);
                //}

                client->template buffer_exchange<source_tmp_type>(l);
                domain_->decomposition()
                    .client()
                    ->template communicate_updownward_assign<source_tmp_type,
                        source_tmp_type>(l, false, false, -1);
                client->template buffer_exchange<target_tmp_type>(l+1);
                domain_->decomposition()
                    .client()
                    ->template communicate_updownward_assign<target_tmp_type,
                        target_tmp_type>(l+1, false, false, -1);

                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->has_data() || !it->data().is_allocated())
                        continue;

                    const bool correction_buffer_only = true;
                    c_cntr_nli_.template nli_intrp_node<source_tmp_type, correction_tmp_type>(
                            it, Source::mesh_type(), _field_idx, 0,
                            correction_buffer_only, false);
                }

                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    int    refinement_level = it->refinement_level();
                    double dx = dx_base / std::pow(2, refinement_level);
                    
                    

                    if (!_kernel->LaplaceLGF())
                    {
                        float_type c_val = _kernel->return_c_level() / dx;
                        c_cntr_nli_.template add_source_correction<target_tmp_type,
                        correction_tmp_type>(it, dx / 2.0, c_val);
                    }
                    else {
                        c_cntr_nli_.template add_source_correction<target_tmp_type,
                        correction_tmp_type>(it, dx / 2.0);
                    }
                }

                for (auto it = domain_->begin(l + 1); it != domain_->end(l + 1);
                     ++it) {
                    /*if (!_kernel->LaplaceLGF())
                    {
                        int    refinement_level = it->refinement_level();
                        double dx = dx_base / std::pow(2, refinement_level);
                        float_type c_val = _kernel->return_c_level() / 2.0 / dx;
                        if (it->locally_owned() && it->has_data() && it->data().is_allocated())
                        {
                            auto& lin_data_1 =
                                it->data_r(target_tmp).linalg_data();
                            auto& lin_data_2 =
                                it->data_r(correction_tmp).linalg_data();

                            xt::noalias(lin_data_2) += lin_data_1 * c_val * c_val;
                        }
                    }*/
                    if (it->locally_owned())
                    {
                        auto& lin_data_1 =
                            it->data_r(correction_tmp).linalg_data();
                        auto& lin_data_2 =
                            it->data_r(source_tmp).linalg_data();

                        xt::noalias(lin_data_2) += lin_data_1 * 1.0;
                    }
                }
            }
#ifdef POISSON_TIMINGS
            const auto     t1_level = clock_type::now();
            mDuration_type tmp = t1_level - t0_level;
            timings_.level[l - domain_->tree()->base_level()] =
                t1_level - t0_level;
#endif
        }

#ifdef POISSON_TIMINGS
        const auto t1_level_interaction = clock_type::now();
        timings_.level_interaction = t1_level_interaction -
                                     t0_level_interaction -
                                     timings_.interpolation;
        const auto t1_all = clock_type::now();
        timings_.global = t1_all - t0_all;
#endif
    }

    template<class field>
    void clean_field()
    {
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->has_data() || !it->data().is_allocated()) continue;

            auto& lin_data = it->data_r(field::tag()).linalg_data();
            std::fill(lin_data.begin(), lin_data.end(), 0.0);
        }
    }

    template<class from, class to>
    void copy_level(int level, std::size_t _field_idx_from = 0,
        std::size_t _field_idx_to = 0, bool with_buffer = false)
    {
        for (auto it = domain_->begin(level); it != domain_->end(level); ++it)
            if (it->locally_owned())
            {
                auto& lin_data_1 =
                    it->data_r(from::tag(), _field_idx_from).linalg_data();
                auto& lin_data_2 =
                    it->data_r(to::tag(), _field_idx_to).linalg_data();

                if (with_buffer) xt::noalias(lin_data_2) = lin_data_1 * 1.0;
                else {
		    if (Dim == 3) {
                    xt::noalias(view(lin_data_2, xt::range(1, -1),
                        xt::range(1, -1), xt::range(1, -1))) = view(lin_data_1,
                        xt::range(1, -1), xt::range(1, -1), xt::range(1, -1));
		    }
		    else {
                    xt::noalias(view(lin_data_2, xt::range(1, -1),
                        xt::range(1, -1))) = view(lin_data_1,
                        xt::range(1, -1), xt::range(1, -1));
		    }
		}
            }
    }

    template<class from, class to>
    void copy_leaf(std::size_t _field_idx_from = 0,
        std::size_t _field_idx_to = 0, bool with_buffer = false)
    {
        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves(); ++it)
            if (it->locally_owned())
            {
                auto& lin_data_1 =
                    it->data_r(from::tag(), _field_idx_from).linalg_data();
                auto& lin_data_2 =
                    it->data_r(to::tag(), _field_idx_to).linalg_data();

                if (with_buffer) xt::noalias(lin_data_2) = lin_data_1 * 1.0;
                /*else
                    xt::noalias(view(lin_data_2, xt::range(1, -1),
                        xt::range(1, -1), xt::range(1, -1))) = view(lin_data_1,
                        xt::range(1, -1), xt::range(1, -1), xt::range(1, -1));*/

		if (Dim == 3) {
                    xt::noalias(view(lin_data_2, xt::range(1, -1),
                        xt::range(1, -1), xt::range(1, -1))) = view(lin_data_1,
                        xt::range(1, -1), xt::range(1, -1), xt::range(1, -1));
		}
		else {
                    xt::noalias(view(lin_data_2, xt::range(1, -1),
                        xt::range(1, -1))) = view(lin_data_1,
                        xt::range(1, -1), xt::range(1, -1));
		}
            }
    }
    template<class From, class To>
    void intrp_to_correction_buffer(std::size_t real_mesh_field_idx,
        std::size_t tmp_type_field_idx, MeshObject mesh_type,
        bool correction_only = true, bool exclude_correction = false,
        bool leaf_boundary = false)
    {
        auto client = domain_->decomposition().client();
        if (!client) return;

        for (int l = domain_->tree()->depth() - 2;
             l >= domain_->tree()->base_level(); --l)
        {
            client->template buffer_exchange<From>(l);

            domain_->decomposition()
                .client()
                ->template communicate_updownward_assign<From, From>(
                    l, false, false, -1, tmp_type_field_idx, leaf_boundary);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->has_data() || !it->data().is_allocated()) continue;
                if (leaf_boundary && !it->leaf_boundary()) continue;

                c_cntr_nli_.template nli_intrp_node<From, To>(it, mesh_type,
                    real_mesh_field_idx, tmp_type_field_idx, correction_only,
                    exclude_correction);
            }
        }
    }

    template<class From, class To>
    void source_coarsify(std::size_t real_mesh_field_idx,
        std::size_t tmp_type_field_idx, MeshObject mesh_type,
        bool correction_only = false, bool exclude_correction = false,
        bool _buffer_exchange = false, bool leaf_boundary = false)
    {

        leaf_boundary=false;
        auto client = domain_->decomposition().client();
        if (!client) return;

        for (int ls = domain_->tree()->depth() - 2;
             ls >= domain_->tree()->base_level(); --ls)
        {
            for (auto it_s = domain_->begin(ls); it_s != domain_->end(ls);
                 ++it_s)
            {
                if (!it_s->has_data() || !it_s->data().is_allocated())
                    continue;
                if (leaf_boundary && !it_s->leaf_boundary()) continue;

                c_cntr_nli_.template nli_antrp_node<From, To>(*it_s, mesh_type,
                    real_mesh_field_idx, tmp_type_field_idx, correction_only,
                    exclude_correction);
            }

            domain_->decomposition()
                .client()
                ->template communicate_updownward_add<To, To>(
                    ls, true, false, -1, tmp_type_field_idx, leaf_boundary);
        }

        if (_buffer_exchange)
        {
            for (int l = domain_->tree()->base_level();
                 l < domain_->tree()->depth() - 1; ++l)
                client->template buffer_exchange<To>(l);
        }
    }

    auto& c_cntr_nli() { return c_cntr_nli_; }

    /** @brief Compute the laplace operator of the target field and store
     *         it in diff_target.
     */
    template<class Target, class DiffTarget>
    void apply_laplace()
    {
        const float_type dx_base = domain_->dx_base();
        const auto       target = Target::tag();
        const auto       difftarget = DiffTarget::tag();

        //Coarsification:
        pcout << "Laplace - coarsification " << std::endl;
        for (int ls = domain_->tree()->depth() - 2;
             ls >= domain_->tree()->base_level(); --ls)
        {
            for (auto it_s = domain_->begin(ls); it_s != domain_->end(ls);
                 ++it_s)
            {
                //if (!it_s->has_data()) continue;
                if (!it_s->has_data() || !it_s->data().is_allocated())
                    continue;
                this->coarsify<Target, Target>(*it_s);
            }

            domain_->decomposition()
                .client()
                ->template communicate_updownward_add<Target, Target>(
                    ls, true, false, -1);
        }

        //Level-Interactions
        pcout << "Laplace - level interactions " << std::endl;
        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->has_data() || !it->locally_owned() ||
                    !it->data().is_allocated())
                    continue;
                auto refinement_level = it->refinement_level();
                auto dx_level = dx_base / std::pow(2, refinement_level);

                auto& diff_target_data =
                    it->data_r(DiffTarget::tag()).linalg_data();

                // laplace of it_t data with zero bcs
                if ((it->is_leaf()))
                {
                    for (auto& node : it->data())
                    {
			if (Dim == 3){
                        node(difftarget) = -6.0 * node(target) +
                                           node.at_offset(target, 0, 0, -1) +
                                           node.at_offset(target, 0, 0, +1) +
                                           node.at_offset(target, 0, -1, 0) +
                                           node.at_offset(target, 0, +1, 0) +
                                           node.at_offset(target, -1, 0, 0) +
                                           node.at_offset(target, +1, 0, 0);
			}
			else {
                        node(difftarget) = -4.0 * node(target) +
                                           node.at_offset(target, 0, -1) +
                                           node.at_offset(target, 0, +1) +
                                           node.at_offset(target, -1, 0) +
                                           node.at_offset(target, +1, 0);
			}
                    }
                }
                diff_target_data *= (1 / dx_level) * (1 / dx_level);
            }
        }
    }

    template<class Source, class Target>
    void solve()
    {
        apply_lgf<Source, Target>();
    }

    template<class Target, class Laplace>
    void laplace_diff()
    {
        //apply_laplace<Target, Laplace>();
    }

    template<class Field_c, class Field_p>
    void coarsify(octant_t* _parent, float_type factor = 1.0,
        bool correction_only = false, bool exclude_correction = false)
    {
        auto parent = _parent;

        for (int i = 0; i < parent->num_children(); ++i)
        {
            auto child = parent->child(i);
            if (child == nullptr || !child->has_data() ||
                !child->locally_owned())
                continue;
            if (correction_only && !child->is_correction()) continue;
            if (exclude_correction && child->is_correction()) continue;

            auto child_view = child->data().descriptor();

            auto cview = child->data().node_field().view(child_view);

            cview.iterate([&](auto& n) {
                const float_type avg = 1. / (std::pow(2,Dim)) * n(Field_c::tag());
                auto             pcoord = n.level_coordinate();
                for (std::size_t d = 0; d < pcoord.size(); ++d)
                    pcoord[d] = std::floor(pcoord[d] / 2.0);
                parent->data_r(Field_p::tag(), pcoord) += avg * factor;
            });
        }
    }

    auto&       timings() noexcept { return timings_; }
    const auto& timings() const noexcept { return timings_; }

    template<class Out>
    void print_timings(Out& os, Out& os_level)
    {
        int width = 20;
        timings_.accumulate(this->domain_->client_communicator());
        const auto pts = this->domain_->get_nPoints();

        os << std::left << std::setw(15) << "npts" << std::setw(width)
           << "global[s]" << std::setw(width) << "gbl rate[pts/s]"
           << std::setw(width) << "gbl eff[s/pt]" << std::setw(width)
           << "coarsing[%]" << std::setw(width) << "level[%]"
           << std::setw(width) << "interp[%]" << std::endl;
        os << std::left << std::scientific << std::setprecision(7)
           << std::setw(15) << pts.back() << std::setw(width)
           << timings_.global.count() / 1.e3 << std::setw(width)
           << pts.back() /
                  static_cast<float_type>(timings_.global.count() / 1.e3)
           << std::setw(width)
           << static_cast<float_type>(timings_.global.count() / 1.e3) /
                  pts.back()
           //<<std::defaultfloat

           <<std::setw(width) <<100.0*static_cast<float_type>(timings_.coarsification.count())/timings_.global.count()
           <<std::setw(width) <<100.0*static_cast<float_type>(timings_.level_interaction.count())/timings_.global.count()
           <<std::setw(width) <<100.0*static_cast<float_type>(timings_.interpolation.count())/timings_.global.count()
       <<std::endl;


       int c=0;
       width=15;
       os_level<<std::left<<std::scientific<<std::setprecision(5)
            <<std::setw(10) <<"level"
            <<std::setw(15) <<"npts"
            <<std::setw(width) <<"gbl[s]"
            <<std::setw(width) <<"rate[pts/s]"
            <<std::setw(width) <<"eff[s/pt]"

            <<std::setw(width) <<"fmm gbl "
            <<std::setw(width) <<"anterp "
            <<std::setw(width) <<"Bx "
            <<std::setw(width) <<"fft "
            <<std::setw(width) <<"fft ratio "
            <<std::setw(width) <<"interp "

            <<std::setw(width) <<"fmm_nl gbl "
            <<std::setw(width) <<"anterp "
            <<std::setw(width) <<"Bx "
            <<std::setw(width) <<"fft "
            <<std::setw(width) <<"fft ratio "
            <<std::setw(width) <<"interp "
       <<std::endl;

       for(std::size_t i=0; i<timings_.level.size();++i)
       {

           auto& t=timings_.level[i];
           auto& fmm=timings_.fmm_level[i];
           auto& fmm_nl=timings_.fmm_level_nl[i];

           os_level<<std::setw(10)<<c
             <<std::setw(15)<<pts[c]
             <<std::setw(width)<<static_cast<float_type>(t.count())/1.e3
             <<std::setw(width)<<pts[c]/(static_cast<float_type>(t.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(t.count())/1.e3)/pts[c]

             <<std::setw(width)<<(static_cast<float_type>(fmm.global.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm.anterp.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm.bx.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm.fftw.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm.fftw_count_max)/fmm.fftw_count_min)
             <<std::setw(width)<<(static_cast<float_type>(fmm.interp.count())/1.e3)

             <<std::setw(width)<<(static_cast<float_type>(fmm_nl.global.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm_nl.anterp.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm_nl.bx.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm_nl.fftw.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm_nl.fftw_count_max)/fmm_nl.fftw_count_min)
             <<std::setw(width)<<(static_cast<float_type>(fmm_nl.interp.count())/1.e3)

           <<std::endl;
           ++c;
       }
       os_level<<std::endl;
       os<<std::endl;
       //os_level<<std::defaultfloat<<std::endl;
       //os<<std::defaultfloat<<std::endl;
   }


   const bool& subtract_non_leaf()const noexcept{return subtract_non_leaf_;}
   bool& subtract_non_leaf()noexcept{return subtract_non_leaf_;}

   const bool& use_correction()const noexcept{return use_correction_;}
   bool& use_correction()noexcept{return use_correction_;}



private:
    domain_type*                      domain_;    ///< domain
    Fmm_t                             fmm_;       ///< fast-multipole
    lgf_lap_t                         lgf_lap_;
    lgf_if_t                          lgf_if_;
    std::vector<helm_t>               lgf_helm_vec;
    int                               N_fourier_modes;
    interpolation_type                c_cntr_nli_;///< Lagrange Interpolation
    parallel_ostream::ParallelOstream pcout=parallel_ostream::ParallelOstream(1);
    bool use_correction_ =true;
    bool subtract_non_leaf_ = false;

    int additional_modes = 0;
    float_type c_z_ = 1.0;

    //Timings:
    struct Timings
    {
        std::vector<mDuration_type>          level;
        std::vector<typename Fmm_t::Timings> fmm_level;
        std::vector<typename Fmm_t::Timings> fmm_level_nl;
        mDuration_type                       global = mDuration_type(0);
        mDuration_type                       coarsification = mDuration_type(0);
        mDuration_type level_interaction = mDuration_type(0);
        mDuration_type interpolation = mDuration_type(0);

        //TODO
        //Gather all and take max
        void accumulate(boost::mpi::communicator _comm) noexcept
        {
            Timings tlocal = *this;

            std::vector<decltype(tlocal.global.count())> clevel;
            clevel.resize(level.size());
            decltype(tlocal.global.count()) cglobal, ccoarsification,
                clevel_interaction, cinterpolation;

            boost::mpi::all_reduce(_comm, tlocal.global.count(), cglobal,
                boost::mpi::maximum<float_type>());
            boost::mpi::all_reduce(_comm, tlocal.coarsification.count(),
                ccoarsification, boost::mpi::maximum<float_type>());
            boost::mpi::all_reduce(_comm, tlocal.level_interaction.count(),
                clevel_interaction, boost::mpi::maximum<float_type>());
            boost::mpi::all_reduce(_comm, tlocal.interpolation.count(),
                cinterpolation, boost::mpi::maximum<float_type>());

            //For levels:
            for (std::size_t i = 0; i < level.size(); ++i)
            {
                fmm_level[i].accumulate(_comm);
                fmm_level_nl[i].accumulate(_comm);

                boost::mpi::all_reduce(_comm, tlocal.level[i].count(),
                    clevel[i], boost::mpi::maximum<float_type>());
                this->level[i] = mDuration_type(clevel[i]);
            }
            this->global = mDuration_type(cglobal);
            this->coarsification = mDuration_type(ccoarsification);
            this->interpolation = mDuration_type(cinterpolation);
            this->level_interaction = mDuration_type(clevel_interaction);
        }

        friend std::ostream& operator<<(std::ostream& os, const Timings& _t)
        {
            //auto nPts=_t.domain_->get_nPoints();
            //const int nLevels=_t.domain_->nLevels();

            const int width = 20;
            os << std::left << std::setw(width) << "global [ms]"
               << std::setw(width) << "coarsification" << std::setw(width)
               << "level_interaction " << std::setw(width) << "interpolation"
               << std::endl;

            os << std::left << std::scientific << std::setprecision(10)
               << std::setw(width) << _t.global.count() << std::setw(width)
               << _t.coarsification.count() << std::setw(width)
               << _t.level_interaction.count() << std::setw(width)
               << _t.interpolation.count() << std::endl;

            os << "Time per level:" << std::endl;
            int c = 0;
            for (auto& t : _t.level)
            {
                os << std::setw(4) << c++ << std::setw(width) << t.count()
                   << std::endl;
            }
            os << std::endl;
            return os;
        }
    };

    Timings timings_;
};

} // namespace solver
} // namespace iblgf

#endif // IBLGF_INCLUDED_POISSON_HPP
