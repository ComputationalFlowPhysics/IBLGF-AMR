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

#ifndef IBLGF_INCLUDED_SOLVER_POISSON_OMP_HPP
#define IBLGF_INCLUDED_SOLVER_POISSON_OMP_HPP

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
class PoissonSolverOMP
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

    PoissonSolverOMP(simulation_type* _simulation, int _N = 0)
    :
    domain_(_simulation->domain_.get()),
    fmm_(domain_,domain_->block_extent()[0]+lBuffer+rBuffer),
    c_cntr_nli_(domain_->block_extent()[0]+lBuffer+rBuffer, _simulation->intrp_order()),
    N_fourier_modes(_N)
    {
        //initializing vector of helmholtz solvers
        //_N is the number of Fourier modes computed at finest level, but c is the wave number at the base level. 
        //Currently the baselevel value is used as the parameter for LGF, the LGF structure includes this parameter and can be changed
        //when change level, thus we only need to initialize using base level parameter
        const int l_max = domain_->tree()->depth();
        const int l_min = domain_->tree()->base_level();
        const int nLevels = l_max - l_min;

        const float_type dx_base = domain_->dx_base();

        c_z_ = _simulation->dictionary()->template get_or<float_type>("L_z", 1.0);

        for (int i = 0; i < N_fourier_modes; i++) {
            float_type c = (static_cast<float_type>(i)+1.0) * 2.0 * M_PI * dx_base/c_z_;
            lgf_helm_vec_ptr.emplace_back(new helm_t(c));
        }

        for (int i = 0; i < N_fourier_modes; i++) {
            float_type c = (static_cast<float_type>(i)+1.0) * 2.0 * M_PI * dx_base/c_z_;
            lgf_helm_vec.emplace_back(helm_t(c));
        }

        Omegas.resize(N_fourier_modes + 1);
        for (int i = 0; i < (N_fourier_modes + 1); i++) {
            float_type Omega = static_cast<float_type>(i) * 2.0 * M_PI / c_z_;
            Omegas[i] = Omega;
        }

        /*for (int i = 0; i < N_fourier_modes; i++) {
            float_type c = (static_cast<float_type>(i)+1.0)/static_cast<float_type>(N_fourier_modes + 1) * 2.0 * M_PI * std::pow(2.0, nLevels - 1);
            lgf_helm_vec.emplace_back(helm_t(c));
        }*/
        
        for (int i = 0; i < source_tmp_type::nFields(); i++) {
            //c_cntr_nli_vec.emplace_back((new interpolation_type(domain_->block_extent()[0]+lBuffer+rBuffer, _simulation->intrp_order())));
            lgf_if_vec.emplace_back((new lgf_if_t()));
        }

        auto n_max_threads = omp_get_max_threads();
        for (int i = 0; i < n_max_threads; i++) {
            c_cntr_nli_vec.emplace_back((new interpolation_type(domain_->block_extent()[0]+lBuffer+rBuffer, _simulation->intrp_order())));
        }

        //c_cntr_nli_vec.resize(source_tmp_type::nFields());


        //setting if only advect a subset of modes, the total simulated modes are additional_modes + 1
        additional_modes = _simulation->dictionary()->template get_or<int>("add_modes", N_fourier_modes);
        //std::cout << "Number of Fourier modes are " << N_fourier_modes <<std::endl;
        if (additional_modes > N_fourier_modes) throw std::runtime_error("Additional modes cannot be higher than the number of Fourier modes");

        adapt_Fourier = _simulation->dictionary()->template get_or<bool>(
            "adapt_Fourier", false);
    }

  public:

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

        const int l_max = (fmm_type != MASK_TYPE::STREAM) ? domain_->tree()->depth() : domain_->tree()->base_level()+1;
        const int l_min = (fmm_type !=  MASK_TYPE::IB2xIB && fmm_type !=  MASK_TYPE::xIB2IB) ? domain_->tree()->base_level() : domain_->tree()->depth()-1;

        const int tot_ref_l = l_max - l_min - 1;

        for (int NthField = 0; NthField < NComp; NthField++) {
            apply_lgf<Source, Target>(&lgf_lap_, lgf_helm_vec_ptr, NthField, fmm_type);
        }
    }


    template<class Source, class Target>
    void apply_helm_if(float_type _alpha_base, int N_modes, int NComp = 3, 
        int fmm_type = MASK_TYPE::AMR2AMR)
    {
        for (int i = 0; i < source_tmp_type::nFields(); i++) {
            lgf_if_vec[i]->alpha_base_level() = _alpha_base;
        }
        lgf_if_.alpha_base_level() = _alpha_base;
        if (N_modes != (N_fourier_modes + 1))
            throw std::runtime_error(
                "Fourier modes do not match in helmholtz solver");
        if (Source::nFields() != N_modes * 2 * NComp)
            throw std::runtime_error(
                "Fourier modes number elements do not match in helmholtz solver");

        const int l_max = (fmm_type != MASK_TYPE::STREAM) ? domain_->tree()->depth() : domain_->tree()->base_level()+1;
        const int l_min = (fmm_type !=  MASK_TYPE::IB2xIB && fmm_type !=  MASK_TYPE::xIB2IB) ? domain_->tree()->base_level() : domain_->tree()->depth()-1;

        const int tot_ref_l = l_max - l_min - 1;


        for (int NthField = 0; NthField < NComp; NthField++) {
            this->apply_if_helm<Source, Target>(lgf_if_vec, Omegas, NthField, fmm_type);
        }
    }


    template<class Source, class Target>
    void apply_lgf_and_helm_ib(int N_modes, const std::vector<bool>& ModesBool, int NComp = 1,
        int fmm_type = MASK_TYPE::AMR2AMR)
    {
        if (N_modes != (N_fourier_modes + 1))
            throw std::runtime_error(
                "Fourier modes do not match in helmholtz solver");
        if (Source::nFields() != N_modes * 2 * NComp)
            throw std::runtime_error(
                "Fourier modes number elements do not match in helmholtz solver");

        const int l_max = (fmm_type != MASK_TYPE::STREAM) ? domain_->tree()->depth() : domain_->tree()->base_level()+1;
        const int l_min = (fmm_type !=  MASK_TYPE::IB2xIB && fmm_type !=  MASK_TYPE::xIB2IB) ? domain_->tree()->base_level() : domain_->tree()->depth()-1;

        const int tot_ref_l = l_max - l_min - 1;

        for (int NthField = 0; NthField < NComp; NthField++) {
            apply_lgf_ib<Source, Target>(&lgf_lap_, lgf_helm_vec_ptr, NthField, ModesBool, fmm_type);
        }
    }

    template<class Source, class Target>
    void apply_helm_if_ib(float_type _alpha_base, int N_modes, const std::vector<bool>& ModesBool, int NComp = 3, 
        int fmm_type = MASK_TYPE::AMR2AMR)
    {
        for (int i = 0; i < source_tmp_type::nFields(); i++) {
            lgf_if_vec[i]->alpha_base_level() = _alpha_base;
        }
        lgf_if_.alpha_base_level() = _alpha_base;
        if (N_modes != (N_fourier_modes + 1))
            throw std::runtime_error(
                "Fourier modes do not match in helmholtz solver");
        if (Source::nFields() != N_modes * 2 * NComp)
            throw std::runtime_error(
                "Fourier modes number elements do not match in helmholtz solver");

        const int l_max = (fmm_type != MASK_TYPE::STREAM) ? domain_->tree()->depth() : domain_->tree()->base_level()+1;
        const int l_min = (fmm_type !=  MASK_TYPE::IB2xIB && fmm_type !=  MASK_TYPE::xIB2IB) ? domain_->tree()->base_level() : domain_->tree()->depth()-1;

        const int tot_ref_l = l_max - l_min - 1;


        for (int NthField = 0; NthField < NComp; NthField++) {
            this->apply_if_helm_ib<Source, Target>(lgf_if_vec, Omegas, NthField, ModesBool, fmm_type);
        }
    }


    template<class Source, class Target, class Kernel_vec>
    void apply_if_helm(Kernel_vec& _kernel_vec, std::vector<float_type> omega, int NthComp, int fmm_type = MASK_TYPE::AMR2AMR, int addLevel=0)
    {
        auto client = domain_->decomposition().client();
        if (!client) return;

        //struct timespec tstart, t_copy, t_prep, t_apply;

        // Cleaning
        const float_type dx_base = domain_->dx_base();
        int starting_idx = (N_fourier_modes + 1) * NthComp * 2;

        //clock_gettime(CLOCK_REALTIME,&tstart);

        #pragma omp parallel for
        for (int mode = 0; mode < (N_fourier_modes + 1); mode++) {
            for (int j = 0; j < 2; j++) {
                int field_idx = mode * 2 + j;
                clean_field<source_tmp_type>(field_idx);
                clean_field<target_tmp_type>(field_idx);

                // Copy source
                if (fmm_type == MASK_TYPE::AMR2AMR)
                    copy_leaf<Source, source_tmp_type>(field_idx + starting_idx, field_idx, true);
                else if (fmm_type == MASK_TYPE::STREAM)
                    copy_level<Source, source_tmp_type>(domain_->tree()->base_level(), field_idx + starting_idx, field_idx, false);
                else if (fmm_type == MASK_TYPE::IB2xIB || fmm_type == MASK_TYPE::xIB2IB)
                    copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, field_idx + starting_idx, field_idx, false);
                else if (fmm_type == MASK_TYPE::IB2AMR)
                    copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, field_idx + starting_idx, field_idx, false);
            }
        }

        //clock_gettime(CLOCK_REALTIME,&t_copy);

        if (fmm_type != MASK_TYPE::STREAM && fmm_type != MASK_TYPE::IB2xIB && fmm_type != MASK_TYPE::xIB2IB)
        {
            // Coarsify
            source_coarsify_OMP<source_tmp_type, source_tmp_type>(Source::mesh_type(), starting_idx);

            // Interpolate to correction buffer
            intrp_to_correction_buffer_OMP<source_tmp_type, source_tmp_type>(Source::mesh_type(), starting_idx);
        }

        const int l_max = (fmm_type != MASK_TYPE::STREAM) ?
                    domain_->tree()->depth() : domain_->tree()->base_level()+1;

        const int l_min = (fmm_type !=  MASK_TYPE::IB2xIB && fmm_type !=  MASK_TYPE::xIB2IB) ?
                    (domain_->tree()->base_level() + addLevel) : domain_->tree()->depth()-1;


        if (l_min >= l_max)
            throw std::runtime_error(
                "Lmax smaller than Lmin in IF");

        for (int l = l_min; l < l_max; ++l)
        {
            
            #pragma omp parallel for
            for (std::size_t field_idx = 0; field_idx < source_tmp_type::nFields(); field_idx++) {
                for (auto it_s = domain_->begin(l); it_s != domain_->end(l); ++it_s) {
                    if (it_s->has_data() && !it_s->locally_owned())
                    {
                        if (!it_s->data().is_allocated()) continue;
                        auto& cp2 = it_s->data_r(source_tmp, field_idx);
                        std::fill(cp2.begin(), cp2.end(), 0.0);
                    }
                }
            }

            for (int idx = 0; idx < _kernel_vec.size(); idx++) {
                _kernel_vec[idx]->change_level(l - domain_->tree()->base_level());
            }

            double dx = dx_base / std::pow(2, l - domain_->tree()->base_level());

            //compute the add scale
            float_type alpha_level = _kernel_vec[0]->alpha_;
            //float_type helm_weights = std::exp(-omega*omega*alpha_level * dx * dx);

            std::vector<float_type> helm_weights_vec;

            for (int mode = 0; mode < (N_fourier_modes + 1); mode++) {
                float_type helm_weights = std::exp(-omega[mode]*omega[mode]*alpha_level * dx * dx);
                for (int j = 0; j < 2; j++) {
                    helm_weights_vec.emplace_back(helm_weights);
                }
            }

            //clock_gettime(CLOCK_REALTIME,&t_prep);

            if (fmm_type == MASK_TYPE::AMR2AMR)
                fmm_.template apply_IF<source_tmp_type, target_tmp_type>(
                        domain_, _kernel_vec, l, false, helm_weights_vec, fmm_type);

            if (!subtract_non_leaf_)
                fmm_.template apply_IF<source_tmp_type, target_tmp_type>(
                    domain_, _kernel_vec, l, true, helm_weights_vec, fmm_type);


            //clock_gettime(CLOCK_REALTIME,&t_apply);


            #pragma omp parallel for
            for (int mode = 0; mode < (N_fourier_modes + 1); mode++) {
                for (int j = 0; j < 2; j++) {
                    int field_idx = mode * 2 + j;
                    copy_level<target_tmp_type, Target>(l, field_idx, field_idx + starting_idx, true);
                }
            }

            //copy_level<target_tmp_type, Target>(l, 0, _field_idx, true);
        }

        /*double dt_copy = (t_copy.tv_sec+t_copy.tv_nsec/1e9)-(tstart.tv_sec+tstart.tv_nsec/1e9);
        double dt_prep = (t_prep.tv_sec+t_prep.tv_nsec/1e9)-(t_copy.tv_sec+t_copy.tv_nsec/1e9);
        double dt_apply = (t_apply.tv_sec+t_apply.tv_nsec/1e9)-(t_prep.tv_sec+t_prep.tv_nsec/1e9);
        if (domain_->client_communicator().rank() == 0) {
            std::cout << "dt_copy = " << dt_copy << " dt_prep = " << dt_prep << " dt_apply = " << dt_apply << std::endl;
            std::cout << "l_min = " << l_min << " l_max = " << l_max << std::endl;
        }*/
    }




    template<class Source, class Target, class Kernel_vec>
    void apply_if_helm_ib(Kernel_vec& _kernel_vec, std::vector<float_type> omega, int NthComp, const std::vector<bool>& ModesBool, int fmm_type = MASK_TYPE::AMR2AMR, int addLevel=0)
    {
        //only to be used within IB solver that does not require interpolations, thus no buffer exchange
        auto client = domain_->decomposition().client();
        if (!client) return;

        //struct timespec tstart, t_copy, t_prep, t_apply;

        std::vector<int> modeToCompute, field_idx2compute;

        modeToCompute.resize(0);
        field_idx2compute.resize(0);

        for (int i = 0; i < ModesBool.size();i++) {
            if (ModesBool[i]) {
                modeToCompute.emplace_back(i);
                field_idx2compute.emplace_back(i*2);
                field_idx2compute.emplace_back(i*2 + 1);
            }
        }

        // Cleaning
        const float_type dx_base = domain_->dx_base();
        int starting_idx = (N_fourier_modes + 1) * NthComp * 2;

        //clock_gettime(CLOCK_REALTIME,&tstart);

        int l = domain_->tree()->depth()-1;

        fmm_.template apply_single_master<source_tmp_type, target_tmp_type>(domain_, _kernel_vec[0].get(), l, false, fmm_type);

        #pragma omp parallel for
        for (int i = 0; i < field_idx2compute.size(); i++) {
            
            int field_idx = field_idx2compute[i];
            int mode = field_idx/2;
            clean_field<source_tmp_type>(field_idx);
            clean_field<target_tmp_type>(field_idx);

            // Copy source
            if (fmm_type == MASK_TYPE::AMR2AMR)
                copy_leaf<Source, source_tmp_type>(field_idx + starting_idx, field_idx, true);
            else if (fmm_type == MASK_TYPE::STREAM)
                copy_level<Source, source_tmp_type>(domain_->tree()->base_level(), field_idx + starting_idx, field_idx, false);
            else if (fmm_type == MASK_TYPE::IB2xIB || fmm_type == MASK_TYPE::xIB2IB)
                copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, field_idx + starting_idx, field_idx, false);
            else if (fmm_type == MASK_TYPE::IB2AMR)
                copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, field_idx + starting_idx, field_idx, false);


            for (auto it_s = domain_->begin(l); it_s != domain_->end(l); ++it_s) {
                if (it_s->has_data() && !it_s->locally_owned())
                {
                    if (!it_s->data().is_allocated()) continue;
                    auto& cp2 = it_s->data_r(source_tmp, field_idx);
                    std::fill(cp2.begin(), cp2.end(), 0.0);
                }
            }


            _kernel_vec[field_idx]->change_level(l - domain_->tree()->base_level());

            double dx = dx_base / std::pow(2, l - domain_->tree()->base_level());

            float_type alpha_level = _kernel_vec[field_idx]->alpha_;

            float_type helm_weights = std::exp(-omega[mode]*omega[mode]*alpha_level * dx * dx);

            if (!subtract_non_leaf_)
                fmm_.template apply_IF_single<source_tmp_type, target_tmp_type>(
                        domain_, _kernel_vec[field_idx].get(), field_idx, l, false, helm_weights, fmm_type);

            copy_level<target_tmp_type, Target>(l, field_idx, field_idx + starting_idx, true);
        }
    }

    template<class Source, class Target, class Kernel, class Kernel_vec>
    void apply_lgf_ib(Kernel* _kernel_lgf, Kernel_vec& _kernel, int NthComp, const std::vector<bool>& ModesBool,
        const int fmm_type, int addLevel = 0)
    {
        auto client = domain_->decomposition().client();
        if (!client) return;

        std::vector<int> modeToCompute, field_idx2compute;

        modeToCompute.resize(0);
        field_idx2compute.resize(0);

        for (int i = 0; i < ModesBool.size();i++) {
            if (ModesBool[i]) {
                modeToCompute.emplace_back(i);
                field_idx2compute.emplace_back(i*2);
                field_idx2compute.emplace_back(i*2 + 1);
            }
        }

        const int l = domain_->tree()->depth()-1;

        //struct timespec tstart, t_copy, t_prep, t_intrp, t_apply, t_corr;

        const float_type dx_base = domain_->dx_base();
        int starting_idx = (N_fourier_modes + 1) * NthComp * 2;


        //clock_gettime(CLOCK_REALTIME,&tstart);

        fmm_.template apply_single_master<source_tmp_type, target_tmp_type>(domain_, _kernel_lgf, l, false, fmm_type);


        // Cleaning
        #pragma omp parallel for
        for (int i = 0; i < modeToCompute.size(); i++) {
            int mode = modeToCompute[i];
            for (int j = 0; j < 2; j++) {
                int field_idx = mode * 2 + j;
                clean_field<source_tmp_type>(field_idx);
                clean_field<target_tmp_type>(field_idx);
                clean_field<correction_tmp_type>(field_idx);

                // Copy source
                if (fmm_type == MASK_TYPE::AMR2AMR)
                    copy_leaf<Source, source_tmp_type>(field_idx + starting_idx, field_idx, true);
                else if (fmm_type == MASK_TYPE::STREAM)
                    copy_level<Source, source_tmp_type>(domain_->tree()->base_level(), field_idx + starting_idx, field_idx, false);
                else if (fmm_type == MASK_TYPE::IB2xIB || fmm_type == MASK_TYPE::xIB2IB)
                    copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, field_idx + starting_idx, field_idx, false);
                else if (fmm_type == MASK_TYPE::IB2AMR)
                    copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, field_idx + starting_idx, field_idx, false);


                for (auto it_s = domain_->begin(l); it_s != domain_->end(l); ++it_s) {
                    if (it_s->has_data() && !it_s->locally_owned())
                    {
                        if (!it_s->data().is_allocated()) continue;
                        auto& cp2 = it_s->data_r(source_tmp, field_idx).linalg_data();
                        cp2 *= 0.0;
                    }
                }
            }

            if (mode == 0) {
                _kernel_lgf->change_level(l - domain_->tree()->base_level());
                fmm_.template apply_LGF_single<source_tmp_type, target_tmp_type>(
                    domain_, _kernel_lgf, 0, l, false, 1.0, fmm_type);
            }
            else {
                _kernel[mode - 1]->change_level(l - domain_->tree()->base_level());
                fmm_.template apply_LGF_single<source_tmp_type, target_tmp_type>(
                    domain_, _kernel[mode - 1].get(), mode, l, false, 1.0, fmm_type);
            }

            // Copy to Target
            for (int j = 0; j < 2; j++) {
                int field_idx = mode * 2 + j;
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it) {
                    if (it->locally_owned() && it->is_leaf())
                    {
                        it->data_r(Target::tag(), field_idx + starting_idx)
                            .linalg()
                            .get()
                            ->cube_noalias_view() =
                            it->data_r(target_tmp, field_idx).linalg_data();
                    }
                }
            }
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
    template<class Source, class Target, class Kernel, class Kernel_vec>
    void apply_lgf(Kernel* _kernel_lgf, Kernel_vec& _kernel, int NthComp,
        const int fmm_type, int addLevel = 0)
    {
        auto client = domain_->decomposition().client();
        if (!client) return;

        //struct timespec tstart, t_copy, t_prep, t_intrp, t_apply, t_corr;

        const float_type dx_base = domain_->dx_base();
        int starting_idx = (N_fourier_modes + 1) * NthComp * 2;


        //clock_gettime(CLOCK_REALTIME,&tstart);


        // Cleaning
        #pragma omp parallel for
        for (int mode = 0; mode < (N_fourier_modes + 1); mode++) {
            for (int j = 0; j < 2; j++) {
                int field_idx = mode * 2 + j;
                clean_field<source_tmp_type>(field_idx);
                clean_field<target_tmp_type>(field_idx);
                clean_field<correction_tmp_type>(field_idx);

                // Copy source
                if (fmm_type == MASK_TYPE::AMR2AMR)
                    copy_leaf<Source, source_tmp_type>(field_idx + starting_idx, field_idx, true);
                else if (fmm_type == MASK_TYPE::STREAM)
                    copy_level<Source, source_tmp_type>(domain_->tree()->base_level(), field_idx + starting_idx, field_idx, false);
                else if (fmm_type == MASK_TYPE::IB2xIB || fmm_type == MASK_TYPE::xIB2IB)
                    copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, field_idx + starting_idx, field_idx, false);
                else if (fmm_type == MASK_TYPE::IB2AMR)
                    copy_level<Source, source_tmp_type>(domain_->tree()->depth()-1, field_idx + starting_idx, field_idx, false);
            }
        }


        //clock_gettime(CLOCK_REALTIME,&t_copy);
        /*double dt_copy = (t_copy.tv_sec+t_copy.tv_nsec/1e9)-(tstart.tv_sec+tstart.tv_nsec/1e9);
        if (domain_->client_communicator().rank() == 0) {
            std::cout << "dt_copy = " << dt_copy << std::endl;
        }*/


        if (fmm_type != MASK_TYPE::STREAM && fmm_type != MASK_TYPE::IB2xIB && fmm_type != MASK_TYPE::xIB2IB)
            source_coarsify_OMP<source_tmp_type, source_tmp_type>(Source::mesh_type(), starting_idx);


        //clock_gettime(CLOCK_REALTIME,&t_intrp);
        //double dt_intrp = (t_intrp.tv_sec+t_intrp.tv_nsec/1e9)-(t_copy.tv_sec+t_copy.tv_nsec/1e9);
        /*if (domain_->client_communicator().rank() == 0) {
            std::cout << "dt_intrp = " << dt_intrp << std::endl;
        }*/


        const int l_max = (fmm_type != MASK_TYPE::STREAM) ?
                    domain_->tree()->depth() : domain_->tree()->base_level()+1;

        const int l_min = (fmm_type !=  MASK_TYPE::IB2xIB && fmm_type !=  MASK_TYPE::xIB2IB) ?
                    (domain_->tree()->base_level() + addLevel) : domain_->tree()->depth()-1;

        if (l_min >= l_max)
            throw std::runtime_error(
                "Lmax smaller than Lmin");

        for (int l = l_min; l < l_max; ++l)
        {


            //clock_gettime(CLOCK_REALTIME,&t_intrp);


            #pragma omp parallel for
            for (int kernel_idx = 0; kernel_idx < (_kernel.size() + 1); kernel_idx++) {
                if (kernel_idx == 0) {
                    _kernel_lgf->change_level(l - domain_->tree()->base_level());
                }
                else {
                    _kernel[kernel_idx - 1]->change_level(l - domain_->tree()->base_level());
                }
            }

            #pragma omp parallel for
            for (std::size_t field_idx = 0; field_idx < source_tmp_type::nFields(); field_idx++) {
                for (auto it_s = domain_->begin(l); it_s != domain_->end(l); ++it_s) {
                    if (it_s->has_data() && !it_s->locally_owned())
                    {
                        if (!it_s->data().is_allocated()) continue;
                        auto& cp2 = it_s->data_r(source_tmp, field_idx).linalg_data();
                        cp2 *= 0.0;
                    }
                }
            }

            
            if (fmm_type == MASK_TYPE::AMR2AMR)
            {
                // outside to everywhere
                fmm_.template apply_LGF<source_tmp_type, target_tmp_type>(
                    domain_, _kernel_lgf, 2, _kernel, 2, l, true, 1.0, fmm_type);

                
                //clock_gettime(CLOCK_REALTIME,&t_apply);
                /*double dt_apply = (t_apply.tv_sec+t_apply.tv_nsec/1e9)-(t_intrp.tv_sec+t_intrp.tv_nsec/1e9);
                if (domain_->client_communicator().rank() == 0) {
                    std::cout << "l = " << l << " dt_apply = " << dt_apply << std::endl;
                }*/

                // Interpolate
                #pragma omp parallel for 
                for (int field_idx = 0; field_idx < target_tmp_type::nFields(); field_idx++)
                {
                    int omp_idx = omp_get_thread_num( );
                    domain_->decomposition()
                        .client()
                        ->template communicate_updownward_assign_OMP<
                            target_tmp_type, target_tmp_type>(
                            l, false, false, -1, field_idx);

                    for (auto it = domain_->begin(l); it != domain_->end(l);
                            ++it)
                    {
                        if (!it->has_data() || !it->data().is_allocated())
                            continue;
                        c_cntr_nli_vec[omp_idx]->template nli_intrp_node<target_tmp_type, target_tmp_type>(
                                it, Source::mesh_type(), field_idx + starting_idx, field_idx, false,
                                false);
                    }
                }


                /*clock_gettime(CLOCK_REALTIME,&t_intrp);
                double dt_intrp = (t_intrp.tv_sec+t_intrp.tv_nsec/1e9)-(t_apply.tv_sec+t_apply.tv_nsec/1e9);
                if (domain_->client_communicator().rank() == 0) {
                    std::cout << "l = " << l << " dt_intrp = " << dt_intrp << std::endl;
                }*/

            }

            // Inside to outside
            fmm_.template apply_LGF<source_tmp_type, target_tmp_type>(
                domain_, _kernel_lgf, 2, _kernel, 2, l, false, 1.0, fmm_type);

            // Copy to Target
            #pragma omp parallel for 
            for (std::size_t _field_idx = 0; _field_idx < target_tmp_type::nFields();++_field_idx)
            {
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it) {
                    if (it->locally_owned() && it->is_leaf())
                    {
                        it->data_r(Target::tag(), _field_idx + starting_idx)
                            .linalg()
                            .get()
                            ->cube_noalias_view() =
                            it->data_r(target_tmp, _field_idx).linalg_data();
                    }
                }
            }


            /*clock_gettime(CLOCK_REALTIME,&t_apply);
            double dt_apply = (t_apply.tv_sec+t_apply.tv_nsec/1e9)-(t_intrp.tv_sec+t_intrp.tv_nsec/1e9);
            if (domain_->client_communicator().rank() == 0) {
                std::cout << "l = " << l << " dt_apply = " << dt_apply << std::endl;
            }*/


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
                client->template buffer_exchange<target_tmp_type>(l+1);
                #pragma omp parallel for 
                for (int kernel_idx = 0; kernel_idx < (_kernel.size() + 1); kernel_idx++) {
                    int omp_idx = omp_get_thread_num( );
                    for (int idx_sub = 0; idx_sub < 2; idx_sub++) {
                        int _field_idx = kernel_idx * 2 + idx_sub;
                        domain_->decomposition()
                            .client()
                            ->template communicate_updownward_assign_OMP<source_tmp_type,
                                source_tmp_type>(l, false, false, -1, _field_idx);

                        domain_->decomposition()
                            .client()
                            ->template communicate_updownward_assign_OMP<target_tmp_type,
                                target_tmp_type>(l+1, false, false, -1, _field_idx);

                        for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                        {
                            if (!it->has_data() || !it->data().is_allocated())
                                continue;

                            const bool correction_buffer_only = true;
                            c_cntr_nli_vec[omp_idx]->template nli_intrp_node<source_tmp_type, correction_tmp_type>(
                                    it, Source::mesh_type(), _field_idx + starting_idx, _field_idx,
                                    correction_buffer_only, false);
                        }

                        for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                        {
                            int    refinement_level = it->refinement_level();
                            double dx = dx_base / std::pow(2, refinement_level);
                            if (kernel_idx == 0) {
                                c_cntr_nli_vec[omp_idx]->template add_source_correction<target_tmp_type,
                                correction_tmp_type>(_field_idx, it, dx / 2.0);
                            }
                            else {
                                float_type c_val = _kernel[kernel_idx - 1]->return_c_level() / dx;
                                c_cntr_nli_vec[omp_idx]->template add_source_correction<target_tmp_type,
                                correction_tmp_type>(_field_idx, it, dx / 2.0, c_val);
                            }
                        }

                        for (auto it = domain_->begin(l + 1); it != domain_->end(l + 1);
                            ++it) {
                            if (it->locally_owned())
                            {
                                auto& lin_data_1 =
                                    it->data_r(correction_tmp, _field_idx).linalg_data();
                                auto& lin_data_2 =
                                    it->data_r(source_tmp, _field_idx).linalg_data();

                                xt::noalias(lin_data_2) += lin_data_1 * 1.0;
                            }
                        }
                    }
                }
            }

            /*clock_gettime(CLOCK_REALTIME,&t_corr);
            double dt_corr = (t_corr.tv_sec+t_corr.tv_nsec/1e9)-(t_apply.tv_sec+t_apply.tv_nsec/1e9);
            if (domain_->client_communicator().rank() == 0) {
                std::cout << "l = " << l << " dt_corr = " << dt_corr << std::endl;
            }*/

        }
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


    template<class field>
    void clean_field(int field_idx)
    {
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->has_data() || !it->data().is_allocated()) continue;

            auto& lin_data = it->data_r(field::tag(), field_idx).linalg_data();
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
        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
             ++it)
        {
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

                if (Dim == 3)
                {
                    xt::noalias(view(lin_data_2, xt::range(1, -1),
                        xt::range(1, -1), xt::range(1, -1))) = view(lin_data_1,
                        xt::range(1, -1), xt::range(1, -1), xt::range(1, -1));
                }
                else
                {
                    xt::noalias(
                        view(lin_data_2, xt::range(1, -1), xt::range(1, -1))) =
                        view(lin_data_1, xt::range(1, -1), xt::range(1, -1));
                }
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
            int ref_level_up = domain_->tree()->depth() - l - 1;
            if (adapt_Fourier)
            {
                int res_modes = real_mesh_field_idx % (2 * N_fourier_modes + 2);

                int divisor = std::pow(2, ref_level_up);

                int N_comp_modes = (additional_modes + 1) * 2 / divisor;
                if (res_modes >= N_comp_modes) continue;
            }

            client->template buffer_exchange<From>(l);

            domain_->decomposition()
                .client()
                ->template communicate_updownward_assign<From, From>(l, false,
                    false, -1, tmp_type_field_idx, leaf_boundary);

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
    void intrp_to_correction_buffer_all_comp(MeshObject mesh_type,
        bool correction_only = true, bool exclude_correction = false,
        bool leaf_boundary = false)
    {
        auto client = domain_->decomposition().client();
        if (!client) return;

        for (int l = domain_->tree()->depth() - 2;
             l >= domain_->tree()->base_level(); --l)
        {
            client->template buffer_exchange<From>(l);

            int ref_level_up = domain_->tree()->depth() - l - 1;

            for (std::size_t _field_idx = 0; _field_idx < From::nFields();
                 ++_field_idx)
            {
                domain_->decomposition()
                    .client()
                    ->template communicate_updownward_assign<From, From>(l,
                        false, false, -1, _field_idx, leaf_boundary);
            }

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->has_data() || !it->data().is_allocated()) continue;
                if (leaf_boundary && !it->leaf_boundary()) continue;

                for (std::size_t _field_idx = 0; _field_idx < From::nFields();
                     ++_field_idx)
                {
                    if (adapt_Fourier)
                    {
                        int NComp = From::nFields() / (2 * N_fourier_modes + 2);
                        int res = From::nFields() % (2 * N_fourier_modes + 2);
                        if (res != 0)
                        {
                            throw std::runtime_error(
                                "nFields in intrp to correction buffer not a multiple of N_modes*2");
                        }

                        int res_modes = _field_idx % (2 * N_fourier_modes + 2);

                        int divisor = std::pow(2, ref_level_up);

                        int N_comp_modes = (additional_modes + 1) * 2 / divisor;
                        if (res_modes >= N_comp_modes) continue;
                    }
                    c_cntr_nli_.template nli_intrp_node<From, To>(it, mesh_type,
                        _field_idx, _field_idx, correction_only,
                        exclude_correction);
                }
            }
        }
    }



    template<class From, class To>
    void intrp_to_correction_buffer_OMP(MeshObject mesh_type, int starting_idx, 
        bool correction_only = true, bool exclude_correction = false,
        bool leaf_boundary = false)
    {
        auto client = domain_->decomposition().client();
        if (!client) return;

        for (int l = domain_->tree()->depth() - 2;
             l >= domain_->tree()->base_level(); --l)
        {
            client->template buffer_exchange<From>(l);

            int ref_level_up = domain_->tree()->depth() - l - 1;

            /*for (std::size_t _field_idx = 0; _field_idx < From::nFields();
                 ++_field_idx)
            {
                domain_->decomposition()
                    .client()
                    ->template communicate_updownward_assign<From, From>(l,
                        false, false, -1, _field_idx, leaf_boundary);
            }*/
            #pragma omp parallel for 
            for (std::size_t _field_idx = 0; _field_idx < From::nFields();
                     ++_field_idx)
            {
                int omp_idx = omp_get_thread_num( );
                domain_->decomposition()
                    .client()
                    ->template communicate_updownward_assign_OMP<From, From>(l,
                        false, false, -1, _field_idx, leaf_boundary);

                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->has_data() || !it->data().is_allocated()) continue;
                    if (leaf_boundary && !it->leaf_boundary()) continue;

                
                    if (adapt_Fourier)
                    {
                        int NComp = From::nFields() / (2 * N_fourier_modes + 2);
                        int res = From::nFields() % (2 * N_fourier_modes + 2);
                        if (res != 0)
                        {
                            throw std::runtime_error(
                                "nFields in intrp to correction buffer not a multiple of N_modes*2");
                        }

                        int res_modes = _field_idx % (2 * N_fourier_modes + 2);

                        int divisor = std::pow(2, ref_level_up);

                        int N_comp_modes = (additional_modes + 1) * 2 / divisor;
                        if (res_modes >= N_comp_modes) continue;
                    }
                    c_cntr_nli_vec[omp_idx]->template nli_intrp_node<From, To>(it, mesh_type,
                        _field_idx + starting_idx, _field_idx, correction_only,
                        exclude_correction);
                
                }
            }
        }
    }

    template<class From, class To>
    void source_coarsify(std::size_t real_mesh_field_idx,
        std::size_t tmp_type_field_idx, MeshObject mesh_type,
        bool correction_only = false, bool exclude_correction = false,
        bool _buffer_exchange = false, bool leaf_boundary = false)
    {
        leaf_boundary = false;
        auto client = domain_->decomposition().client();
        if (!client) return;

        for (int ls = domain_->tree()->depth() - 2;
             ls >= domain_->tree()->base_level(); --ls)
        {
            int ref_level_up = domain_->tree()->depth() - ls - 1;

            if (adapt_Fourier)
            {
                int res_modes = real_mesh_field_idx % (2 * N_fourier_modes + 2);

                int divisor = std::pow(2, ref_level_up);

                int N_comp_modes = (additional_modes + 1) * 2 / divisor;
                if (res_modes >= N_comp_modes) continue;
            }

            for (auto it_s = domain_->begin(ls); it_s != domain_->end(ls);
                 ++it_s)
            {
                if (!it_s->has_data() || !it_s->data().is_allocated()) continue;
                if (leaf_boundary && !it_s->leaf_boundary()) continue;

                c_cntr_nli_.template nli_antrp_node<From, To>(*it_s, mesh_type,
                    real_mesh_field_idx, tmp_type_field_idx, correction_only,
                    exclude_correction);
            }

            domain_->decomposition()
                .client()
                ->template communicate_updownward_add<To, To>(ls, true, false,
                    -1, tmp_type_field_idx, leaf_boundary);
        }

        if (_buffer_exchange)
        {
            for (int l = domain_->tree()->base_level();
                 l < domain_->tree()->depth() - 1; ++l)
                client->template buffer_exchange<To>(l);
        }
    }

    template<class From, class To>
    void source_coarsify_all_comp(MeshObject mesh_type,
        bool correction_only = false, bool exclude_correction = false,
        bool _buffer_exchange = false, bool leaf_boundary = false)
    {

        leaf_boundary=false;
        auto client = domain_->decomposition().client();
        if (!client) return;

        for (int l = domain_->tree()->depth() - 2;
             l >= domain_->tree()->base_level(); --l)
        {
            int ref_level_up = domain_->tree()->depth() - l - 1;

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->has_data() || !it->data().is_allocated()) continue;
                if (leaf_boundary && !it->leaf_boundary()) continue;

                for (std::size_t _field_idx = 0; _field_idx < From::nFields();
                     ++_field_idx)
                {
                    if (adapt_Fourier)
                    {
                        int NComp = From::nFields() / (2 * N_fourier_modes + 2);
                        int res = From::nFields() % (2 * N_fourier_modes + 2);
                        if (res != 0)
                        {
                            throw std::runtime_error(
                                "nFields in source corasify not a multiple of N_modes*2");
                        }

                        int res_modes = _field_idx % (2 * N_fourier_modes + 2);

                        int divisor = std::pow(2, ref_level_up);

                        int N_comp_modes = (additional_modes + 1) * 2 / divisor;
                        if (res_modes >= N_comp_modes) continue;
                    }
                    c_cntr_nli_.template nli_antrp_node<From, To>(*it, mesh_type,
                    _field_idx, _field_idx, correction_only,
                    exclude_correction);
                }
            }

            for (std::size_t _field_idx = 0; _field_idx < From::nFields();
                 ++_field_idx)
            {
                domain_->decomposition()
                    .client()
                    ->template communicate_updownward_add<To, To>(
                    l, true, false, -1, _field_idx, leaf_boundary);
            }
        }

        if (_buffer_exchange)
        {
            for (int l = domain_->tree()->base_level();
                 l < domain_->tree()->depth() - 1; ++l)
                client->template buffer_exchange<To>(l);
        }
    }


    template<class From, class To>
    void source_coarsify_OMP(MeshObject mesh_type, std::size_t starting_idx,
        bool correction_only = false, bool exclude_correction = false,
        bool _buffer_exchange = false, bool leaf_boundary = false)
    {

        leaf_boundary=false;
        auto client = domain_->decomposition().client();
        if (!client) return;

        for (int l = domain_->tree()->depth() - 2;
             l >= domain_->tree()->base_level(); --l)
        {
            int ref_level_up = domain_->tree()->depth() - l - 1;

            #pragma omp parallel for 
            for (std::size_t _field_idx = 0; _field_idx < From::nFields();
                     ++_field_idx)
            {
                int omp_idx = omp_get_thread_num( );

                if (adapt_Fourier)
                {
                    int NComp = From::nFields() / (2 * N_fourier_modes + 2);
                    int res = From::nFields() % (2 * N_fourier_modes + 2);
                    if (res != 0)
                    {
                        throw std::runtime_error(
                            "nFields in source corasify not a multiple of N_modes*2");
                    }

                    int res_modes = _field_idx % (2 * N_fourier_modes + 2);

                    int divisor = std::pow(2, ref_level_up);

                    int N_comp_modes = (additional_modes + 1) * 2 / divisor;
                    if (res_modes >= N_comp_modes) continue;
                }
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->has_data() || !it->data().is_allocated()) continue;
                    if (leaf_boundary && !it->leaf_boundary()) continue;

                    
                    c_cntr_nli_vec[omp_idx]->template nli_antrp_node<From, To>(*it, mesh_type,
                    _field_idx + starting_idx, _field_idx, correction_only,
                    exclude_correction);
                }
                domain_->decomposition()
                    .client()
                    ->template communicate_updownward_add_OMP<To, To>(
                    l, true, false, -1, _field_idx, leaf_boundary);
            }

            /*#pragma omp master 
            for (std::size_t _field_idx = 0; _field_idx < From::nFields();
                 ++_field_idx)
            {
                domain_->decomposition()
                    .client()
                    ->template communicate_updownward_add<To, To>(
                    l, true, false, -1, _field_idx, leaf_boundary);
            }*/
        }

        if (_buffer_exchange)
        {
            for (int l = domain_->tree()->base_level();
                 l < domain_->tree()->depth() - 1; ++l)
                client->template buffer_exchange<To>(l);
        }
    }

    auto& c_cntr_nli() { return c_cntr_nli_; }
    auto& c_cntr_nli_vec_(int i) { return c_cntr_nli_vec[i]; }

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
    std::vector<float_type>           Omegas;
    std::vector<helm_t>               lgf_helm_vec;
    std::vector<std::unique_ptr<helm_t>>               lgf_helm_vec_ptr;
    std::vector<std::unique_ptr<lgf_if_t>>               lgf_if_vec;
    int                               N_fourier_modes;
    interpolation_type                c_cntr_nli_;///< Lagrange Interpolation
    std::vector<std::unique_ptr<interpolation_type>>                c_cntr_nli_vec;
    parallel_ostream::ParallelOstream pcout=parallel_ostream::ParallelOstream(1);
    bool use_correction_ =true;
    bool subtract_non_leaf_ = false;

    int additional_modes = 0;
    bool adapt_Fourier = false;
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
