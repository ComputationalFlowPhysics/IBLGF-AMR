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

#ifndef IBLGF_INCLUDED_FMM_OMP
#define IBLGF_INCLUDED_FMM_OMP

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <omp.h>

// IBLGF-specific

#include <algorithm>
#include <list>
#include <vector>
#include <iostream>
#include <cmath>
#include <functional>
#include <cstring>
#include <fftw3.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include <iblgf/types.hpp>
#include <iblgf/global.hpp>
#include <iblgf/linalg/linalg.hpp>
#include <iblgf/fmm/fmm_nli.hpp>
#include <iblgf/IO/parallel_ostream.hpp>
#include <iblgf/utilities/convolution.hpp>

namespace iblgf
{
namespace fmm
{

using namespace domain;

template<class Setup>
class Fmm_OMP
{
  public: //Ctor:
    static constexpr std::size_t Dim = Setup::Dim;

    using dims_t = types::vector_type<int, Dim>;
    using datablock_t = typename Setup::datablock_t;
    using block_dsrp_t = typename datablock_t::block_descriptor_type;
    using domain_t = typename Setup::domain_t;
    using octant_t = typename domain_t::octant_t;
    using MASK_LIST = typename octant_t::MASK_LIST;
    using MASK_TYPE = typename octant_t::MASK_TYPE;

    //Fields:
    using fmm_s_type = typename Setup::fmm_s_type;
    using fmm_t_type = typename Setup::fmm_t_type;

    static constexpr auto fmm_s = Setup::fmm_s;
    static constexpr auto fmm_t = Setup::fmm_t;

    using convolution_t = fft::Convolution<Dim>;

  public:
    Fmm_OMP(domain_t* _domain, int Nb)
    : domain(_domain)
    , lagrange_intrp(Nb)
    {
        //pcout << "starting Ctor " << fmm_t_type::nFields() << " " << Nb << std::endl;
        for (int i = 0; i < fmm_t_type::nFields(); i++) {
            //pcout << i << std::endl;
            convs_.emplace_back((new convolution_t(dims_t(Nb), dims_t(Nb))));
            lagrange_intrp_vec.emplace_back((new Nli(Nb)));
        }
        //pcout << "finished Ctor" << std::endl;
    }

    template<class Source, class Target, class Kernel_vec>
    void apply(domain_t* domain_, Kernel_vec& _kernel, int level,
        bool non_leaf_as_source, float_type add_with_scale = 1.0,
        int fmm_type = MASK_TYPE::AMR2AMR)
    {
        //use LGF_range as the range of field_idx using kernel_lgf
        //use vec_itv as the number of field_idx using the same kernel
        //i.e. _kernel[k] is used is field_idx [LGF_range + k*vec_itv, LGF_range + k*vec_itv + vec_itv)
        const float_type dx_base = domain_->dx_base();
        auto refinement_level = level - domain_->tree()->base_level();
        auto dx_level = dx_base / std::pow(2, refinement_level);

        base_level_ = level;
        if (fmm_type == MASK_TYPE::STREAM)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::STREAM);
        else if (fmm_type == MASK_TYPE::AMR2AMR)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::AMR2AMR, refinement_level, non_leaf_as_source);
        else if (fmm_type == MASK_TYPE::IB2xIB)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::IB2xIB);
        else if (fmm_type == MASK_TYPE::xIB2IB)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::xIB2IB);
        else if (fmm_type == MASK_TYPE::IB2AMR)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::IB2AMR, refinement_level);

        if (_kernel[0]->neighbor_only())
        {

            sort_bx_octants(domain, _kernel[0].get());

            #pragma omp parallel for
            for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {

                fmm_init_zero<fmm_s_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Source);
                fmm_init_zero<fmm_t_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Target);


                fmm_init_copy<Source, fmm_s_type>(field_idx, domain_);


            }

            #pragma omp parallel for
            for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {

                float_type scale = (_kernel[field_idx]->neighbor_only()) ? 1.0 : dx_level;
                pcout<<"fmm_Bx start" << std::endl;
                //domain_->client_communicator().barrier();
                fmm_Bx_local(field_idx, domain_, _kernel[field_idx].get(), scale);

            /*}

            #pragma omp master
            for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {        
                float_type scale = (_kernel[field_idx]->neighbor_only()) ? 1.0 : dx_level;*/
                fmm_Bx_global(field_idx, domain_, _kernel[field_idx].get(), scale);
            }

            #pragma omp parallel for
            for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {        
                fmm_add_equal<Target, fmm_t_type>(field_idx, domain_, add_with_scale);
            }

            return;
        }

        //pcout<<"FMM For Level "<< level << " Start ---------------------------"<<std::endl;
        // New logic using masks
        // 1. Give masks to the base level

        //// The straight forward way:
        // 1.1 if non_leaf_as_source is true then
        // non-leaf's mask_fmm_source is 1 and non-leaf's mask_fmm_target is 0,
        // while leaf's mask_fmm_source is 1 and leaf's mask_fmm_target is 1

        // 1.2 if non_leaf_as_source is false then
        // non-leaf's mask_fmm_source is 0 and non-leaf's mask_fmm_target is 1,
        // while leaf's mask_fmm_source is 1 and leaf's mask_fmm_target is 1

        //// !However! we do it a bit differently !
        // 1.1 if non_leaf_as_source is true then
        // non-leaf's mask_fmm_source is 1 and non-leaf's mask_fmm_target is 1,
        // while leaf's mask_fmm_source is 0 and leaf's mask_fmm_target is 0
        // BUT WITH MINUS SIGN!!!!!

        // 1.2 if non_leaf_as_source is false then
        // non-leaf's mask_fmm_source is 1 and non-leaf's mask_fmm_target is 1,
        // while leaf's mask_fmm_source is 1 and leaf's mask_fmm_target is 1
        // BUT WITH PLUS SIGN!!!!!

        // This way one has one time more calculaion for non-leaf's source but
        // one time fewer for leaf's leaf_source
        // and non-leaves can be much fewer

        //// Initialize Masks
        // done at master node

        sort_bx_octants(domain, _kernel[0].get());

        ////Initialize for each fmm//zero ing all tree
        #pragma omp parallel for
        for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {
            fmm_init_zero<fmm_s_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Source);
            fmm_init_zero<fmm_t_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Target);

            fmm_init_copy<Source, fmm_s_type>(field_idx, domain_);

#ifdef POISSON_TIMINGS
            timings_ = Timings();
#endif
        }
        
        // Anterpolation
#ifdef POISSON_TIMINGS
        auto t0_anterp = clock_type::now();
#endif
        fmm_antrp(domain_);
#ifdef POISSON_TIMINGS
        auto t1_anterp = clock_type::now();
        timings_.anterp = t1_anterp - t0_anterp;
#endif

        #pragma omp parallel for
        for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {

#ifdef POISSON_TIMINGS
            auto t0_bx = clock_type::now();
#endif
            float_type scale = (_kernel[field_idx]->neighbor_only()) ? 1.0 : dx_level;
            pcout<<"fmm_Bx start" << std::endl;
            fmm_Bx_local(field_idx, domain_, _kernel[field_idx].get(), scale);
#ifdef POISSON_TIMINGS
            auto t1_bx = clock_type::now();
            timings_.bx = t1_bx - t0_bx;
#endif
        /*}

        #pragma omp master
        for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {        
            float_type scale = (_kernel[field_idx]->neighbor_only()) ? 1.0 : dx_level;*/
            fmm_Bx_global(field_idx, domain_, _kernel[field_idx].get(), scale);
        }


#ifdef POISSON_TIMINGS
        auto t0_interp = clock_type::now();
#endif
        // Interpolation
        pcout<<"FMM INTRP start" << std::endl;
        fmm_intrp(domain_);
        pcout<<"FMM INTRP done" << std::endl;

#ifdef POISSON_TIMINGS
        auto t1_interp = clock_type::now();
        timings_.interp = t1_interp - t0_interp;
#endif

        #pragma omp parallel for
        for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {
            // Copy back
            fmm_add_equal<Target, fmm_t_type>(field_idx, domain_, add_with_scale);
        }
    }


    template<class Source, class Target, class Kernel_vec>
    void apply_IF(domain_t* domain_, Kernel_vec& _kernel, int level,
        bool non_leaf_as_source, std::vector<float_type> add_with_scale,
        int fmm_type = MASK_TYPE::AMR2AMR)
    {
        const float_type dx_base = domain_->dx_base();
        auto refinement_level = level - domain_->tree()->base_level();
        auto dx_level = dx_base / std::pow(2, refinement_level);

        base_level_ = level;
        if (fmm_type == MASK_TYPE::STREAM)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::STREAM);
        else if (fmm_type == MASK_TYPE::AMR2AMR)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::AMR2AMR, refinement_level, non_leaf_as_source);
        else if (fmm_type == MASK_TYPE::IB2xIB)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::IB2xIB);
        else if (fmm_type == MASK_TYPE::xIB2IB)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::xIB2IB);
        else if (fmm_type == MASK_TYPE::IB2AMR)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::IB2AMR, refinement_level);

        

        sort_bx_octants(domain, _kernel[0].get());

        #pragma omp parallel for
        for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {

            fmm_init_zero<fmm_s_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Source);
            fmm_init_zero<fmm_t_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Target);
            fmm_init_copy<Source, fmm_s_type>(field_idx, domain_);
        /*}

        #pragma omp parallel for
        for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {*/

            float_type scale = 1.0;
            fmm_Bx_local(field_idx, domain_, _kernel[field_idx].get(), scale);

        /*}

        #pragma omp master
        for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {        
            float_type scale = 1.0;*/
            fmm_Bx_global(field_idx, domain_, _kernel[field_idx].get(), scale);
        /*}

        #pragma omp parallel for
        for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {*/        
            fmm_add_equal<Target, fmm_t_type>(field_idx, domain_, add_with_scale[field_idx]);
        }

        return;
    }



    template<class Source, class Target, class Kernel_lgf, class Kernel_vec>
    void apply_LGF(domain_t* domain_, Kernel_lgf* kernel_lgf, int LGF_range, Kernel_vec& _kernel, int vec_itv, int level,
        bool non_leaf_as_source, float_type add_with_scale = 1.0,
        int fmm_type = MASK_TYPE::AMR2AMR)
    {
        //use LGF_range as the range of field_idx using kernel_lgf
        //use vec_itv as the number of field_idx using the same kernel
        //i.e. _kernel[k] is used is field_idx [LGF_range + k*vec_itv, LGF_range + k*vec_itv + vec_itv)
        if ((LGF_range + _kernel.size() * vec_itv) != fmm_s_type::nFields()) {
            throw std::runtime_error("Number of fields does not match in apply_LGF");
        }
        const float_type dx_base = domain_->dx_base();
        auto refinement_level = level - domain_->tree()->base_level();
        auto dx_level = dx_base / std::pow(2, refinement_level);

        base_level_ = level;
        if (fmm_type == MASK_TYPE::STREAM)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::STREAM);
        else if (fmm_type == MASK_TYPE::AMR2AMR)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::AMR2AMR, refinement_level, non_leaf_as_source);
        else if (fmm_type == MASK_TYPE::IB2xIB)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::IB2xIB);
        else if (fmm_type == MASK_TYPE::xIB2IB)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::xIB2IB);
        else if (fmm_type == MASK_TYPE::IB2AMR)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::IB2AMR, refinement_level);

        // New logic using masks
        // 1. Give masks to the base level

        //// The straight forward way:
        // 1.1 if non_leaf_as_source is true then
        // non-leaf's mask_fmm_source is 1 and non-leaf's mask_fmm_target is 0,
        // while leaf's mask_fmm_source is 1 and leaf's mask_fmm_target is 1

        // 1.2 if non_leaf_as_source is false then
        // non-leaf's mask_fmm_source is 0 and non-leaf's mask_fmm_target is 1,
        // while leaf's mask_fmm_source is 1 and leaf's mask_fmm_target is 1

        //// !However! we do it a bit differently !
        // 1.1 if non_leaf_as_source is true then
        // non-leaf's mask_fmm_source is 1 and non-leaf's mask_fmm_target is 1,
        // while leaf's mask_fmm_source is 0 and leaf's mask_fmm_target is 0
        // BUT WITH MINUS SIGN!!!!!

        // 1.2 if non_leaf_as_source is false then
        // non-leaf's mask_fmm_source is 1 and non-leaf's mask_fmm_target is 1,
        // while leaf's mask_fmm_source is 1 and leaf's mask_fmm_target is 1
        // BUT WITH PLUS SIGN!!!!!

        // This way one has one time more calculaion for non-leaf's source but
        // one time fewer for leaf's leaf_source
        // and non-leaves can be much fewer

        //// Initialize Masks
        // done at master node

        sort_bx_octants(domain, kernel_lgf);

        ////Initialize for each fmm//zero ing all tree
        /*#pragma omp parallel for
        for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {
            fmm_init_zero<fmm_s_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Source);
            fmm_init_zero<fmm_t_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Target);

            fmm_init_copy<Source, fmm_s_type>(field_idx, domain_);


        }
        
        // Anterpolation

        fmm_antrp(domain_);*/

        float_type scale = dx_level;

        #pragma omp parallel for
        for (int kernel_idx = 0; kernel_idx < (_kernel.size() + 1); kernel_idx++) {

            if (kernel_idx == 0) {
                for (int inner_range = 0; inner_range < LGF_range; inner_range++) {
                    
                    int field_idx = inner_range;

                    fmm_init_zero<fmm_s_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Source);
                    fmm_init_zero<fmm_t_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Target);
                    fmm_init_copy<Source, fmm_s_type>(field_idx, domain_);

                    fmm_antrp(field_idx, domain_);

                    fmm_Bx_local(inner_range, domain_, kernel_lgf, scale);
                    fmm_Bx_global(inner_range, domain_, kernel_lgf, scale);

                    fmm_intrp(field_idx, domain_);

                    fmm_add_equal<Target, fmm_t_type>(field_idx, domain_, add_with_scale);
                }
            }
            else {
                for (int inner_range = 0; inner_range < vec_itv; inner_range++) {
                    int field_idx = inner_range + (kernel_idx - 1) * vec_itv + LGF_range;

                    fmm_init_zero<fmm_s_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Source);
                    fmm_init_zero<fmm_t_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Target);
                    fmm_init_copy<Source, fmm_s_type>(field_idx, domain_);

                    fmm_antrp(field_idx, domain_);

                    fmm_Bx_local(field_idx, domain_, _kernel[kernel_idx - 1].get(), scale);
                    fmm_Bx_global(field_idx, domain_, _kernel[kernel_idx - 1].get(), scale);

                    fmm_intrp(field_idx, domain_);

                    fmm_add_equal<Target, fmm_t_type>(field_idx, domain_, add_with_scale);
                }
            }

            /*if (kernel_idx == 0) {
                for (int inner_range = 0; inner_range < LGF_range; inner_range++) {
                    fmm_Bx_global(inner_range, domain_, kernel_lgf, scale);
                }
            }
            else {
                for (int inner_range = 0; inner_range < vec_itv; inner_range++) {
                    int field_idx = inner_range + (kernel_idx - 1) * vec_itv + LGF_range;
                    fmm_Bx_global(field_idx, domain_, _kernel[kernel_idx - 1].get(), scale);
                }
            }*/
        }



        // Interpolation
        /*fmm_intrp(domain_);

        #pragma omp parallel for
        for (int field_idx = 0; field_idx < fmm_s_type::nFields(); field_idx++) {
            // Copy back
            fmm_add_equal<Target, fmm_t_type>(field_idx, domain_, add_with_scale);
        }*/
    }


    template<class Source, class Target, class Kernel>
    void apply_LGF_single(domain_t* domain_, Kernel* kernel, int kernel_idx, int level,
        bool non_leaf_as_source, float_type add_with_scale = 1.0,
        int fmm_type = MASK_TYPE::AMR2AMR)
    {
        //this solves only a single index of FMM, it is specially designed for IB solver 
        //one thread can only access a single kernel to ensure thread safety
        
        const float_type dx_base = domain_->dx_base();
        auto refinement_level = level - domain_->tree()->base_level();
        auto dx_level = dx_base / std::pow(2, refinement_level);

        /*base_level_ = level;
        if (fmm_type == MASK_TYPE::STREAM)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::STREAM);
        else if (fmm_type == MASK_TYPE::AMR2AMR)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::AMR2AMR, refinement_level, non_leaf_as_source);
        else if (fmm_type == MASK_TYPE::IB2xIB)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::IB2xIB);
        else if (fmm_type == MASK_TYPE::xIB2IB)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::xIB2IB);
        else if (fmm_type == MASK_TYPE::IB2AMR)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::IB2AMR, refinement_level);*/

        

        //sort_bx_octants(domain, kernel);

        float_type scale = dx_level;


        for (int inner_range = 0; inner_range < 2; inner_range++) {
                
            int field_idx = inner_range + (kernel_idx) * 2;

            fmm_init_zero<fmm_s_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Source);
            fmm_init_zero<fmm_t_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Target);
            fmm_init_copy<Source, fmm_s_type>(field_idx, domain_);

            fmm_antrp(field_idx, domain_);

            fmm_Bx_local(field_idx, domain_, kernel, scale);
            fmm_Bx_global(field_idx, domain_, kernel, scale);

            fmm_intrp(field_idx, domain_);

            fmm_add_equal<Target, fmm_t_type>(field_idx, domain_, add_with_scale);
        }
    }

    template<class Source, class Target, class Kernel>
    void apply_single_master(domain_t* domain_, Kernel* kernel, int level,
        bool non_leaf_as_source, int fmm_type = MASK_TYPE::AMR2AMR)
    {
        const float_type dx_base = domain_->dx_base();
        auto refinement_level = level - domain_->tree()->base_level();
        auto dx_level = dx_base / std::pow(2, refinement_level);
        base_level_ = level;
        if (fmm_type == MASK_TYPE::STREAM)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::STREAM);
        else if (fmm_type == MASK_TYPE::AMR2AMR)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::AMR2AMR, refinement_level, non_leaf_as_source);
        else if (fmm_type == MASK_TYPE::IB2xIB)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::IB2xIB);
        else if (fmm_type == MASK_TYPE::xIB2IB)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::xIB2IB);
        else if (fmm_type == MASK_TYPE::IB2AMR)
            fmm_mask_idx_ = octant_t::fmm_mask_idx_gen(MASK_TYPE::IB2AMR, refinement_level);

        sort_bx_octants(domain, kernel);
    }


    template<class Source, class Target, class Kernel>
    void apply_IF_single(domain_t* domain_, Kernel* kernel, int field_idx, int level,
        bool non_leaf_as_source, float_type add_with_scale,
        int fmm_type = MASK_TYPE::AMR2AMR)
    {
        //this solves only a single index of FMM, it is specially designed for IB solver 
        //now every single thread can access one field_idx worth of data
        const float_type dx_base = domain_->dx_base();
        auto refinement_level = level - domain_->tree()->base_level();
        auto dx_level = dx_base / std::pow(2, refinement_level);
        

        

        //sort_bx_octants(domain, kernel);

        fmm_init_zero<fmm_s_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Source);
        fmm_init_zero<fmm_t_type>(field_idx, domain_, MASK_LIST::Mask_FMM_Target);
        fmm_init_copy<Source, fmm_s_type>(field_idx, domain_);
    

        float_type scale = 1.0;
        fmm_Bx_local(field_idx, domain_, kernel, scale);

        fmm_Bx_global(field_idx, domain_, kernel, scale);
    
        fmm_add_equal<Target, fmm_t_type>(field_idx, domain_, add_with_scale);
        
        return;
    }

    template<class Kernel>
    void sort_bx_octants(domain_t* domain_, Kernel* _kernel)
    {
        sorted_octants_.clear();

        const int level_max = (_kernel->neighbor_only()) ? base_level_ : 0;

        for (int level = base_level_; level >= level_max; --level)
        {
            for (auto it = domain_->begin(level); it != domain_->end(level);
                 ++it)
            {
                bool _neighbor = (level == base_level_) ? true : false;
                if (!(it->has_data()) ||
                    !it->fmm_mask(fmm_mask_idx_, MASK_LIST::Mask_FMM_Target))
                    continue;

                int recv_m_send_count =
                    domain_->decomposition()
                        .client()
                        ->template communicate_induced_fields_recv_m_send_count<
                            fmm_t_type, fmm_t_type>(
                            it, _kernel, _neighbor, fmm_mask_idx_);

                sorted_octants_.emplace_back(
                    std::make_pair(*it, recv_m_send_count));
            }
        }

        std::sort(sorted_octants_.begin(), sorted_octants_.end(),
            [&](const auto& e0, const auto& e1) {
                return e0.second > e1.second;
            });
    }

    template<class Kernel>
    void fmm_Bx(int field_idx, domain_t* domain_, Kernel* _kernel, float_type scale)
    {
        const bool start_communication = true;

        for (auto B_it = sorted_octants_.begin(); B_it != sorted_octants_.end();
                ++B_it)
        {
            auto       it = B_it->first;
            const int  level = it->level();
            const bool _neighbor = (level == base_level_) ? true : false;

            if (it->locally_owned())
                compute_influence_field(field_idx, &(*it), _kernel, base_level_ - level, scale, _neighbor);
        }
        #pragma omp ordered
        {
        for (auto B_it = sorted_octants_.begin(); B_it != sorted_octants_.end();
                ++B_it)
        {
            auto       it = B_it->first;
            const int  level = it->level();
            const bool _neighbor = (level == base_level_) ? true : false;

            if (B_it->second != 0)
            {
                domain_->decomposition()
                .client()
                ->template communicate_induced_fields_idx<fmm_t_type,
                fmm_t_type>(field_idx, &(*it), this, _kernel, base_level_ - level,
                        scale, _neighbor, start_communication, fmm_mask_idx_);
            }
        }
        

        domain_->decomposition().client()->finish_induced_field_communication();
        }
    }

    template<class Kernel>
    void fmm_Bx_local(int field_idx, domain_t* domain_, Kernel* _kernel, float_type scale)
    {
        const bool start_communication = true;

        for (auto B_it = sorted_octants_.begin(); B_it != sorted_octants_.end();
                ++B_it)
        {
            auto       it = B_it->first;
            const int  level = it->level();
            const bool _neighbor = (level == base_level_) ? true : false;

            if (it->locally_owned())
                compute_influence_field(field_idx, &(*it), _kernel, base_level_ - level, scale, _neighbor);
        }
    }

    template<class Kernel>
    void fmm_Bx_global(int field_idx, domain_t* domain_, Kernel* _kernel, float_type scale)
    {
        const bool start_communication = true;

        for (auto B_it = sorted_octants_.begin(); B_it != sorted_octants_.end();
                ++B_it)
        {
            auto       it = B_it->first;
            const int  level = it->level();
            const bool _neighbor = (level == base_level_) ? true : false;

            if (B_it->second != 0)
            {
                domain_->decomposition()
                .client()
                ->template communicate_induced_fields_idx<fmm_t_type,
                fmm_t_type>(field_idx, &(*it), this, _kernel, base_level_ - level,
                        scale, _neighbor, start_communication, fmm_mask_idx_);
            }
        }
        

        domain_->decomposition().client()->finish_induced_field_communication_idx();
    }


    template<class Kernel>
    void compute_influence_field(int field_idx, octant_t* it, Kernel* _kernel, int level_diff,
        float_type dx_level, bool neighbor) noexcept
    {
        if (!(it->has_data()) ||
            !it->fmm_mask(fmm_mask_idx_, MASK_LIST::Mask_FMM_Target))
            return;

        convs_[field_idx]->fft_backward_field_clean();

        if (neighbor)
        {
            for (int i = 0; i < it->nNeighbors(); ++i)
            {
                auto n_s = it->neighbor(i);
                if (n_s && n_s->locally_owned() &&
                    n_s->fmm_mask(fmm_mask_idx_, MASK_LIST::Mask_FMM_Source))
                { fmm_tt(field_idx, n_s, it, _kernel, 0); }
            }
        }

        if (!_kernel->neighbor_only())
        {
            for (int i = 0; i < it->influence_number(); ++i)
            {
                auto n_s = it->influence(i);
                if (n_s && n_s->locally_owned() &&
                    n_s->fmm_mask(fmm_mask_idx_, MASK_LIST::Mask_FMM_Source))
                { fmm_tt(field_idx, n_s, it, _kernel, level_diff); }
            }
        }

        const auto   t_extent = it->data_r(fmm_t).real_block().extent();
        block_dsrp_t extractor(dims_t(0), t_extent);

        float_type _scale =
            (_kernel->neighbor_only()) ? 1.0 : dx_level * dx_level;
        convs_[field_idx]->apply_backward(extractor, it->data_r(fmm_t, field_idx), _scale);
    }

    template<class f1, class f2>
    void fmm_add_equal(int field_idx, domain_t* domain_, float_type scale)
    {
        for (auto it = domain_->begin(base_level_);
             it != domain_->end(base_level_); ++it)
        {
            if (it->has_data() && it->locally_owned() &&
                it->fmm_mask(fmm_mask_idx_, MASK_LIST::Mask_FMM_Target))
            {
                it->data_r(f1::tag(), field_idx).linalg().get()->cube_noalias_view() +=
                    it->data_r(f2::tag(), field_idx).linalg_data() * scale;
            }
        }
    }

    template<class f1, class f2>
    void fmm_minus_equal(int field_idx, domain_t* domain_)
    {
        for (auto it = domain_->begin(base_level_);
             it != domain_->end(base_level_); ++it)
        {
            if (it->has_data() && it->locally_owned() &&
                it->fmm_mask(fmm_mask_idx_, MASK_LIST::Mask_FMM_Target))
            {
                it->data_r(f1::tag(), field_idx).linalg().get()->cube_noalias_view() -=
                    it->data_r(f2::tag(), field_idx).linalg_data();
            }
        }
    }

    template<class field>
    void fmm_init_zero(int field_idx, domain_t* domain_, int mask_id)
    {
        for (int level = base_level_; level >= 0; --level)
        {
            for (auto it = domain_->begin(level); it != domain_->end(level);
                 ++it)
            {
                if (it->has_data() && it->fmm_mask(fmm_mask_idx_, mask_id))
                {
                    for (auto& e : it->data_r(field::tag(), field_idx)) e = 0.0;
                }
            }
        }
    }

    template<class from, class to>
    void fmm_init_copy(int field_idx, domain_t* domain_)
    {
        // Neglecting the data in the buffer

        for (auto it = domain_->begin(base_level_);
             it != domain_->end(base_level_); ++it)
        {
            if (it->has_data() && it->locally_owned() &&
                it->fmm_mask(fmm_mask_idx_, MASK_LIST::Mask_FMM_Source))
            {
                auto lin_data_1 = it->data_r(from::tag(), field_idx).linalg_data();
                auto lin_data_2 = it->data_r(to::tag(), field_idx).linalg_data();

                if (Dim == 3) {
		xt::noalias(view(lin_data_2, xt::range(1, -1), xt::range(1, -1),
                    xt::range(1, -1))) = view(lin_data_1, xt::range(1, -1),
                    xt::range(1, -1), xt::range(1, -1));
		} 
		else {
		xt::noalias(view(lin_data_2, xt::range(1, -1), xt::range(1, -1))) 
			= view(lin_data_1, xt::range(1, -1),xt::range(1, -1));
		}
            }
        }
    }

    void fmm_intrp(int field_idx, domain_t* domain_)
    {
        const int mask_id = MASK_LIST::Mask_FMM_Target;
        for (int level = 1; level < base_level_; ++level)
        {
            domain_->decomposition()
                .client()
                ->template communicate_updownward_assign_OMP<fmm_t_type,
                    fmm_t_type>(level, false, true, fmm_mask_idx_, field_idx);

        
            for (auto it = domain_->begin(level); it != domain_->end(level);
                ++it)
            {
                if (it->has_data() && it->fmm_mask(fmm_mask_idx_, mask_id))
                    lagrange_intrp_vec[field_idx]->nli_intrp_node<fmm_t_type>(field_idx, 
                        it, mask_id, fmm_mask_idx_);
            }
        }
    }

    void fmm_intrp(domain_t* domain_)
    {
        const int mask_id = MASK_LIST::Mask_FMM_Target;
        #pragma omp parallel for 
        for (int field_idx = 0; field_idx < fmm_t_type::nFields(); field_idx++) {
            for (int level = 1; level < base_level_; ++level)
            {
            /*#pragma omp master 
            for (int field_idx = 0; field_idx < fmm_t_type::nFields(); field_idx++)
            {*/
                domain_->decomposition()
                    .client()
                    ->template communicate_updownward_assign_OMP<fmm_t_type,
                        fmm_t_type>(level, false, true, fmm_mask_idx_, field_idx);
            //}

            
                for (auto it = domain_->begin(level); it != domain_->end(level);
                    ++it)
                {
                    if (it->has_data() && it->fmm_mask(fmm_mask_idx_, mask_id))
                        lagrange_intrp_vec[field_idx]->nli_intrp_node<fmm_t_type>(field_idx, 
                            it, mask_id, fmm_mask_idx_);
                }
            }
        }
    }

    void fmm_antrp(domain_t* domain_)
    {
        const int mask_id = MASK_LIST::Mask_FMM_Source;
        #pragma omp parallel for 
        for (int field_idx = 0; field_idx < fmm_t_type::nFields(); field_idx++) {
            for (int level = base_level_ - 1; level >= 0; --level)
            {
            //#pragma omp barrier

            
                for (auto it = domain_->begin(level); it != domain_->end(level);
                    ++it)
                {
                    if (it->has_data() && it->fmm_mask(fmm_mask_idx_, mask_id))
                        lagrange_intrp_vec[field_idx]->nli_antrp_node<fmm_s_type>(field_idx,
                            it, mask_id, fmm_mask_idx_);
                }
            /*}

            #pragma omp master 
            for (int field_idx = 0; field_idx < fmm_t_type::nFields(); field_idx++) {*/
                domain_->decomposition()
                    .client()
                    ->template communicate_updownward_add_OMP<fmm_s_type, fmm_s_type>(
                        level, true, true, fmm_mask_idx_, field_idx);
            /*}

            #pragma omp parallel for 
            for (int field_idx = 0; field_idx < fmm_t_type::nFields(); field_idx++) {*/
                for (auto it = domain_->begin(level); it != domain_->end(level);
                    ++it)
                {
                    if (!it->locally_owned() && it->has_data() &&
                        it->data().is_allocated())
                    {
                        auto& cp2 = it->data_r(fmm_s, field_idx).linalg_data();
                        cp2 *= 0.0;
                    }
                }
            }
        }
    }

    void fmm_antrp(int field_idx, domain_t* domain_)
    {
        const int mask_id = MASK_LIST::Mask_FMM_Source;
        for (int level = base_level_ - 1; level >= 0; --level)
        {
        
            for (auto it = domain_->begin(level); it != domain_->end(level);
                ++it)
            {
                if (it->has_data() && it->fmm_mask(fmm_mask_idx_, mask_id))
                    lagrange_intrp_vec[field_idx]->nli_antrp_node<fmm_s_type>(field_idx,
                        it, mask_id, fmm_mask_idx_);
            }
        
            domain_->decomposition()
                .client()
                ->template communicate_updownward_add_OMP<fmm_s_type, fmm_s_type>(
                    level, true, true, fmm_mask_idx_, field_idx);
        
            for (auto it = domain_->begin(level); it != domain_->end(level);
                ++it)
            {
                if (!it->locally_owned() && it->has_data() &&
                    it->data().is_allocated())
                {
                    auto& cp2 = it->data_r(fmm_s, field_idx).linalg_data();
                    cp2 *= 0.0;
                }
            }
        }
    }

    template<class Kernel>
    void fmm_tt(int field_idx, octant_t* o_s, octant_t* o_t, Kernel* _kernel, int level_diff)
    {
        const auto t0_fft = clock_type::now();

        const auto t_base = o_t->data_r(fmm_t).real_block().base();
        const auto s_base = o_s->data_r(fmm_s).real_block().base();

        // Get extent of Source region
        const auto s_extent = o_s->data_r(fmm_s).real_block().extent();
        const auto shift = t_base - s_base;

        // Calculate the dimensions of the LGF to be allocated
        const auto base_lgf = shift - (s_extent - 1);
        const auto extent_lgf = 2 * (s_extent)-1;

        block_dsrp_t lgf_block(base_lgf, extent_lgf);

        convs_[field_idx]->apply_forward_add(
            lgf_block, _kernel, level_diff, o_s->data_r(fmm_s, field_idx));

        const auto t1_fft = clock_type::now();
        //timings_.fftw += t1_fft - t0_fft;
        //timings_.fftw_count_max += 1;
    }

    struct Timings
    {
        mDuration_type global = mDuration_type(0);
        mDuration_type anterp = mDuration_type(0);
        mDuration_type bx = mDuration_type(0);
        mDuration_type interp = mDuration_type(0);
        mDuration_type fftw = mDuration_type(0);
        std::size_t    fftw_count_max = 0;
        std::size_t    fftw_count_min = 0;

        void accumulate(boost::mpi::communicator _comm) noexcept
        {
            Timings tlocal = *this;

            decltype(tlocal.global.count()) cglobal, canterp, cbx, cinterp,
                cfftw;
            std::size_t cfftw_count_max, cfftw_count_min;

            //auto comp=[&](const auto& v0, const auto& v1){    return v0>v1? v0  :v1;};
            //auto min_comp=[&](const auto& v0, const auto& v1){return v0>v1? v1  :v0;};

            auto comp = boost::mpi::maximum<float_type>();
            auto min_comp = boost::mpi::minimum<float_type>();

            boost::mpi::all_reduce(_comm, tlocal.global.count(), cglobal, comp);
            boost::mpi::all_reduce(_comm, tlocal.anterp.count(), canterp, comp);
            boost::mpi::all_reduce(_comm, tlocal.bx.count(), cbx, comp);
            boost::mpi::all_reduce(_comm, tlocal.interp.count(), cinterp, comp);
            boost::mpi::all_reduce(_comm, tlocal.fftw.count(), cfftw, comp);

            boost::mpi::all_reduce(
                _comm, tlocal.fftw_count_max, cfftw_count_min, min_comp);
            boost::mpi::all_reduce(
                _comm, tlocal.fftw_count_max, cfftw_count_max, comp);

            this->global = mDuration_type(cglobal);
            this->anterp = mDuration_type(canterp);
            this->interp = mDuration_type(cinterp);
            this->bx = mDuration_type(cbx);
            this->fftw = mDuration_type(cfftw);
            this->fftw_count_min = cfftw_count_min;
            this->fftw_count_max = cfftw_count_max;
        }
    };

    const auto& timings() const noexcept { return timings_; }
    auto&       timings() noexcept { return timings_; }

  private:
    domain_t* domain;

  public:
    Nli lagrange_intrp;
    std::vector<std::unique_ptr<Nli>> lagrange_intrp_vec;

  private:
    int                               fmm_mask_idx_;
    int                               base_level_;
    std::vector<std::unique_ptr<convolution_t>>        convs_; ///< fft convolution
    parallel_ostream::ParallelOstream pcout =
        parallel_ostream::ParallelOstream(1);

    std::vector<std::pair<octant_t*, int>> sorted_octants_;

  private: //timings
    Timings timings_;
};

} // namespace fmm
} // namespace iblgf

#endif //IBLGF_INCLUDED_FMM_HPP
