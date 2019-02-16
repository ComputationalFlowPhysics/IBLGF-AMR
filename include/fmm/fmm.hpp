#ifndef IBLGF_INCLUDED_FMM
#define IBLGF_INCLUDED_FMM

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

// IBLGF-specific
#include <types.hpp>
#include <algorithm>
#include <list>
#include <vector>
#include <iostream>
#include <cmath>
#include <functional>
#include <cstring>
#include <fftw3.h>


//test
#include <global.hpp>
#include <simulation.hpp>
#include <linalg/linalg.hpp>
#include <fmm/fmm_nli.hpp>
#include <domain/domain.hpp>
#include <domain/octree/tree.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <domain/octree/tree.hpp>
#include <IO/parallel_ostream.hpp>
#include "../utilities/convolution.hpp"

namespace fmm
{

using namespace domain;


template<class Setup>
class Fmm
{
public: //Ctor:
    static constexpr std::size_t Dim=Setup::Dim;

    using dims_t = types::vector_type<int,Dim>;
    using datablock_t  = typename Setup::datablock_t;
    using block_dsrp_t = typename datablock_t::block_descriptor_type;
    using domain_t = typename  Setup::domain_t;

    //Fields:
    using fmm_s = typename Setup::fmm_s;
    using fmm_t = typename Setup::fmm_t;

public:
    Fmm(int Nb)
    :lagrange_intrp(Nb),
    conv_( dims_t{{Nb,Nb,Nb}}, dims_t{{Nb,Nb, Nb}} )
    {
    }

    template<
        class Source,
        class Target
        >
    void fmm_for_level(domain_t* domain_, 
                       int level, 
                       bool for_non_leaf=false)
    {

        clock_t fmm_start = clock();
        std::cout << "------------------------------------"  << std::endl;
        std::cout << "Fmm - Level - " << level << std::endl;

        const float_type dx_base=domain_->dx_base();
        auto refinement_level = domain_-> begin(level)->refinement_level();
        auto dx_level =  dx_base/std::pow(2, refinement_level);

        // Find the subtree
        auto o_start = domain_-> begin(level);
        auto o_end   = domain_-> end(level);
        o_end--;

        if (for_non_leaf)
        {
            while ((o_start != domain_->end(level)) && (o_start->is_leaf()==true) ) o_start++;
            if (o_start == domain_->end(level))
            {
                std::cout<< "All leaves" << std::endl;
                return;
            }
            while (o_end->is_leaf()==true) o_end--;
        }

        // Initialize for each fmm // zero ing all tree
        std::cout << "Fmm - initialize " << std::endl;
        fmm_init_zero<fmm_s>(domain_, level, o_start, o_end);

        // Copy to temporary variables // only the base level
        std::cout << "Fmm - copy source " << std::endl;
        fmm_init_copy<Source, fmm_s>(domain_, level, o_start, o_end, for_non_leaf);
        //fmm_init_copy<Source, fmm_tmp>(domain_, level, o_start, o_end, for_non_leaf);

        // Antrp for all // from base level up
        clock_t fmm_antrp_start = clock();
        std::cout << "Fmm - antrp " << std::endl;
        fmm_antrp<fmm_s>(domain_, level, o_start, o_end);
        clock_t fmm_antrp_end = clock();
        double  fmm_antrp_time = (double) (fmm_antrp_end-fmm_antrp_start) / CLOCKS_PER_SEC;
        std::cout << "Fmm - antrp - done / time = "<< fmm_antrp_time << " (s * threads)" << std::endl;
        //fmm_antrp<fmm_tmp>(domain_, level, o_start, o_end);

        // Nearest neighbors and self
        clock_t fmm_B0_start = clock();
        std::cout << "Fmm - B0 " << std::endl;
        fmm_B0<fmm_s, fmm_t>(domain_, level, o_start, o_end, dx_level);
        clock_t fmm_B0_end = clock();
        double  fmm_B0_time = (double) (fmm_B0_end-fmm_B0_start) / CLOCKS_PER_SEC;
        std::cout << "Fmm - B0 -    done / time = "<< fmm_B0_time << " (s * threads)" << std::endl;

        // FMM 189
        std::cout << "Fmm - B1 and up" << std::endl;
        clock_t fmm_Bx_start = clock();
        fmm_Bx<fmm_s, fmm_t>(domain_, level, o_start, o_end, dx_level);
        clock_t fmm_Bx_end = clock();
        double  fmm_Bx_time = (double) (fmm_Bx_end-fmm_Bx_start) / CLOCKS_PER_SEC;
        std::cout << "Fmm - Bx    - done / time = "<< fmm_Bx_time << " (s * threads)" << std::endl;

        // Intrp
        std::cout << "Fmm - intrp " << std::endl;
        clock_t fmm_intrp_start = clock();
        fmm_intrp<fmm_t>(domain_, level, o_start, o_end);
        clock_t fmm_intrp_end = clock();
        double  fmm_intrp_time = (double) (fmm_intrp_end-fmm_intrp_start) / CLOCKS_PER_SEC;
        std::cout << "Fmm - intrp - done / time = "<< fmm_intrp_time << " (s * threads)" << std::endl;

        // Copy back
        std::cout << "Fmm - output " << std::endl;
        if (!for_non_leaf)
            fmm_add_equal<Target, fmm_t>(domain_, level, o_start, o_end, for_non_leaf);
        else
            fmm_minus_equal<Target, fmm_t>(domain_, level, o_start, o_end, for_non_leaf);

        clock_t fmm_end = clock();
        double fmm_time = (double) (fmm_end-fmm_start) / CLOCKS_PER_SEC;
        std::cout << "Fmm - level - done / time = "<< fmm_time << " (s * threads)" << std::endl;

    }
  template<
        class Source,
        class Target
        >
    void fmm_for_level_test(domain_t* domain_, 
                            int level, 
                            bool for_non_leaf=false)
    {
        std::cout << "------------------------------------"  << std::endl;
        std::cout << "Fmm - Level - " << level << std::endl;

        const float_type dx_base=domain_->dx_base();
        auto refinement_level = domain_-> begin(level)->refinement_level();
        auto dx_level =  dx_base/std::pow(2, refinement_level);

        // Find the subtree
        auto o_start = domain_-> begin(level);
        auto o_end   = domain_-> end(level);
        o_end--;

        if (for_non_leaf)
        {
            while ((o_start != domain_->end(level)) && 
                   (o_start->is_leaf()==true) ) o_start++;

            if (o_start == domain_->end(level)) { return; }
            while (o_end->is_leaf()==true) o_end--;
        }

        // Initialize for each fmm // zero ing all tree
        fmm_init_zero<fmm_s>(domain_, level, o_start, o_end);
        fmm_init_zero<fmm_t>(domain_, level, o_start, o_end);


        // Copy to temporary variables // only the base level
        fmm_init_copy<Source, fmm_s>(domain_, level, o_start, o_end, for_non_leaf);

        //anterpolation communication
        //domain_->decomposition().
        //    template communicate_updownward_pass<fmm_s, fmm_t>(true);

        // Nearest neighbors and self
        fmm_B0<fmm_s, fmm_t>(domain_, level, o_start, o_end, dx_level);

        // FMM 189
        //fmm_Bx<fmm_s, fmm_t>(domain_, level, o_start, o_end, dx_level);
        domain_->decomposition().template communicate_influence<fmm_s, fmm_t>(level);


        //interpolation communication
        domain_->decomposition().
            template communicate_updownward_pass<fmm_s, fmm_t>(false);

        // Copy back
        if (!for_non_leaf)
            fmm_add_equal<Target, fmm_t>(domain_, level, o_start, o_end, for_non_leaf);
        else
            fmm_minus_equal<Target, fmm_t>(domain_, level, o_start, o_end, for_non_leaf);

    }



    template<
        class s,
        class t,
        class octant_itr_t
    >
    void fmm_Bx(domain_t* domain_, 
                int level, 
                octant_itr_t o_start, 
                octant_itr_t o_end, 
                float_type dx_level)
    {

        auto o_1 = (*o_start);
        auto o_2 = (*o_end);
        auto o_1_old = o_1;
        auto o_2_old = o_2;
        int l = level;

        if (o_start->level() != o_end->level())
            throw std::runtime_error("Level has to be the same");

        while (o_1_old->key() != o_2_old->key() )
        {
            std::cout << "Bx  - level = " << l << std::endl;

            auto level_o_1 = domain_->tree()->find(l, o_1->key());
            auto level_o_2 = domain_->tree()->find(l, o_2->key());
            auto level_o_2_dup = level_o_2;
            level_o_2++;

            for (auto it_t = level_o_1;
                    it_t!=(level_o_2); ++it_t)
            {

                if(!it_t) continue;
                for (std::size_t i=0; i< it_t->influence_number(); ++i)
                {
                    auto n_s = it_t->influence(i);
                    if (n_s)
                    {
                        if (n_s->inside(level_o_1,  level_o_2_dup))
                        {
                            fmm_fft<s,t>(n_s, it_t, level-l, dx_level);
                        }
                    }
                }
            }

            o_1_old = o_1;
            o_2_old = o_2;

            o_1 = o_1->parent();
            o_2 = o_2->parent();
            l--;
        }
    }

    template<
        class s,
        class t,
        class octant_itr_t
    >
    void fmm_B0(domain_t* domain_, 
                int level, 
                octant_itr_t o_start, 
                octant_itr_t o_end, 
                float_type dx_level)
    {
        int level_diff = 0;
        if (o_start->level() != o_end->level())
            throw std::runtime_error("Level has to be the same");

        auto o_end_2 = o_end;
        o_end++;

        for (auto it_t = o_start; it_t!=(o_end); ++it_t)
        {

            if(!it_t->data()) continue;
            for (int i=0; i<27; ++i)
            {
                auto n_s = it_t->neighbor(i);
                if (n_s)
                    if (n_s->inside(o_start, o_end_2))
                    {
                        fmm_fft<s,t>(n_s, it_t, level_diff, dx_level);
                    }
            }
        }
    }

    template<
        class f1,
        class f2,
        class octant_itr_t
    >
    void fmm_add_equal(domain_t* domain_, 
                       int level, 
                       octant_itr_t o_start, 
                       octant_itr_t o_end, 
                       bool for_non_leaf)
    {

        if (o_start->level() != o_end->level())
            throw std::runtime_error("Level has to be the same");

        o_end++;

        for (auto it = o_start; it!=(o_end); ++it)
        {
            if(it->data())
            {
                if ( !( (for_non_leaf) && (it->is_leaf()) ))
                it->data()->template get_linalg<f1>().get()->
                    cube_noalias_view() +=
                                it->data()->template get_linalg_data<f2>();
            }
        }

    }

    template<
        class f1,
        class f2,
        class octant_itr_t
    >
    void fmm_minus_equal(domain_t* domain_, 
                         int level, 
                         octant_itr_t o_start, 
                         octant_itr_t o_end, 
                         bool for_non_leaf)
    {

        if (o_start->level() != o_end->level())
            throw std::runtime_error("Level has to be the same");

        o_end++;

        for (auto it = o_start; it!=(o_end); ++it)
        {
            if(it->data())
            {
                if ( !( (for_non_leaf) && (it->is_leaf()) ))
                it->data()->template get_linalg<f1>().get()->
                    cube_noalias_view() -=
                            it->data()->template get_linalg_data<f2>();
            }
        }

    }

    template<class f,
        class octant_itr_t
    >
    void fmm_init_zero(domain_t* domain_, 
                       int level,
                       octant_itr_t o_start, 
                       octant_itr_t o_end)
    {
        auto o_1 = (*o_start);
        auto o_2 = (*o_end);

        auto o_1_old = o_1;
        auto o_2_old = o_2;

        if (o_1->level() != o_2->level())
            throw std::runtime_error("Level has to be the same");

        while (o_1_old->key() != o_2_old->key() )
        {
            auto level_o_1 = domain_->tree()->find(level, o_1->key());
            auto level_o_2 = domain_->tree()->find(level, o_2->key());
            level_o_2++;

            for (auto it = level_o_1;
                    it!=(level_o_2); ++it)
            {
                {
                    if(!it->data())continue;
                    for(auto& e: it->data()->template get_data<f>())
                        e=0.0;
                }
            }

            o_1_old = o_1;
            o_2_old = o_2;

            o_1 = o_1->parent();
            o_2 = o_2->parent();
            level--;
        }

    }

    template<
        class from,
        class to,
        class octant_itr_t
    >
    void fmm_init_copy(domain_t* domain_, 
                       int level,
                       octant_itr_t o_start, 
                       octant_itr_t o_end, 
                       bool for_non_leaf)
    {
        if (o_start->level() != o_end->level())
            throw std::runtime_error("Level has to be the same");

        o_end++;
        for (auto it = o_start; it!=(o_end); ++it)
        {
            if(it->data())
            {
                if ( !( (for_non_leaf) && (it->is_leaf()) ))

                it->data()->template get_linalg<to>().get()->
                    cube_noalias_view() =
                     it->data()->template get_linalg_data<from>() * 1.0;
            }
        }

    }

    template< class fmm_t,
        class octant_itr_t
        >
    void fmm_intrp(domain_t* domain_, 
                   int level,
                   octant_itr_t o_start, 
                   octant_itr_t o_end)
    {
        level--;
        auto o_1 = o_start->parent();
        auto o_2 = o_end->parent();

        if (o_1->key() != o_2->key() )
            fmm_intrp<fmm_t>(domain_, level, o_1, o_2);

        auto level_o_1 = domain_->tree()->find(level, o_1->key());
        auto level_o_2 = domain_->tree()->find(level, o_2->key());
        level_o_2++;

        std::cout<< "Fmm - intrp - level: " << level << std::endl;

        for (auto it = level_o_1;
                it!=(level_o_2); ++it)
        {
            if(it->data())
            {
                lagrange_intrp.nli_intrp_node<fmm_t>(it);
            }
        }
    }

    template< class fmm_s,
        class octant_itr_t
    >
    void fmm_antrp(domain_t* domain_, 
                   int level,
                   octant_itr_t o_start, 
                   octant_itr_t o_end)
    {

        // start with one level up and call the parents of each
        level--;
        auto o_1 = o_start->parent();
        auto o_2 = o_end->parent();
        auto o_1_old = o_1;
        auto o_2_old = o_2;

        if (o_1->level() != o_2->level())
            throw std::runtime_error("Level has to be the same");

        while (o_1_old->key() != o_2_old->key() )
        {
            std::cout<< "Fmm - antrp - level: " << level << std::endl;
            auto level_o_1 = domain_->tree()->find(level, o_1->key());
            auto level_o_2 = domain_->tree()->find(level, o_2->key());
            level_o_2++;

            for (auto it = level_o_1;
                    it!=(level_o_2); ++it)
            {
                if(it->data())
                    lagrange_intrp.nli_antrp_node<fmm_s>(it);
            }

            // go 1 level up
            level--;
            o_1_old = o_1;
            o_2_old = o_2;
            o_1 = o_1->parent();
            o_2 = o_2->parent();
        }
    }

    template<
        class S,
        class T,
        class octant_t,
        class octant_itr_t
    >
    void fmm_fft(octant_t o_s, 
                 octant_itr_t o_t, 
                 int level_diff, 
                 float_type dx_level)
    {

        const auto t_base = o_t->data()->template get<T>().
                                        real_block().base();
        const auto s_base = o_s->data()->template get<S>().
                                real_block().base();

        if(!o_s->locally_owned())return;

        // Get extent of Source region
        const auto s_extent = o_s->data()->template get<S>().
                                real_block().extent();
        const auto shift    = t_base - s_base;

        // Calculate the dimensions of the LGF to be allocated
        const auto base_lgf   = shift - (s_extent - 1);
        const auto extent_lgf = 2 * (s_extent) - 1;

        block_dsrp_t lgf_block(base_lgf, extent_lgf);
        block_dsrp_t extractor(s_base, s_extent);

        conv_.apply_lgf(lgf_block, level_diff,
                o_s->data()->template get<S>(),
                extractor,
                o_t->data()->template get<T>(),
                dx_level*dx_level);

    }


    public:
        Nli lagrange_intrp;
    private:
        std::vector<float_type>     lgf;
        fft::Convolution            conv_;      ///< fft convolution
    };

}

#endif //IBLGF_INCLUDED_FMM_HPP
