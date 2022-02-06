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

#ifndef IBLGF_INCLUDED_FMM
#define IBLGF_INCLUDED_FMM

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

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
template<class Domain>
struct FmmMaskBuilder
{
    using octant_t = typename Domain::octant_t;
    using MASK_TYPE = typename octant_t::MASK_TYPE;
    using MASK_LIST = typename octant_t::MASK_LIST;

public:
    //static void fmm_mask_build(Domain* domain_, bool subtract_non_leaf_)
    //{
    //    fmm_IB2IB_mask(domain_);
    //    fmm_IB2AMR_mask(domain_);
    //    fmm_vortex_streamfun_mask(domain_);
    //    fmm_lgf_mask_build(domain_, subtract_non_leaf_);
    //}

    static void fmm_IB2xIB_mask(Domain* domain_)
    {

        int fmm_mask_idx = octant_t::fmm_mask_idx_gen(MASK_TYPE::IB2xIB);

        // clean
        for (auto it = domain_->begin();
             it != domain_->end(); ++it)
        {
            it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);
            it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, false);
        }

        int l = domain_->tree()->depth()-1;

        for (auto it = domain_->begin(l);
             it != domain_->end(l); ++it)
        {
            if (it->is_ib())
            {
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, true);
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, true);
                it->add_load(it->neighbor_number() );
            }
            if (it->is_extended_ib())
            {
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, true);
                it->add_load(it->neighbor_number() );
            }
        }

        fmm_upward_pass_masks(domain_, l, MASK_LIST::Mask_FMM_Source,
                MASK_LIST::Mask_FMM_Target, fmm_mask_idx);
        fmm_clean_no_inf_masks(domain_, l, MASK_LIST::Mask_FMM_Source,
            MASK_LIST::Mask_FMM_Target, fmm_mask_idx);

    }

    static void fmm_xIB2IB_mask(Domain* domain_)
    {

        int fmm_mask_idx = octant_t::fmm_mask_idx_gen(MASK_TYPE::xIB2IB);

        // clean
        for (auto it = domain_->begin();
             it != domain_->end(); ++it)
        {
            it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);
            it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, false);
        }

        int l = domain_->tree()->depth()-1;

        for (auto it = domain_->begin(l);
             it != domain_->end(l); ++it)
        {
            if (it->is_ib())
            {
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, true);
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, true);
                it->add_load(it->neighbor_number() );
            }
            if (it->is_extended_ib())
            {
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, true);
                it->add_load(it->neighbor_number() );
            }
        }

        fmm_upward_pass_masks(domain_, l, MASK_LIST::Mask_FMM_Source,
                MASK_LIST::Mask_FMM_Target, fmm_mask_idx);
        fmm_clean_no_inf_masks(domain_, l, MASK_LIST::Mask_FMM_Source,
            MASK_LIST::Mask_FMM_Target, fmm_mask_idx);

    }

    static void fmm_IB2AMR_mask(Domain* domain_)
    {

        // find all parents that are
        for (int l = domain_->tree()->depth()-2;
             l >= domain_->tree()->base_level() ; --l)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l);
                 ++it)
            {
                for (int c = 0; c < it->num_children(); ++c)
                    if (it->child(c) && it->child(c)->has_data() && it->child(c)->is_ib())
                        it->is_ib()=true;
            }

        }

        for (int l = domain_->tree()->depth()-1;
             l >= domain_->tree()->base_level() ; --l)
        {

            int fmm_mask_idx = octant_t::fmm_mask_idx_gen(MASK_TYPE::IB2AMR,
                    l-domain_->tree()->base_level());

            // clean
            for (auto it = domain_->begin();
                    it != domain_->end(); ++it)
            {
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, false);
            }

            for (auto it = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, true);

                if (it->is_ib())
                {
                    it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, true);
                    it->add_load(it->neighbor_number() * 2);
                }

            }

            fmm_upward_pass_masks(domain_, l, MASK_LIST::Mask_FMM_Source,
                    MASK_LIST::Mask_FMM_Target, fmm_mask_idx);

        }

    }


    static void fmm_vortex_streamfun_mask(Domain* domain_)
    {
        int base_level = domain_->tree()->base_level();
        int fmm_mask_idx = octant_t::fmm_mask_idx_gen(MASK_TYPE::STREAM);
        //int fmm_mask_idx = 0;

        for (auto it = domain_->begin();
             it != domain_->end(); ++it)
        {
            it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);
            it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, false);
            it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);
            it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, false);
        }

        for (auto it = domain_->begin(base_level);
             it != domain_->end(base_level); ++it)
        {
            if (!it->has_data()) continue;

            if (!it->is_correction())
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, true);
            else
                //it->fmm_mask(fmm_mask_idx,MASK_LIST::Mask_FMM_Source, true);
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);

            if (it->is_correction())
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, true);
            else
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, true);
        }

        fmm_upward_pass_masks(domain_, base_level, MASK_LIST::Mask_FMM_Source,
            MASK_LIST::Mask_FMM_Target, fmm_mask_idx);
    }


    static void fmm_laplacian_BC_mask(Domain* domain_)
    {
        int base_level = domain_->tree()->base_level();
        int fmm_mask_idx = octant_t::fmm_mask_idx_gen(MASK_TYPE::Laplacian_BC);
        //int fmm_mask_idx = 0;

        for (auto it = domain_->begin();
             it != domain_->end(); ++it)
        {
            it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);
            it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, false);
            it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);
            it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, false);
        }

        for (auto it = domain_->begin(base_level);
             it != domain_->end(base_level); ++it)
        {
            if (!it->has_data()) continue;

            if (!it->is_correction())
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, true);
            else
                //it->fmm_mask(fmm_mask_idx,MASK_LIST::Mask_FMM_Source, true);
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);

            if (it->is_correction())
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, true);
            else
                it->fmm_mask(fmm_mask_idx, MASK_LIST::Mask_FMM_Target, false);
        }

        fmm_upward_pass_masks(domain_, base_level, MASK_LIST::Mask_FMM_Source,
            MASK_LIST::Mask_FMM_Target, fmm_mask_idx);
    }

    static void fmm_clean_load(Domain* domain_)
    {
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        { it->load() = 0; }
    }

    static void fmm_lgf_mask_build(Domain* domain_, bool subtract_non_leaf)
    {
        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            fmm_dry(domain_, l, false, subtract_non_leaf);
            fmm_dry(domain_, l, true, subtract_non_leaf);
        }
    }

    static void fmm_dry(Domain* domain_, int base_level,
        bool non_leaf_as_source, bool subtract_non_leaf)
    {
        int refinement_level = base_level - domain_->tree()->base_level();
        //int fmm_mask_idx = refinement_level * 2 + non_leaf_as_source + 1;
        int fmm_mask_idx = octant_t::fmm_mask_idx_gen(MASK_TYPE::AMR2AMR, refinement_level, non_leaf_as_source);

        fmm_dry_init_base_level_masks(domain_, base_level, non_leaf_as_source,
            fmm_mask_idx, subtract_non_leaf);
        fmm_upward_pass_masks (domain_, base_level, MASK_LIST::Mask_FMM_Source,
            MASK_LIST::Mask_FMM_Target, fmm_mask_idx);
        fmm_clean_no_inf_masks(domain_, base_level, MASK_LIST::Mask_FMM_Source,
            MASK_LIST::Mask_FMM_Target, fmm_mask_idx);
    }

    static void fmm_clean_no_inf_masks(Domain* domain_, int base_level,
        int mask_source_id, int mask_target_id, const int _fmm_mask_idx)
    {
        for (int level=0; level<base_level; ++level)
        {
            for (auto it = domain_->begin(level); it != domain_->end(level);
                 ++it)
            {

                //domain_->tree()->influence_list_build(it.ptr());
                //domain_->tree()->neighbor_list_build(it.ptr());
                //for source masks
                if (it->fmm_mask(_fmm_mask_idx, mask_source_id))
                {
                    if (it->parent() && it->parent()->has_data() && it->parent()->fmm_mask(_fmm_mask_idx, mask_source_id) )
                        continue;

                    bool has_target=false;
                    for (int i = 0; i < it->influence_number(); ++i)
                    {
                        auto n_s = it->influence(i);
                        if (n_s && n_s->has_data() &&
                                n_s->fmm_mask(_fmm_mask_idx, mask_target_id))
                        {
                            has_target = true;
                            break;
                        }
                    }

                    //for (int i = 0; i < it->nNeighbors(); ++i)
                    //{
                    //    auto n_s = it->neighbor(i);
                    //    if (n_s && n_s->has_data() &&
                    //            n_s->fmm_mask(_fmm_mask_idx, mask_target_id))
                    //    {
                    //        has_target = true;
                    //        break;
                    //    }
                    //}

                    if (!has_target)
                        it->fmm_mask(_fmm_mask_idx, mask_source_id, false);
                }

                // for target masks
                if (it->fmm_mask(_fmm_mask_idx, mask_target_id))
                {
                    if ( it->parent() && it->parent()->has_data() && it->parent()->fmm_mask(_fmm_mask_idx, mask_target_id) )
                        continue;

                    bool has_source=false;
                    for (int i = 0; i < it->influence_number(); ++i)
                    {
                        auto n_s = it->influence(i);
                        if (n_s && n_s->has_data() &&
                                n_s->fmm_mask(_fmm_mask_idx, mask_source_id))
                        {
                            has_source = true;
                            break;
                        }
                    }

                    //for (int i = 0; i < it->nNeighbors(); ++i)
                    //{
                    //    auto n_s = it->neighbor(i);
                    //    if (n_s && n_s->has_data() &&
                    //            n_s->fmm_mask(_fmm_mask_idx, mask_source_id))
                    //    {
                    //        has_source = true;
                    //        break;
                    //    }
                    //}

                    if (!has_source)
                        it->fmm_mask(_fmm_mask_idx, mask_target_id, false);
                }

            }
        }
    }

    static void fmm_upward_pass_masks(Domain* domain_, int base_level,
        int mask_source_id, int mask_target_id, const int _fmm_mask_idx,
        float_type _base_factor = 1.0)
    {
        // for all levels
        for (int level = base_level - 1; level >= 0; --level)
        {
            //_base_factor *=0.6;
            // parent's mask is true if any of its child's mask is true
            for (auto it = domain_->begin(level); it != domain_->end(level);
                 ++it)
            {
                it->fmm_mask(_fmm_mask_idx, mask_source_id, false);
                for (int c = 0; c < it->num_children(); ++c)
                {
                    if (it->child(c) && it->child(c)->has_data() &&
                        it->child(c)->fmm_mask(_fmm_mask_idx, mask_source_id))
                    {
                        it->fmm_mask(_fmm_mask_idx, mask_source_id, true);
                        break;
                    }
                }
            }

            for (auto it = domain_->begin(level); it != domain_->end(level);
                 ++it)
            {
                // including ghost parents
                it->fmm_mask(_fmm_mask_idx, mask_target_id, false);
                for (int c = 0; c < it->num_children(); ++c)
                {
                    if (it->child(c) && it->child(c)->has_data() &&
                        it->child(c)->fmm_mask(_fmm_mask_idx, mask_target_id))
                    {
                        it->fmm_mask(_fmm_mask_idx, mask_target_id, true);
                        break;
                    }
                }

                // calculate load
                if (it->fmm_mask(_fmm_mask_idx, mask_target_id))
                    for (int i = 0; i < it->influence_number(); ++i)
                    {
                        const auto inf = it->influence(i);
                        if (inf && inf->fmm_mask(_fmm_mask_idx, mask_source_id))
                            it->add_load(1.0 * _base_factor);
                    }
            }
        }
    }

    static void fmm_dry_init_base_level_masks(Domain* domain_, int base_level,
        bool non_leaf_as_source, int _fmm_mask_idx, bool subtract_non_leaf,
        int _load_factor = 1, bool _neighbor_only = false)
    {
        for (auto it = domain_->begin(base_level);
             it != domain_->end(base_level); ++it)
        {
            it->fmm_mask(_fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);
            it->fmm_mask(_fmm_mask_idx, MASK_LIST::Mask_FMM_Target, false);
            it->fmm_mask(_fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);
            it->fmm_mask(_fmm_mask_idx, MASK_LIST::Mask_FMM_Target, false);
        }

        if (non_leaf_as_source)
        {
            for (auto it = domain_->begin(base_level);
                 it != domain_->end(base_level); ++it)
            {
                if (!it->has_data()) continue;
                bool correction_parent = false;
                for (std::size_t i = 0; i < it->num_children(); ++i)
                    if (it->child(i) && it->child(i)->has_data() &&
                        it->child(i)->is_correction())
                        correction_parent = true;

                if (subtract_non_leaf)
                {
                    if (it->is_leaf() || it->is_correction())
                    {
                        it->fmm_mask(
                            _fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);

                        if (!correction_parent)
                            it->fmm_mask(_fmm_mask_idx,
                                MASK_LIST::Mask_FMM_Target, false);
                        else
                            it->fmm_mask(_fmm_mask_idx,
                                MASK_LIST::Mask_FMM_Target, true);
                    }
                    else
                    {
                        it->fmm_mask(
                            _fmm_mask_idx, MASK_LIST::Mask_FMM_Source, true);
                        //if (!it->is_correction())
                        it->fmm_mask(
                            _fmm_mask_idx, MASK_LIST::Mask_FMM_Target, true);

                        if (!_neighbor_only)
                            it->add_load(it->influence_number() * _load_factor);

                        it->add_load(it->neighbor_number() * _load_factor);
                    }
                }
                else
                {
                    if (it->is_leaf() || it->is_correction())
                    {
                        it->fmm_mask(
                            _fmm_mask_idx, MASK_LIST::Mask_FMM_Source, true);

                        if (it->is_leaf())
                            it->fmm_mask(_fmm_mask_idx,
                                MASK_LIST::Mask_FMM_Target, true);
                        else
                            it->fmm_mask(_fmm_mask_idx,
                                MASK_LIST::Mask_FMM_Target, false);
                    }
                    else
                    {
                        it->fmm_mask(
                            _fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);
                        it->fmm_mask(
                            _fmm_mask_idx, MASK_LIST::Mask_FMM_Target, true);

                        if (!_neighbor_only)
                            it->add_load(it->influence_number() * _load_factor);

                        it->add_load(it->neighbor_number() * _load_factor);
                    }
                }
            }
        }
        else
        {
            for (auto it = domain_->begin(base_level);
                 it != domain_->end(base_level); ++it)
            {
                if (!it->has_data()) continue;

                //bool correction_parent=false;
                //for (std::size_t i=0; i<it->num_children(); ++i)
                //    if (it->child(i) && it->child(i)->has_data() && it->child(i)->is_correction())
                //        correction_parent = true;

                if (subtract_non_leaf)
                {
                    it->fmm_mask(
                        _fmm_mask_idx, MASK_LIST::Mask_FMM_Source, true);

                    it->fmm_mask(
                        _fmm_mask_idx, MASK_LIST::Mask_FMM_Target, true);
                }
                else
                {
                    if (it->is_leaf() || it->is_correction())
                    {
                        it->fmm_mask(
                            _fmm_mask_idx, MASK_LIST::Mask_FMM_Source, false);

                        if (it->is_leaf())
                            it->fmm_mask(_fmm_mask_idx,
                                MASK_LIST::Mask_FMM_Target, true);
                        else
                            it->fmm_mask(_fmm_mask_idx,
                                MASK_LIST::Mask_FMM_Target, false);
                    }
                    else
                    {
                        it->fmm_mask(
                            _fmm_mask_idx, MASK_LIST::Mask_FMM_Source, true);
                        it->fmm_mask(
                            _fmm_mask_idx, MASK_LIST::Mask_FMM_Target, false);
                    }
                }

                if (!_neighbor_only)
                    it->add_load(it->influence_number() * _load_factor);

                it->add_load(it->neighbor_number() * _load_factor);
            }
        }
    }
};

using namespace domain;

template<class Setup>
class Fmm
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
    Fmm(domain_t* _domain, int Nb)
    : domain(_domain)
    , lagrange_intrp(Nb)
    , conv_(dims_t(Nb), dims_t(Nb))
    {
	/*dims_t tmp1, tmp2;
	for(int i = 0; i < Dim; i++) {
	    tmp1[i] = Nb;
	    tmp2[i] = Nb;
	}
	conv_ = convolution_t(tmp1, tmp2);*/
    }

    template<class Source, class Target, class Kernel>
    void apply(domain_t* domain_, Kernel* _kernel, int level,
        bool non_leaf_as_source, float_type add_with_scale = 1.0,
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

        if (_kernel->neighbor_only())
        {
            //pcout<<"Integrating factor for level: "<< level << std::endl;
            fmm_init_zero<fmm_s_type>(domain_, MASK_LIST::Mask_FMM_Source);
            fmm_init_zero<fmm_t_type>(domain_, MASK_LIST::Mask_FMM_Target);
            fmm_init_copy<Source, fmm_s_type>(domain_);

            // Sort_BX
            // For IF one one doesn't have to scale it with dx, so the scale is
            // set to be 1.0 here
            sort_bx_octants(domain, _kernel);
            fmm_Bx(domain_, _kernel, 1.0);

            fmm_add_equal<Target, fmm_t_type>(domain_, add_with_scale);

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

        ////Initialize for each fmm//zero ing all tree
        fmm_init_zero<fmm_s_type>(domain_, MASK_LIST::Mask_FMM_Source);
        fmm_init_zero<fmm_t_type>(domain_, MASK_LIST::Mask_FMM_Target);

        //// Copy to temporary variables // only the base level
        fmm_init_copy<Source, fmm_s_type>(domain_);
        sort_bx_octants(domain, _kernel);

#ifdef POISSON_TIMINGS
        timings_ = Timings();
        //domain_->client_communicator().barrier();
#endif

        //// Anterpolation
        //pcout<<"FMM Antrp start" << std::endl;
#ifdef POISSON_TIMINGS
        auto t0_anterp = clock_type::now();
#endif
        fmm_antrp(domain_);
#ifdef POISSON_TIMINGS
        //domain_->client_communicator().barrier();
        auto t1_anterp = clock_type::now();
        timings_.anterp = t1_anterp - t0_anterp;
#endif

        //domain_->client_communicator().barrier();

        //// FMM influence list
        //pcout<<"FMM Bx start" << std::endl;
        //fmm_Bx_itr_build(domain_, level);
#ifdef POISSON_TIMINGS
        auto t0_bx = clock_type::now();
#endif
        float_type scale = (_kernel->neighbor_only()) ? 1.0 : dx_level;
        fmm_Bx(domain_, _kernel, scale);
#ifdef POISSON_TIMINGS
        //domain_->client_communicator().barrier();
        auto t1_bx = clock_type::now();
        timings_.bx = t1_bx - t0_bx;
#endif

#ifdef POISSON_TIMINGS
        //domain_->client_communicator().barrier();
        auto t0_interp = clock_type::now();
#endif
        //// Interpolation
        //pcout<<"FMM INTRP start" << std::endl;
        fmm_intrp(domain_);

#ifdef POISSON_TIMINGS
        //domain_->client_communicator().barrier();
        auto t1_interp = clock_type::now();
        timings_.interp = t1_interp - t0_interp;
        timings_.global = t1_interp - t0_anterp;
#endif

        //std::cout<<"FMM INTRP done" << std::endl;

        //// Copy back
        //if (!non_leaf_as_source)
        fmm_add_equal<Target, fmm_t_type>(domain_, add_with_scale);
        //else
        //fmm_minus_equal<Target, fmm_t_type>(domain_);

        //std::cout<<"Rank "<<world.rank() << " FFTW_count = ";
        //std::cout<<conv_.fft_count << std::endl;
        //pcout<<"FMM For Level "<< level << " End -------------------------"<<std::endl;
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
    void fmm_Bx(domain_t* domain_, Kernel* _kernel, float_type scale)
    {
#define packMessages

#ifdef packMessages
        const bool start_communication = false;
        bool       combined_messages = false;
        int        c = 0;
#else
        const bool start_communication = true;
#endif

        for (auto B_it = sorted_octants_.begin(); B_it != sorted_octants_.end();
                ++B_it)
        {
            auto       it = B_it->first;
            const int  level = it->level();
            const bool _neighbor = (level == base_level_) ? true : false;

            if (it->locally_owned())
                compute_influence_field(
                        &(*it), _kernel, base_level_ - level, scale, _neighbor);

            if (B_it->second != 0)
            {
                domain_->decomposition()
                .client()
                ->template communicate_induced_fields<fmm_t_type,
                fmm_t_type>(&(*it), this, _kernel, base_level_ - level,
                        scale, _neighbor, start_communication, fmm_mask_idx_);
            }
#ifdef packMessages
            else if (!combined_messages)
            {
                domain_->decomposition()
                .client()
                ->template combine_induced_field_messages<fmm_t_type,
                fmm_t_type>();
                combined_messages = true;
            }
            if (c % 5 == 0 && combined_messages)
            {
                domain_->decomposition()
                .client()
                ->template check_combined_induced_field_communication<
                fmm_t_type, fmm_t_type>(false);
            }
            ++c;
#endif
        }

#ifdef packMessages
        if (!combined_messages)
        {
            domain_->decomposition()
            .client()
            ->template combine_induced_field_messages<fmm_t_type,
            fmm_t_type>();
            combined_messages = true;
        }
        if (combined_messages)
        {
            domain_->decomposition()
            .client()
            ->template check_combined_induced_field_communication<
            fmm_t_type, fmm_t_type>(true);
        }
#else
        domain_->decomposition().client()->finish_induced_field_communication();
#endif
    }


    template<class Kernel>
    void compute_influence_field(octant_t* it, Kernel* _kernel, int level_diff,
        float_type dx_level, bool neighbor) noexcept
    {
        if (!(it->has_data()) ||
            !it->fmm_mask(fmm_mask_idx_, MASK_LIST::Mask_FMM_Target))
            return;

        conv_.fft_backward_field_clean();

        if (neighbor)
        {
            for (int i = 0; i < it->nNeighbors(); ++i)
            {
                auto n_s = it->neighbor(i);
                if (n_s && n_s->locally_owned() &&
                    n_s->fmm_mask(fmm_mask_idx_, MASK_LIST::Mask_FMM_Source))
                { fmm_tt(n_s, it, _kernel, 0); }
            }
        }

        if (!_kernel->neighbor_only())
        {
            for (int i = 0; i < it->influence_number(); ++i)
            {
                auto n_s = it->influence(i);
                if (n_s && n_s->locally_owned() &&
                    n_s->fmm_mask(fmm_mask_idx_, MASK_LIST::Mask_FMM_Source))
                { fmm_tt(n_s, it, _kernel, level_diff); }
            }
        }

        const auto   t_extent = it->data_r(fmm_t).real_block().extent();
        block_dsrp_t extractor(dims_t(0), t_extent);

        float_type _scale =
            (_kernel->neighbor_only()) ? 1.0 : dx_level * dx_level;
        conv_.apply_backward(extractor, it->data_r(fmm_t), _scale);
    }

    template<class f1, class f2>
    void fmm_add_equal(domain_t* domain_, float_type scale)
    {
        for (auto it = domain_->begin(base_level_);
             it != domain_->end(base_level_); ++it)
        {
            if (it->has_data() && it->locally_owned() &&
                it->fmm_mask(fmm_mask_idx_, MASK_LIST::Mask_FMM_Target))
            {
                it->data_r(f1::tag()).linalg().get()->cube_noalias_view() +=
                    it->data_r(f2::tag()).linalg_data() * scale;
            }
        }
    }

    template<class f1, class f2>
    void fmm_minus_equal(domain_t* domain_)
    {
        for (auto it = domain_->begin(base_level_);
             it != domain_->end(base_level_); ++it)
        {
            if (it->has_data() && it->locally_owned() &&
                it->fmm_mask(fmm_mask_idx_, MASK_LIST::Mask_FMM_Target))
            {
                it->data_r(f1::tag()).linalg().get()->cube_noalias_view() -=
                    it->data_r(f2::tag()).linalg_data();
            }
        }
    }

    template<class field>
    void fmm_init_zero(domain_t* domain_, int mask_id)
    {
        for (int level = base_level_; level >= 0; --level)
        {
            for (auto it = domain_->begin(level); it != domain_->end(level);
                 ++it)
            {
                if (it->has_data() && it->fmm_mask(fmm_mask_idx_, mask_id))
                {
                    for (auto& e : it->data_r(field::tag())) e = 0.0;
                }
            }
        }
    }

    template<class from, class to>
    void fmm_init_copy(domain_t* domain_)
    {
        // Neglecting the data in the buffer

        for (auto it = domain_->begin(base_level_);
             it != domain_->end(base_level_); ++it)
        {
            if (it->has_data() && it->locally_owned() &&
                it->fmm_mask(fmm_mask_idx_, MASK_LIST::Mask_FMM_Source))
            {
                auto lin_data_1 = it->data_r(from::tag()).linalg_data();
                auto lin_data_2 = it->data_r(to::tag()).linalg_data();

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

    void fmm_intrp(domain_t* domain_)
    {
        const int mask_id = MASK_LIST::Mask_FMM_Target;
        for (int level = 1; level < base_level_; ++level)
        {
            domain_->decomposition()
                .client()
                ->template communicate_updownward_assign<fmm_t_type,
                    fmm_t_type>(level, false, true, fmm_mask_idx_);

            for (auto it = domain_->begin(level); it != domain_->end(level);
                 ++it)
            {
                if (it->has_data() && it->fmm_mask(fmm_mask_idx_, mask_id))
                    lagrange_intrp.nli_intrp_node<fmm_t_type>(
                        it, mask_id, fmm_mask_idx_);
            }
        }
    }

    void fmm_antrp(domain_t* domain_)
    {
        const int mask_id = MASK_LIST::Mask_FMM_Source;
        for (int level = base_level_ - 1; level >= 0; --level)
        {
            for (auto it = domain_->begin(level); it != domain_->end(level);
                 ++it)
            {
                if (it->has_data() && it->fmm_mask(fmm_mask_idx_, mask_id))
                    lagrange_intrp.nli_antrp_node<fmm_s_type>(
                        it, mask_id, fmm_mask_idx_);
            }

            domain_->decomposition()
                .client()
                ->template communicate_updownward_add<fmm_s_type, fmm_s_type>(
                    level, true, true, fmm_mask_idx_);

            for (auto it = domain_->begin(level); it != domain_->end(level);
                 ++it)
            {
                if (!it->locally_owned() && it->has_data() &&
                    it->data().is_allocated())
                {
                    auto& cp2 = it->data_r(fmm_s).linalg_data();
                    cp2 *= 0.0;
                }
            }
        }
    }

    template<class Kernel>
    void fmm_tt(octant_t* o_s, octant_t* o_t, Kernel* _kernel, int level_diff)
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

        conv_.apply_forward_add(
            lgf_block, _kernel, level_diff, o_s->data_r(fmm_s));

        const auto t1_fft = clock_type::now();
        timings_.fftw += t1_fft - t0_fft;
        timings_.fftw_count_max += 1;
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

  private:
    int                               fmm_mask_idx_;
    int                               base_level_;
    convolution_t                     conv_; ///< fft convolution
    parallel_ostream::ParallelOstream pcout =
        parallel_ostream::ParallelOstream(1);

    std::vector<std::pair<octant_t*, int>> sorted_octants_;

  private: //timings
    Timings timings_;
};

} // namespace fmm
} // namespace iblgf

#endif //IBLGF_INCLUDED_FMM_HPP
