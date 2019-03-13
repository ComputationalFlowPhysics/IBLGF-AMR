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
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

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
    using domain_t = typename Setup::domain_t;
    using octant_t = typename domain_t::octant_t;
    using MASK_LIST = typename octant_t::MASK_LIST;

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
                       bool non_leaf_as_source=false)
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

        if (non_leaf_as_source)
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
        fmm_init_copy<Source, fmm_s>(domain_, level, o_start, o_end, non_leaf_as_source);
        //fmm_init_copy<Source, fmm_tmp>(domain_, level, o_start, o_end, non_leaf_as_source);

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
        if (!non_leaf_as_source)
            fmm_add_equal<Target, fmm_t>(domain_, level, o_start, o_end, non_leaf_as_source);
        else
            fmm_minus_equal<Target, fmm_t>(domain_, level, o_start, o_end, non_leaf_as_source);

        clock_t fmm_end = clock();
        double fmm_time = (double) (fmm_end-fmm_start) / CLOCKS_PER_SEC;
        std::cout << "Fmm - level - done / time = "<< fmm_time << " (s * threads)" << std::endl;

    }

    template< class Source, class Target >
    void fmm_for_level_test(domain_t* domain_,
                            int level,
                            bool non_leaf_as_source=false)
    {

        pcout<<"FMM For Level "<< level << " Start ---------------------------"<<std::endl;
        const float_type dx_base=domain_->dx_base();
        auto refinement_level = level-domain_->tree()->base_level();
        auto dx_level =  dx_base/std::pow(2, refinement_level);

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

        //std::cout<<"FMM init base level masks" << std::endl;
        fmm_init_base_level_masks(domain_, level, non_leaf_as_source);
        //std::cout<<"FMM upward masks" << std::endl;
        fmm_upward_pass_masks(domain_, level);
        //std::cout<<"FMM sync masks" << std::endl;
        fmm_sync_masks(domain_, level);
        //fmm_upward_pass_masks(domain_, level);

        ////Initialize for each fmm // zero ing all tree
        fmm_init_zero<fmm_s>(domain_, level);
        fmm_init_zero<fmm_t>(domain_, level);

        //// Copy to temporary variables // only the base level
        fmm_init_copy<Source, fmm_s>(domain_, level);

        //// Anterpolation
        pcout<<"FMM Antrp start" << std::endl;
        fmm_antrp<fmm_s>(domain_, level);

        //// FMM Neighbors
        pcout<<"FMM B0 start" << std::endl;
        fmm_B0<fmm_s, fmm_t>(domain_, level, dx_level);

        //// FMM influence list
        pcout<<"FMM Bx start" << std::endl;
        //fmm_Bx_itr_build(domain_, level);
        fmm_Bx<fmm_s, fmm_t>(domain_, level, dx_level);

        //// Interpolation
        pcout<<"FMM INTRP start" << std::endl;
        fmm_intrp<fmm_t>(domain_, level);
        //std::cout<<"FMM INTRP done" << std::endl;

        //// Copy back
        if (!non_leaf_as_source)
            fmm_add_equal<Target, fmm_t>(domain_, level);
        else
            fmm_minus_equal<Target, fmm_t>(domain_, level);

        boost::mpi::communicator world;
        //std::cout<<"Rank "<<world.rank() << " FFTW_count = ";
        //std::cout<<conv_.fft_count << std::endl;
        pcout<<"FMM For Level "<< level << " End ---------------------------"<<std::endl;
    }

    void fmm_Bx_itr_build(domain_t* domain_, int base_level)
    {
        for (int level=base_level; level>=0; --level)
        {
            bool _neighbor = (level==base_level)? true:false;
            for (auto it = domain_->begin(level);
                    it != domain_->end(level);
                    ++it)
            {
                if (!(it->data()) || !it->mask(MASK_LIST::Mask_FMM_Target) )
                    continue;

                int recv_m_send_count =
                    domain_->decomposition().client()->template
                        communicate_induced_fields_recv_m_send_count<fmm_t, fmm_t>(it, _neighbor);

                Bx_itr.emplace(recv_m_send_count, *it);
            }
        }
    }

    void fmm_init_base_level_masks(domain_t* domain_, int base_level, bool non_leaf_as_source)
    {

        if (non_leaf_as_source)
        {
            for (auto it = domain_->begin(base_level);
                    it != domain_->end(base_level); ++it)
            {
                if ( it->is_leaf() || !it->locally_owned() )
                {
                    it->mask(MASK_LIST::Mask_FMM_Source, false);
                    it->mask(MASK_LIST::Mask_FMM_Target, false);
                } else
                {
                    it->mask(MASK_LIST::Mask_FMM_Source, true);
                    it->mask(MASK_LIST::Mask_FMM_Target, true);
                }
            }
        } else
        {
            for (auto it = domain_->begin(base_level);
                    it != domain_->end(base_level); ++it)
            if (it->locally_owned())
            {
                it->mask(MASK_LIST::Mask_FMM_Source, true);
                it->mask(MASK_LIST::Mask_FMM_Target, true);
            } else
            {
                it->mask(MASK_LIST::Mask_FMM_Source, false);
                it->mask(MASK_LIST::Mask_FMM_Target, false);
            }
        }
    }

    void fmm_upward_pass_masks(domain_t* domain_, int base_level)
    {
        // for all levels
        for (int level=base_level-1; level>=0; --level)
        {
            // parent's mask is true if any of its child's mask is true
            for (auto it = domain_->begin(level);
                    it != domain_->end(level);
                    ++it)
            {
                // including ghost parents
                it->mask(MASK_LIST::Mask_FMM_Source, false);
                for ( int c = 0; c < it->num_children(); ++c )
                {
                    if ( it->child(c) && it->child(c)->mask(MASK_LIST::Mask_FMM_Source) )
                    {
                        it->mask(MASK_LIST::Mask_FMM_Source, true);
                        break;
                    }
                }

                it->mask(MASK_LIST::Mask_FMM_Target, false);
                for ( int c = 0; c < it->num_children(); ++c)
                {
                    if ( it->child(c) && it->child(c)->mask(MASK_LIST::Mask_FMM_Target) )
                    {
                        it->mask(MASK_LIST::Mask_FMM_Target, true);
                        break;
                    }
                }

            }

            domain_->decomposition().client()-> template
                    communicate_mask_single_level_updownward_OR(level,
                            MASK_LIST::Mask_FMM_Source, true);

            domain_->decomposition().client()-> template
                    communicate_mask_single_level_updownward_OR(level,
                            MASK_LIST::Mask_FMM_Target, true);
        }
    }

    void fmm_sync_masks(domain_t* domain_, int base_level)
    {
        fmm_sync_parent_masks(domain_, base_level);
        //std::cout<<"FMM SYNC parent MASKS done" << std::endl;
        fmm_sync_inf_masks(domain_, base_level);
        //std::cout<<"FMM SYNC inf MASKS done" << std::endl;
        fmm_sync_child_mask(domain_, base_level);
        //std::cout<<"FMM SYNC child MASKS done" << std::endl;
    }

    void fmm_sync_parent_masks(domain_t* domain_, int base_level)
    {
        for (int level=base_level-1; level>=0; --level)
        {
            domain_->decomposition().client()-> template
                    communicate_mask_single_level_updownward_OR(level,
                            MASK_LIST::Mask_FMM_Source,false);

            domain_->decomposition().client()-> template
                    communicate_mask_single_level_updownward_OR(level,
                            MASK_LIST::Mask_FMM_Target,false);
        }
    }

    void fmm_sync_inf_masks(domain_t* domain_, int base_level)
    {

        for (int level=base_level; level>=0; --level)
        {
            bool neighbor_ = (level==base_level)? true:false;
            neighbor_ = true;

            domain_->decomposition().client()-> template
                    communicate_mask_single_level_inf_sync(level,
                            MASK_LIST::Mask_FMM_Source, neighbor_);

            domain_->decomposition().client()-> template
                    communicate_mask_single_level_inf_sync(level,
                            MASK_LIST::Mask_FMM_Target, neighbor_);
        }
    }

    void fmm_sync_child_mask(domain_t* domain_, int base_level)
    {

        for (int level=base_level-1; level>=0; --level)
        {

            domain_->decomposition().client()-> template
                    communicate_mask_single_level_child_sync(level,
                            MASK_LIST::Mask_FMM_Source);

            domain_->decomposition().client()-> template
                    communicate_mask_single_level_child_sync(level,
                            MASK_LIST::Mask_FMM_Target);
        }
    }


    auto initialize_upward_iterator(int level, domain_t* domain_,bool _upward)
    {
        std::vector<std::pair<octant_t*, int>> octants;
        for (auto it = domain_->begin(level); it != domain_->end(level); ++it)
        {
            int recv_m_send_count=domain_-> decomposition().client()->
                updownward_pass_mcount(*it,_upward);

            octants.emplace_back(std::make_pair(*it,recv_m_send_count));
        }
        //Sends=10000, recv1-10000, no_communication=0
        //descending order
        std::sort(octants.begin(), octants.end(),[&](const auto& e0, const auto& e1)
                {return e0.second> e1.second;  });
        return octants;
    }

    template<
        class s,
        class t
    >
    void fmm_Bx(domain_t* domain_,
                int base_level,
                float_type dx_level)
    {
        std::vector<std::pair<octant_t*, int>> octants;
        for (int level=base_level; level>=0; --level)
        {
            for (auto it = domain_->begin(level); it != domain_->end(level); ++it)
            {
                bool _neighbor = (level==base_level)? true:false;
                if (!(it->data()) || !it->mask(MASK_LIST::Mask_FMM_Target) )
                    continue;

                int recv_m_send_count =
                    domain_->decomposition().client()->template
                    communicate_induced_fields_recv_m_send_count<fmm_t, fmm_t>(it, _neighbor);

                octants.emplace_back(std::make_pair(*it,recv_m_send_count));
            }
        }
        //Sends=10000, recv1-10000, no_communication=0
        std::sort(octants.begin(), octants.end(),[&](const auto& e0, const auto& e1)
                {return e0.second> e1.second;  });

        const bool start_communication = false;
        bool combined_messages=false;

        for (auto B_it=octants.begin(); B_it!=octants.end(); ++B_it)
        {
            auto it =B_it->first;
            int level = it->level();

            bool _neighbor = (level==base_level)? true:false;
            if (!(it->data()) || !it->mask(MASK_LIST::Mask_FMM_Target) )
                continue;

            for (std::size_t i=0; i< it->influence_number(); ++i)
            {
                auto n_s = it->influence(i);
                if (n_s && n_s->locally_owned()
                        && n_s->mask(MASK_LIST::Mask_FMM_Source))
                {

                    fmm_tt<s,t>(n_s, it, base_level-level, dx_level);
                }
            }

            //setup the tasks
            domain_->decomposition().client()->template
                communicate_induced_fields<fmm_t, fmm_t>(it, _neighbor, start_communication);

            if(!combined_messages && B_it->second==0)
            {
                if(!combined_messages)
                {
                    domain_->decomposition().client()->template
                        combine_induced_field_messages<fmm_t, fmm_t>();
                    combined_messages=true;
                }
                domain_->decomposition().client()->template
                    check_combined_induced_field_communication<fmm_t,fmm_t>(false);
            }
        }

        //Finish the communication
        TIME_CODE(time_communication_Bx, SINGLE_ARG(
        domain_->decomposition().client()->template
            check_combined_induced_field_communication<fmm_t,fmm_t>(true);
        ))

        //boost::mpi::communicator w;
        //std::cout<<"Rank "<<w.rank()<<" "
        //<<"FMM time_communication_Bx: " <<time_communication_Bx.count()<<" "
        //<<std::endl;

    }

    template<
        class s,
        class t
    >
    void fmm_B0(domain_t* domain_,
                int base_level,
                float_type dx_level)
    {

        int level_diff = 0;

        for (auto it = domain_->begin(base_level);
                it != domain_->end(base_level); ++it)
        {
            if ( !it->mask(MASK_LIST::Mask_FMM_Target) ) continue;

            for (int i=0; i<27; ++i)
            {
                auto n_s = it->neighbor(i);

                if (n_s && n_s->locally_owned()
                        && n_s->mask(MASK_LIST::Mask_FMM_Source))
                {
                    fmm_tt<s,t>(n_s, it, level_diff, dx_level);
                }
            }
         }
    }

    template< class f1, class f2 >
    void fmm_add_equal(domain_t* domain_, int base_level)
    {

        for (auto it = domain_->begin(base_level);
                it != domain_->end(base_level);
                ++it)
        {
            if(it->data() && it->locally_owned() && it->mask(MASK_LIST::Mask_FMM_Target))
            {
                it->data()->template get_linalg<f1>().get()->
                    cube_noalias_view() +=
                                it->data()->template get_linalg_data<f2>();
            }
        }

    }

    template< class f1, class f2 >
    void fmm_minus_equal(domain_t* domain_, int base_level)
    {
        for (auto it = domain_->begin(base_level);
                it != domain_->end(base_level);
                ++it)
        {
            if(it->data() && it->locally_owned() && it->mask(MASK_LIST::Mask_FMM_Target))
            {

                it->data()->template get_linalg<f1>().get()->
                    cube_noalias_view() -=
                            it->data()->template get_linalg_data<f2>();
            }
        }

    }

    template< class fmm_s >
    void fmm_init_zero(domain_t* domain_, int base_level)
    {
        for (int level=base_level; level>=0; --level)
        {
            for (auto it = domain_->begin(level);
                    it != domain_->end(level);
                    ++it)
            {
                    if(it->data() /*&& it->mask(MASK_LIST::Mask_FMM_Source)*/)
                    {
                        for(auto& e: it->data()->template get_data<fmm_s>())
                            e=0.0;
                    }
            }
        }
    }


    template<
        class from,
        class to
    >
    void fmm_init_copy(domain_t* domain_, int base_level)
    {
        for (auto it = domain_->begin(base_level);
                it != domain_->end(base_level);
                ++it)
        {
            if(it->data() && it->locally_owned() && it->mask(MASK_LIST::Mask_FMM_Source))
            {
                it->data()->template get_linalg<to>().get()->
                    cube_noalias_view() =
                     it->data()->template get_linalg_data<from>() * 1.0;
            }
        }

    }

    template< class fmm_t >
    void fmm_intrp(domain_t* domain_, int base_level)
    {

        //for (int level=1; level<base_level; ++level)
        //{
        //    //sort octants such that internal cells are first
        //    auto octants=initialize_upward_iterator(level,domain_,false);
        //    bool finished=false;

        //    //Start communications
        //    for (auto B_it=octants.rbegin(); B_it!=octants.rend(); ++B_it)
        //    {
        //        auto it =B_it->first;
        //            domain_->decomposition().client()->
        //                template communicate_updownward_assign<fmm_t, fmm_t>(it, false);
        //        if(B_it->second ==0 ) break;
        //    }

        //    //Do inner communications first
        //    for (auto B_it=octants.begin(); B_it!=octants.end(); ++B_it)
        //    {
        //        auto it =B_it->first;
        //        if(it->data() && it->mask(MASK_LIST::Mask_FMM_Target) )
        //        {
        //            if(B_it->second<0 && !finished)
        //            {
        //                domain_->decomposition().client()-> template
        //                    finish_updownward_pass_communication_assign<fmm_t, fmm_t>();
        //                finished=true;
        //            }

        //            if(it->data() && it->mask(MASK_LIST::Mask_FMM_Target) )
        //                lagrange_intrp.nli_intrp_node<fmm_t>(it);
        //        }
        //    }//octants in level
        //}

        for (int level=1; level<base_level; ++level)
        {
            domain_->decomposition().client()-> template
                    communicate_updownward_assign<fmm_t, fmm_t>(level,false);

            for (auto it = domain_->begin(level);
                    it != domain_->end(level);
                    ++it)
            {
                if(it->data() && it->mask(MASK_LIST::Mask_FMM_Target) )
                    lagrange_intrp.nli_intrp_node<fmm_t>(it);
            }
        }
    }

    template< class fmm_s>
    void fmm_antrp(domain_t* domain_, int base_level)
    {
        //for (int level=base_level-1; level>=0; --level)
        //{
        //    auto octants=initialize_upward_iterator(level,domain_,true);
        //    for (auto B_it=octants.begin(); B_it!=octants.end(); ++B_it)
        //    {
        //        auto it =B_it->first;
        //        if(it->data() && it->mask(MASK_LIST::Mask_FMM_Source) )
        //            lagrange_intrp.nli_antrp_node<fmm_s>(it);

        //        domain_->decomposition().client()->
        //            template communicate_updownward_add<fmm_s, fmm_s>(it, true);
        //    }

        //    //domain_->decomposition().client()->
        //    //    template communicate_updownward_add<fmm_s, fmm_s>(level, true);

        //    for (auto it = domain_->begin(level);
        //            it != domain_->end(level);
        //            ++it)
        //        if (!it->locally_owned())
        //    {
        //            auto& cp2 = it ->data()->template get_linalg_data<fmm_s>();
        //            cp2*=0.0;
        //    }
        //}

        for (int level=base_level-1; level>=0; --level)
        {
            for (auto it = domain_->begin(level);
                    it != domain_->end(level);
                    ++it)
            {
                if(it->data() && it->mask(MASK_LIST::Mask_FMM_Source) )
                    lagrange_intrp.nli_antrp_node<fmm_s>(it);
            }

            domain_->decomposition().client()->
                template communicate_updownward_add<fmm_s, fmm_s>(level, true);

            for (auto it = domain_->begin(level);
                    it != domain_->end(level);
                    ++it)
                if (!it->locally_owned())
                {
                    auto& cp2 = it ->data()->template get_linalg_data<fmm_s>();
                    cp2*=0.0;
                }
        }
    }

    template<
        class S,
        class T,
        class octant_t,
        class octant_itr_t
    >
    void fmm_tt(octant_t o_s,
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
        std::multimap<int,octant_t*>     Bx_itr;
        parallel_ostream::ParallelOstream pcout=parallel_ostream::ParallelOstream(1);

    private: //timings

        mDuration_type time_communication_Bx;
        mDuration_type time_communication_B0;
        mDuration_type time_communication_interp;
        mDuration_type time_communication_anterp;

        mDuration_type time_fftw;
        mDuration_type time_interp;
        mDuration_type time_anterp;

        mDuration_type time_Bx_all;
        mDuration_type time_interp_all;
        mDuration_type time_anterp_all;

    };

}

#endif //IBLGF_INCLUDED_FMM_HPP
