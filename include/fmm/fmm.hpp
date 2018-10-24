#ifndef IBLGF_INCLUDED_FMM
#define IBLGF_INCLUDED_FMM

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

// IBLGF-specific
#include <types.hpp>
#include <fmm/fmm_nli.hpp>
#include <domain/octree/tree.hpp>


//test
#include <global.hpp>
#include <simulation.hpp>
#include <linalg/linalg.hpp>
#include <domain/domain.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <domain/octree/tree.hpp>
#include <IO/parallel_ostream.hpp>
#include "../utilities/convolution.hpp"


namespace fmm
{

    class Fmm
    {

    static constexpr int fmm_lBuffer=0; ///< Lower left buffer for interpolation
    static constexpr int fmm_rBuffer=1; ///< Lower left buffer for interpolation

    public:
        Fmm(int Nb)
        :lagrange_intrp(Nb)
        {

        }

        template<
            template<size_t> class Source,
            template<size_t> class Target,
            template<size_t> class fmm_s,
            template<size_t> class fmm_t
            >
        void fmm_for_level(auto& domain_, int level, bool for_non_leaf = false)
        {

            std::cout << "------------------------------------"  << std::endl;
            std::cout << "Fmm - Level - " << level << std::endl;

            // Find the subtree
            auto o_start = domain_-> begin(level);
            auto o_end   = domain_-> end(level);
            o_end--;

            if (for_non_leaf)
            {
                while (o_start->is_leaf()==true) o_start++;
                while (o_end->is_leaf()==true) o_end--;
            }

            // Initialize for each fmm // zero ing all tree
            std::cout << "Fmm - initialize " << std::endl;
            fmm_init_zero<fmm_s>(domain_, level, o_start, o_end);
            fmm_init_zero<fmm_t>(domain_, level, o_start, o_end);

            // Copy to temporary variables // only the base level
            std::cout << "Fmm - copy source " << std::endl;
            fmm_init_copy<Source, fmm_s>(domain_, level, o_start, o_end);

            // Antrp for all // from base level up
            std::cout << "Fmm - antrp " << std::endl;
            fmm_antrp<fmm_s>(domain_, level, o_start, o_end);

            // Nearest neighbors and self
            std::cout << "Fmm - b0 " << std::endl;
            fmm_b0<fmm_s, fmm_t>(domain_, level, o_start, o_end);

            // FMM 196
            std::cout << "Fmm - b1 and up" << std::endl;

            // Intrp

        }

        auto common_ancestor(auto octant_1, auto octant_2)
        {

            if (octant_1->level() != octant_2->level())
                throw std::runtime_error("Level has to be the same");

            // ??? here just because octant_1 is the iterator type.
            auto o_1 = octant_1->self();
            auto o_2 = octant_2->self();

            while (o_1->key() != o_2->key())
            {
                o_1 = o_1->parent();
                o_2 = o_2->parent();

                std::cout<<*o_1<<std::endl;
                std::cout<<*o_2<<std::endl;
            }

            return o_1;
        }

        void key_range(auto& domain_, int level, bool for_non_leaf = false)
        {
            for (auto it  = domain_->begin(level);
                      it != domain_->end(level); ++it)
            {
                if (for_non_leaf)
                    if (it->is_leaf())
                    {
                    }

            }
        }

        template<
            template<size_t> class s,
            template<size_t> class t
        >
        void fmm_b0(auto& domain_, int level, auto o_start, auto o_end)
        {
            if (o_start->level() != o_end->level())
                throw std::runtime_error("Level has to be the same");
            o_end++;

            for (auto it_t = o_start; it_t!=(o_end); ++it_t)
            {
                //find neighbors


                std::cout<<"find neibor test"<< std::endl;
                std::cout<<"self"<< std::endl;
                std::cout<<it_t->key() << std::endl;
                std::cout<<"neighbors"<< std::endl;



                for (int i=0; i<27; ++i)
                {
                    auto n = it_t->neighbor(i);
                    if (n)
                    {
                    }
                    else
                    {
                        int id = i;
                        int tmp = id;
                        auto idx = tmp % 3 -1;
                        tmp /=3;
                        auto idy = tmp % 3 -1;
                        tmp /=3;
                        auto idz = tmp % 3 -1;

                    std::cout<<it_t->key().neighbor({{idx,idy,idz}})<< std::endl;
                    }
                }


                //if (k2 == nullptr)
                //std::cout<<"No luck"<< std::endl;
                //else
                //std::cout<<k2<< std::endl;


                //
                //for (auto it_s = it_t->neighbors.begin();
                //        auto it_s != it_t->neighbors.end(); it_s++)
                //{

                //}
            }

        }


        template<template<size_t> class f>
        void fmm_init_zero(auto& domain_, int level, auto o_start, auto o_end)
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
            template<size_t> class from,
            template<size_t> class to
            >
        void fmm_init_copy(auto& domain_, int level, auto o_start, auto o_end)
        {
            if (o_start->level() != o_end->level())
                throw std::runtime_error("Level has to be the same");

            o_end++;
            for (auto it = o_start; it!=(o_end); ++it)
            {
                 if(it->data())
                 {

                    it->data()->template get_linalg_data<to>()=
                         it->data()->template get_linalg_data<from>()*1.0;

                     //for(auto& e: it->data()->template get_data<to>())
                     //    std::cout<< e << ", ";

                    //auto& d  = it->data()->template get_data<to>();
                    //auto& d2 = it->data()->template get_linalg_data<to>();

                    //std::cout<< "===="<< std::endl;
                    //std::cout<< &d[0] << std::endl;
                    //std::cout<< &(d2(0,0,0)) << std::endl;
                    //std::cout<<std::endl;

                    }
                }

        }

        template< template<size_t> class fmm_s >
        void fmm_antrp(auto& domain_, int level, auto o_start, auto o_end)
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

    public:
        Nli lagrange_intrp;
    };

}

#endif //IBLGF_INCLUDED_FMM_HPP
