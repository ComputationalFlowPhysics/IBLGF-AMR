#ifndef IBLGF_INCLUDED_FMM_HPP
#define IBLGF_INCLUDED_FMM_HPP

#include <iostream>     // std::cout
#include <algorithm>    // std::max/ min
#include <vector>


// IBLGF-specific
#include <types.hpp>
#include <linalg/linalg.hpp>
#include <typeinfo>

#include <domain/domain.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <domain/octree/tree.hpp>

namespace fmm
{
    class Nli
    {

    public: // constructor

        Nli() = delete;
        Nli(int Nb_)
            : Nb_(Nb_),
            antrp_(Nb_ * Nb_ * 2, 0.0),
            antrp_mat_(&(antrp_[0]), Nb_, Nb_ * 2),
            antrp_sub_{std::vector<float_type>(Nb_ * Nb_, 0.0),std::vector<float_type>(Nb_ * Nb_, 0.0)},
            antrp_mat_sub_{linalg::Mat_t(&antrp_sub_[0][0], Nb_, Nb_ ), linalg::Mat_t(&antrp_sub_[1][0], Nb_, Nb_ )}
        {
            std::cout <<"antrp_mem, "<< &antrp_[0]<< std::endl;
            std::cout << &(antrp_mat_.data_(0,0)) << std::endl;

            std::cout <<"antrp_sub_mem, "<< &antrp_sub_[0][0]<< std::endl;
            std::cout << &(antrp_mat_sub_[0].data_(0,0))<< std::endl;

            antrp_mat_calc(antrp_mat_.data_, Nb_);
            antrp_mat_sub_[0].data_ = xt::view(antrp_mat_.data_, xt::range(0, Nb_), xt::range( 0  , Nb_  ));
            antrp_mat_sub_[1].data_ = xt::view(antrp_mat_.data_, xt::range(0, Nb_), xt::range( Nb_, 2*Nb_));

            std::cout << antrp_mat_.data_ << std::endl;
            auto t1 = antrp_;
            for (auto i: t1)
                  std::cout << i << ' ';
            std::cout<<std::endl;
        }


    public: // functionalities

        template<template<size_t> class field>
        void nli_antrp_node(auto parent)
            {
                auto& parent_linalg_data = parent->data()->template get_linalg_data<field>();

                //std::cout<<parent->key() << std::endl;
                //std::cout <<parent_linalg_data << std::endl;
                for (int i = 0; i < parent->num_children(); ++i)
                {
                    auto child = parent->child(i);
                    if (child == nullptr) continue;

                    //std::cout<<"child # " << i << std::endl;
                    auto& child_linalg_data  = child ->data()->template get_linalg_data<field>();
                    nli_antrp_node(child_linalg_data, parent_linalg_data, i);
                }
                //std::cout << parent_linalg_data << std::endl;

            }


        void nli_antrp_node(auto& child, auto& parent, int child_idx)
        {
            int n = child.shape()[0];
            int idx_x = (child_idx & ( 1 << 0 )) >> 0;
            int idx_y = (child_idx & ( 1 << 1 )) >> 1;
            int idx_z = (child_idx & ( 1 << 2 )) >> 2;

            //todo make it not temporary ???
            xt::xtensor<float_type, 2> nli_aux_2d_antrp(std::array<size_t, 2>{{ n,n }});
            xt::xtensor<float_type, 3> nli_aux_3d_antrp(std::array<size_t, 3>{{ n,n,n }});

            for (int q = 0; q<n; ++q)
            {
                for (int l=0; l<n; ++l)
                {
                    // For Z
                    xt::noalias( view(nli_aux_2d_antrp, xt::all(), l) ) =
                        xt::linalg::dot( antrp_mat_sub_[idx_z].data_,
                                            view(child, q, l, xt::all()) );
                }

                for (int l=0; l<n; ++l)
                {
                    // For Y
                    xt::noalias( view(nli_aux_3d_antrp, q, l, xt::all()) ) =
                        xt::linalg::dot( antrp_mat_sub_[idx_y].data_,
                                            view(nli_aux_2d_antrp, l, xt::all()) );
                }
            }

            for (int p = 0; p <n; ++p)
            {
                for (int q = 0; q < n; ++q)
                {
                    // For X
                    xt::noalias( view(parent, xt::all(), q, p) ) +=
                        xt::linalg::dot( antrp_mat_sub_[idx_x].data_,
                                            view(nli_aux_3d_antrp, q, p, xt::all()) );
                }
            }
        }

        template<template<size_t> class field>
        void intrp(auto parent)
        {
        }



    private:
        void antrp_mat_calc(auto& antrp_mat_, int Nb_)
        {

            for (int c = 0; c < Nb_*2-1; ++c){

                int c_p = c - Nb_;

                if (c % 2 == 0){
                    int p = c / 2;
                    antrp_mat_(p, c) = 1;
                }
                else{
                    double temp_sum = 0.0;
                    int bd_l = -1;
                    int bd_r = -1;

                    if (c_p < -1){
                        bd_l = std::max(0   , ((c + 1) / 2  - pts_cap / 2 ));
                        bd_r = std::min(Nb_, bd_l + pts_cap);
                    }
                    else{
                        bd_r = std::min(Nb_ , ((c + 1) / 2 + pts_cap / 2));
                        bd_l = std::max(0   , bd_r - pts_cap);
                    }

                    for (int p = bd_l; p<bd_r; ++p){
                        double temp_mult = 1.0;
                        int p_p = p*2 - Nb_ + 1;

                        // the barycentric coefficients
                        for (int l = bd_l; l< bd_r; ++l){
                            if (l != p){
                                temp_mult /= -(2*l - Nb_ + 1 - p_p);
                            }
                        }

                        temp_mult          /= (double)(c_p - p_p + 1);
                        antrp_mat_(p, c)  = temp_mult;
                        temp_sum           += temp_mult;
                    }

                    view(antrp_mat_, xt::all(),  c) /= temp_sum;
                }

            }

            for (int c = Nb_*2 - 1; c>Nb_-1; --c){
                view(antrp_mat_, xt::all(),  c) =
                view(antrp_mat_, xt::all(),  c-1);
            }

        }


    //private:
    public:
        const int pts_cap = 10;

        // antrp mat

        int Nb_;
        std::vector<float_type> antrp_;
        linalg::Mat_t antrp_mat_;

        std::array<std::vector<float_type>,2> antrp_sub_;
        std::array<linalg::Mat_t, 2> antrp_mat_sub_;
    };

}

#endif //IBLGF_INCLUDED_FMM_HPP
