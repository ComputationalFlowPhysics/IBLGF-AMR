#ifndef IBLGF_INCLUDED_C_CENTER_NLI
#define IBLGF_INCLUDED_C_CENTER_NLI

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

namespace interpolation
{
    class cell_center_nli
    {

    public: // constructor

        cell_center_nli() = delete;
        cell_center_nli(size_t Nb_)
            : Nb_(Nb_),
            antrp_(Nb_ * Nb_ * 2, 0.0),
            antrp_mat_(&(antrp_[0]), Nb_, Nb_ * 2),
            antrp_sub_{std::vector<float_type>(Nb_ * Nb_, 0.0),std::vector<float_type>(Nb_ * Nb_, 0.0)},
            antrp_mat_sub_{linalg::Mat_t(&antrp_sub_[0][0], Nb_, Nb_ ), linalg::Mat_t(&antrp_sub_[1][0], Nb_, Nb_ )},
            nli_aux_1d_intrp(std::array<size_t, 1>{{ Nb_ }}),
            nli_aux_2d_intrp(std::array<size_t, 2>{{ Nb_, Nb_ }}),
            nli_aux_3d_intrp(std::array<size_t, 3>{{ Nb_, Nb_, Nb_ }}),

            nli_aux_2d_antrp(std::array<size_t, 2>{{ Nb_, Nb_ }}),
            nli_aux_3d_antrp(std::array<size_t, 3>{{ Nb_, Nb_, Nb_ }}),
            nli_aux_1d_antrp_tmp(std::array<size_t, 1>{{ Nb_}}),

            child_target_L_tmp(std::array<size_t, 3>{{(size_t)Nb_,(size_t)Nb_,(size_t)Nb_}})
        {
            antrp_mat_calc(antrp_mat_.data_, Nb_);
            antrp_mat_sub_[0].data_ = xt::view(antrp_mat_.data_, xt::range(0, Nb_), xt::range( 0  , Nb_  ));
            antrp_mat_sub_[1].data_ = xt::view(antrp_mat_.data_, xt::range(0, Nb_), xt::range( Nb_, 2*Nb_));
        }


    public: // functionalities

        template< class from,
         class to,
        typename octant_t
        >
        void add_source_correction(octant_t parent, double dx)
            {

                for (int i = 0; i < parent->num_children(); ++i)
                {
                    auto child = parent->child(i);
                    if (!child ) continue;
                    if (!child->locally_owned()) continue;
                    if (!child->data()) continue;

                    //child_target_L_tmp *= 0.0;
                    auto& child_target_tmp  = child ->data()->template get_linalg_data<from>();

                    auto& child_linalg_data  = child ->data()->template get_linalg_data<to>();
                    //child_linalg_data -= (child_target_L_tmp * (1.0/(dx *dx)));

                    for ( int i =1; i<Nb_-1; ++i){
                        for ( int j = 1; j<Nb_-1; ++j){
                            for ( int k = 1; k<Nb_-1; ++k){
                                // differences in definition of mem layout
                                //child_target_L_tmp(i,j,k)  = - 6.0 * child_target_tmp(i,j,k);
                                //child_target_L_tmp(i,j,k) += child_target_tmp(i,j,k-1);
                                //child_target_L_tmp(i,j,k) += child_target_tmp(i,j,k+1);
                                //child_target_L_tmp(i,j,k) += child_target_tmp(i,j-1,k);
                                //child_target_L_tmp(i,j,k) += child_target_tmp(i,j+1,k);
                                //child_target_L_tmp(i,j,k) += child_target_tmp(i+1,j,k);
                                //child_target_L_tmp(i,j,k) += child_target_tmp(i-1,j,k);

                                child_linalg_data(i,j,k) += 6.0 * child_target_tmp(i,j,k) * (1.0/(dx *dx));
                                child_linalg_data(i,j,k) -= child_target_tmp(i,j,k-1) * (1.0/(dx *dx));
                                child_linalg_data(i,j,k) -= child_target_tmp(i,j,k+1) * (1.0/(dx *dx));
                                child_linalg_data(i,j,k) -= child_target_tmp(i,j-1,k) * (1.0/(dx *dx));
                                child_linalg_data(i,j,k) -= child_target_tmp(i,j+1,k) * (1.0/(dx *dx));
                                child_linalg_data(i,j,k) -= child_target_tmp(i+1,j,k) * (1.0/(dx *dx));
                                child_linalg_data(i,j,k) -= child_target_tmp(i-1,j,k) * (1.0/(dx *dx));


                                //if(std::isnan(child_target_L_tmp(i,j,k)))
                                //{
                                //    std::cout<<"LHS"<<std::endl;
                                //    std::cout<<"this is nan at level = " << child->level()<<std::endl;
                                //    std::cout<<"parent locally owned" << parent->locally_owned()<<std::endl;
                                //}
                                //if(std::isnan(child_target_tmp(i,j,k)))
                                //{
                                //    std::cout<<"RHS"<<std::endl;
                                //    std::cout<<"this is nan at level = " << child->level()<<std::endl;
                                //    std::cout<<"parent locally owned" << parent->locally_owned()<<std::endl;
                                //}

                            }
                        }
                    }

                }
            }


        template< class from,
             class to,
            typename octant_t
        >
        void nli_intrp_node(octant_t parent)
            {
                auto& parent_linalg_data = parent->data()->template get_linalg_data<from>();

                for (int i = 0; i < parent->num_children(); ++i)
                {
                    auto child = parent->child(i);
                    if (child == nullptr || !child ->data()) continue;

                    auto& child_linalg_data  = child ->data()->template get_linalg_data<to>();
                    nli_intrp_node(child_linalg_data, parent_linalg_data, i);

                }

            }


        template<typename linalg_data_t>
        void nli_intrp_node(linalg_data_t& child, linalg_data_t& parent, int child_idx)
        {
            int n = child.shape()[0];
            int idx_x = (child_idx & ( 1 << 0 )) >> 0;
            int idx_y = (child_idx & ( 1 << 1 )) >> 1;
            int idx_z = (child_idx & ( 1 << 2 )) >> 2;

            for (int q = 0; q<n; ++q)
            {
                for (int l=0; l<n; ++l)
                {
                    // Column major
                    xt::noalias( view( nli_aux_2d_intrp, xt::all(), l )) =
                        xt::linalg::dot( view(parent, xt::all(), l, q), antrp_mat_sub_[idx_x].data_ );

                }

                for (int l=0; l<n; ++l)
                {
                    // For Y
                    // Column major
                    xt::noalias( view(nli_aux_3d_intrp, l, xt::all(), q) ) =
                        xt::linalg::dot( view(nli_aux_2d_intrp, l, xt::all()), antrp_mat_sub_[idx_y].data_);
                }
            }

            for (int p = 0; p <n; ++p)
            {
                for (int q = 0; q < n; ++q)
                {
                    // For Z
                    // Column major
                    xt::noalias( view(child, q, p, xt::all()) ) +=
                        xt::linalg::dot( view(nli_aux_3d_intrp, q, p, xt::all()), antrp_mat_sub_[idx_z].data_ );
                }
            }
        }


        template< class field, typename octant_t>
        void nli_antrp_node(octant_t parent)
            {
                auto& parent_linalg_data = parent->data()->template get_linalg_data<field>();

                for (int i = 0; i < parent->num_children(); ++i)
                {

                    auto child = parent->child(i);
                    if (child == nullptr) continue;

                    auto& child_linalg_data  = child ->data()->template get_linalg_data<field>();
                    nli_antrp_node(child_linalg_data, parent_linalg_data, i);
                }


            }


        template<typename linalg_data_t>
        void nli_antrp_node(linalg_data_t& child, linalg_data_t& parent,
                int child_idx)
        {
            int n = child.shape()[0];
            int idx_x = (child_idx & ( 1 << 0 )) >> 0;
            int idx_y = (child_idx & ( 1 << 1 )) >> 1;
            int idx_z = (child_idx & ( 1 << 2 )) >> 2;

            for (int q = 0; q<n; ++q)
            {
                for (int l=0; l<n; ++l)
                {
                   // Column major
                    xt::noalias(nli_aux_1d_antrp_tmp)= view(child, q, l, xt::all()) ;
                    xt::noalias( view(nli_aux_2d_antrp, xt::all(), l) ) =
                        xt::linalg::dot( antrp_mat_sub_[idx_z].data_,
                                            nli_aux_1d_antrp_tmp );
                }

                for (int l=0; l<n; ++l)
                {
                    // For Y
                    // Column major
                    xt::noalias( view(nli_aux_3d_antrp, xt::all(), l, q) ) =
                        xt::linalg::dot( antrp_mat_sub_[idx_y].data_,
                                            view(nli_aux_2d_antrp, l, xt::all()) );
                }
            }

            for (int p = 0; p < n; ++p)
            {
                for (int q = 0; q < n; ++q)
                {
                    // For X
                    // Column major
                    xt::noalias( view(parent, xt::all(), q, p) ) +=
                        xt::linalg::dot( antrp_mat_sub_[idx_x].data_,
                                           view(nli_aux_3d_antrp, q,p,xt::all()));
                }
            }
        }



    private:
        template<typename linalg_data_t>
        void antrp_mat_calc(linalg_data_t& antrp_mat_, int Nb_)
        {

            for (int c = 1; c < Nb_*2-1; ++c){

                double c_p = c - Nb_ + 0.5;

                //if (c % 2 == 0){
                //    int p = c / 2;
                //    antrp_mat_(p, c-1) = 1.0;
                //}
                //else
                    long double temp_sum = 0.0;
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

                    for (int p = bd_l; p<bd_r; ++p)
                    {
                        long double temp_mult = 1.0;
                        int p_p = p*2 - Nb_  + 1;

                        // the barycentric coefficients
                        for (int l = bd_l; l< bd_r; ++l){
                            if (l != p){
                                temp_mult /= -(long double)(2*l - Nb_ + 1 - p_p);
                            }
                        }

                        temp_mult          /= (long double)(c_p - p_p);
                        antrp_mat_(p, c-1)  = (double)temp_mult;
                        temp_sum           += temp_mult;

                    }
                    view(antrp_mat_, xt::all(),  c-1) /= (double)temp_sum;

            }

            for (int c = Nb_*2 - 1; c>Nb_-1; --c){
                view(antrp_mat_, xt::all(),  c) =
                view(antrp_mat_, xt::all(),  c-2);
            }

        }

    //private:
    public:
        const int pts_cap = 3;

        // antrp mat

        int Nb_;
        std::vector<float_type> antrp_;
        linalg::Mat_t antrp_mat_;

        std::array<std::vector<float_type>,2> antrp_sub_;
        std::array<linalg::Mat_t, 2> antrp_mat_sub_;

    private:
        xt::xtensor<float_type, 1> nli_aux_1d_intrp;
        xt::xtensor<float_type, 2> nli_aux_2d_intrp;
        xt::xtensor<float_type, 3> nli_aux_3d_intrp;

        xt::xtensor<float_type, 2> nli_aux_2d_antrp;
        xt::xtensor<float_type, 3> nli_aux_3d_antrp;
        xt::xtensor<float_type, 1> nli_aux_1d_antrp_tmp;

        xt::xtensor<float_type,3>
            child_target_L_tmp;
    };

}

#endif //IBLGF_INCLUDED_FMM_HPP
