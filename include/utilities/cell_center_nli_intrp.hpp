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

    using MeshObject = domain::MeshObject;


    public: // constructor

        cell_center_nli() = delete;
        cell_center_nli(size_t Nb_)
            : Nb_(Nb_),
            antrp_relative_pos_0_(Nb_ * Nb_ * 2, 0.0),
            antrp_mat_relative_pos_0_(&(antrp_relative_pos_0_[0]), Nb_, Nb_ * 2),
            antrp_relative_pos_1_(Nb_ * Nb_ * 2, 0.0),
            antrp_mat_relative_pos_1_(&(antrp_relative_pos_1_[0]), Nb_, Nb_ * 2),
            antrp_sub_{
                        std::vector<float_type>(Nb_ * Nb_, 0.0),
                        std::vector<float_type>(Nb_ * Nb_, 0.0),
                        std::vector<float_type>(Nb_ * Nb_, 0.0),
                        std::vector<float_type>(Nb_ * Nb_, 0.0)
                        },
            antrp_mat_sub_{
                        linalg::Mat_t(&antrp_sub_[0][0], Nb_, Nb_ ),
                        linalg::Mat_t(&antrp_sub_[1][0], Nb_, Nb_ ),
                        linalg::Mat_t(&antrp_sub_[2][0], Nb_, Nb_ ),
                        linalg::Mat_t(&antrp_sub_[3][0], Nb_, Nb_ )
                        },
            nli_aux_1d_intrp(std::array<size_t, 1>{{ Nb_ }}),
            nli_aux_2d_intrp(std::array<size_t, 2>{{ Nb_, Nb_ }}),
            nli_aux_3d_intrp(std::array<size_t, 3>{{ Nb_, Nb_, Nb_ }}),
            nli_aux_2d_antrp(std::array<size_t, 2>{{ Nb_, Nb_ }}),
            nli_aux_3d_antrp(std::array<size_t, 3>{{ Nb_, Nb_, Nb_ }}),
            nli_aux_1d_antrp_tmp(std::array<size_t, 1>{{ Nb_}}),
            child_combine_(std::array<size_t, 3>{{2*Nb_,2*Nb_,2*Nb_}}),
            antrp_mat_sub_simple_{
                            xt::xtensor<float_type, 2>(std::array<size_t, 2>{{Nb_,Nb_*2-2}}),
                            xt::xtensor<float_type, 2>(std::array<size_t, 2>{{Nb_,Nb_*2-2}})
                        },
            antrp_mat_sub_simple_sub_{
                            xt::xtensor<float_type, 2>(std::array<size_t, 2>{{Nb_,Nb_}}),
                            xt::xtensor<float_type, 2>(std::array<size_t, 2>{{Nb_,Nb_}}),
                            xt::xtensor<float_type, 2>(std::array<size_t, 2>{{Nb_,Nb_}}),
                            xt::xtensor<float_type, 2>(std::array<size_t, 2>{{Nb_,Nb_}})
                        }
        {
            antrp_mat_relative_pos_0_calc(antrp_mat_relative_pos_0_.data_, Nb_);
            antrp_mat_relative_pos_1_calc(antrp_mat_relative_pos_1_.data_, Nb_);

            antrp_mat_sub_[0].data_ =
                xt::view(antrp_mat_relative_pos_0_.data_,xt::range(0,Nb_),xt::range(0  ,Nb_  ));
            antrp_mat_sub_[1].data_ =
                xt::view(antrp_mat_relative_pos_0_.data_,xt::range(0,Nb_),xt::range(Nb_,2*Nb_));

            antrp_mat_sub_[2].data_ =
                xt::view(antrp_mat_relative_pos_1_.data_,xt::range(0,Nb_),xt::range(0  ,Nb_  ));
            antrp_mat_sub_[3].data_ =
                xt::view(antrp_mat_relative_pos_1_.data_,xt::range(0,Nb_),xt::range(Nb_,2*Nb_));

            //antrp_mat_sub_2_ = antrp_mat_sub_;

            // TODO switch to the commutative coarsifying instead of the simple
            // one used here
            std::fill(antrp_mat_sub_simple_[0].begin(), antrp_mat_sub_simple_[0].end(), 0.0);
            std::fill(antrp_mat_sub_simple_[1].begin(), antrp_mat_sub_simple_[1].end(), 0.0);

            for (size_t i=1;i<Nb_-1;++i)
                    antrp_mat_sub_simple_[0](i,i*2-1)=2.0;

            for (size_t i=1;i<Nb_-1;++i)
            {
                    antrp_mat_sub_simple_[1](i,i*2-1)=1.0;
                    antrp_mat_sub_simple_[1](i,i*2)=1.0;
            }

            xt::noalias(antrp_mat_sub_simple_sub_[0]) =
                view(antrp_mat_sub_simple_[0],xt::all(),xt::range(0,Nb_));
            xt::noalias(antrp_mat_sub_simple_sub_[1]) =
                view(antrp_mat_sub_simple_[0],xt::all(),xt::range(Nb_-2,2*Nb_-2));

            xt::noalias(antrp_mat_sub_simple_sub_[2]) =
                view(antrp_mat_sub_simple_[1],xt::all(),xt::range(0,Nb_));
            xt::noalias(antrp_mat_sub_simple_sub_[3]) =
                view(antrp_mat_sub_simple_[1],xt::all(),xt::range(Nb_-2,2*Nb_-2));

            //std::cout<< antrp_mat_sub_simple_[0]<< std::endl;
            //std::cout<< antrp_mat_sub_simple_[1]<< std::endl;

        }


    public: // functionalities

        template<
            class from,
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
                    if (!child->data()->is_allocated()) continue;

                    auto& child_target_tmp  = child ->data()->template get_linalg_data<from>();

                    auto& child_linalg_data  = child ->data()->template get_linalg_data<to>();
                    for ( int i =1; i<Nb_-1; ++i){
                        for ( int j = 1; j<Nb_-1; ++j){
                            for ( int k = 1; k<Nb_-1; ++k){
                                child_linalg_data(i,j,k) += 6.0 * child_target_tmp(i,j,k) * (1.0/(dx *dx));
                                child_linalg_data(i,j,k) -= child_target_tmp(i,j,k-1) * (1.0/(dx *dx));
                                child_linalg_data(i,j,k) -= child_target_tmp(i,j,k+1) * (1.0/(dx *dx));
                                child_linalg_data(i,j,k) -= child_target_tmp(i,j-1,k) * (1.0/(dx *dx));
                                child_linalg_data(i,j,k) -= child_target_tmp(i,j+1,k) * (1.0/(dx *dx));
                                child_linalg_data(i,j,k) -= child_target_tmp(i+1,j,k) * (1.0/(dx *dx));
                                child_linalg_data(i,j,k) -= child_target_tmp(i-1,j,k) * (1.0/(dx *dx));
                            }
                        }
                    }

                }
            }


        template< class from,
             class to,
            typename octant_t
        >
        void nli_intrp_node(octant_t parent, MeshObject mesh_obj, std::size_t _field_idx , bool correction_only = false, bool exclude_correction = false)
            {
                auto& parent_linalg_data = parent->data()->template get_linalg_data<from>();

                for (int i = 0; i < parent->num_children(); ++i)
                {
                    auto child = parent->child(i);
                    if (child == nullptr ||
                            !child ->data() ||
                            !child->data()->is_allocated())
                        continue;

                    if (correction_only && !child->is_correction())
                        continue;

                    if (exclude_correction && child->is_correction())
                        continue;

                    auto& child_linalg_data = child ->data()->template get_linalg_data<to>();
                    nli_intrp_node(child_linalg_data, parent_linalg_data,
                            i, mesh_obj, _field_idx);
                }

            }


        template<typename linalg_data_t>
        void nli_intrp_node(linalg_data_t& child, linalg_data_t& parent, int child_idx, MeshObject mesh_obj, std::size_t _field_idx )
        {
            int n = child.shape()[0];
            int idx_x = (child_idx & ( 1 << 0 )) >> 0;
            int idx_y = (child_idx & ( 1 << 1 )) >> 1;
            int idx_z = (child_idx & ( 1 << 2 )) >> 2;

            // Relative position 0 -> coincide with child
            // Relative position 1 -> half cell off with the child

            std::array<int, 3> relative_positions{{1,1,1}};
            if (mesh_obj == MeshObject::face)
                relative_positions[_field_idx]=0;
            else if (mesh_obj == MeshObject::cell)
            {
            }
            else if (mesh_obj == MeshObject::edge)
            {
                relative_positions[0]=0;
                relative_positions[1]=0;
                relative_positions[2]=0;
                relative_positions[_field_idx]=1;
            }
            else
                throw std::runtime_error(
                        "Wrong type of mesh to be interpolated");

            idx_x += relative_positions[0]*max_relative_pos;
            idx_y += relative_positions[1]*max_relative_pos;
            idx_z += relative_positions[2]*max_relative_pos;

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


        template< class from,
             class to,
            typename octant_t
        >
        void nli_antrp_node(octant_t parent, MeshObject mesh_obj, std::size_t _field_idx)
            {
                auto& parent_linalg_data = parent->data()->template get_linalg_data<to>();

                for (int i = 0; i < parent->num_children(); ++i)
                {
                    auto child = parent->child(i);
                    if (child == nullptr ||
                            !child ->data() ||
                            !child->data()->is_allocated())
                        continue;

                    if (child->is_correction())
                        continue;

                    auto& child_linalg_data = child ->data()->
                        template get_linalg_data<from>();

                    nli_antrp_node(child_linalg_data, parent_linalg_data, i, mesh_obj, _field_idx);
                }
            }


        template<typename linalg_data_t>
        void nli_antrp_node(linalg_data_t& child, linalg_data_t& parent,
                int child_idx, MeshObject mesh_obj, std::size_t _field_idx )
        {
            int n = child.shape()[0];
            int idx_x = (child_idx & ( 1 << 0 )) >> 0;
            int idx_y = (child_idx & ( 1 << 1 )) >> 1;
            int idx_z = (child_idx & ( 1 << 2 )) >> 2;
            xt::noalias( view(child,(1-idx_x)*(Nb_-1),xt::all(),xt::all()) ) *=0.0;
            xt::noalias( view(child,xt::all(),(1-idx_y)*(Nb_-1),xt::all()) ) *=0.0;
            xt::noalias( view(child,xt::all(),xt::all(),(1-idx_z)*(Nb_-1)) ) *=0.0;

            // Relative position 0 -> coincide with child
            // Relative position 1 -> half cell off with the child

            std::array<int, 3> relative_positions{{1,1,1}};
            if (mesh_obj == MeshObject::face)
            {
                relative_positions[_field_idx]=0;
            }
            else if (mesh_obj == MeshObject::cell)
            {
            }
            else if (mesh_obj == MeshObject::edge)
            {
                relative_positions[0]=0;
                relative_positions[1]=0;
                relative_positions[2]=0;
                relative_positions[_field_idx]=1;
            }
            else
                throw std::runtime_error(
                        "Wrong type of mesh to be interpolated");

            idx_x += relative_positions[0]*max_relative_pos;
            idx_y += relative_positions[1]*max_relative_pos;
            idx_z += relative_positions[2]*max_relative_pos;

            for (int q = 0; q<n; ++q)
            {
                for (int l=0; l<n; ++l)
                {
                   // Column major
                    //TODO swith back to commutative coarsifying
                    //now using simple
                    xt::noalias(nli_aux_1d_antrp_tmp)= view(child, q, l, xt::all()) ;
                    xt::noalias( view(nli_aux_2d_antrp, xt::all(), l) ) =
                        xt::linalg::dot( antrp_mat_sub_simple_sub_[idx_z],
                                            nli_aux_1d_antrp_tmp );
                }

                for (int l=0; l<n; ++l)
                {
                    // For Y
                    // Column major
                    xt::noalias( view(nli_aux_3d_antrp, xt::all(), l, q) ) =
                        xt::linalg::dot( antrp_mat_sub_simple_sub_[idx_y],
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
                        xt::linalg::dot( antrp_mat_sub_simple_sub_[idx_x],
                                           view(nli_aux_3d_antrp, q,p,xt::all()))/8.0;
                }
            }
        }



    private:
        template<typename linalg_data_t>
        void antrp_mat_relative_pos_0_calc(linalg_data_t& antrp_mat_, int Nb_)
        {

            for (int c = 1; c < Nb_*2-1; ++c){

                int c_p = c - Nb_;

                if (c % 2 == 0){
                    int p = c / 2;
                    antrp_mat_(p, c-1) = 1.0;
                }
                else{
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
                        int p_p = p*2 - Nb_ + 1;

                        // the barycentric coefficients
                        for (int l = bd_l; l< bd_r; ++l){
                            if (l != p){
                                temp_mult /= -(long double)(2*l - Nb_ + 1 - p_p);
                            }
                        }

                        temp_mult          /= (long double)(c_p - p_p + 1);
                        antrp_mat_(p, c-1)  = (double)temp_mult;
                        temp_sum           += temp_mult;
                    }
                    view(antrp_mat_, xt::all(),  c-1) /= (double)temp_sum;
                }

            }

            for (int c = Nb_*2 - 1; c>Nb_-1; --c){
                view(antrp_mat_, xt::all(),  c) =
                view(antrp_mat_, xt::all(),  c-2);
            }

        }


        template<typename linalg_data_t>
        void antrp_mat_relative_pos_1_calc(linalg_data_t& antrp_mat_relative_pos_1_, int Nb_)
        {

            for (int c = 1; c < Nb_*2-1; ++c){

                double c_p = c - Nb_ + 0.5;

                //if (c % 2 == 0){
                //    int p = c / 2;
                //    antrp_mat_relative_pos_1_(p, c-1) = 1.0;
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
                        antrp_mat_relative_pos_1_(p, c-1)  = (double)temp_mult;
                        temp_sum           += temp_mult;

                    }
                    view(antrp_mat_relative_pos_1_, xt::all(),  c-1) /= (double)temp_sum;

            }

            for (int c = Nb_*2 - 1; c>Nb_-1; --c){
                view(antrp_mat_relative_pos_1_, xt::all(),  c) =
                view(antrp_mat_relative_pos_1_, xt::all(),  c-2);
            }

        }

    //private:
    public:
        const int pts_cap = 6;
        int Nb_;

        std::vector<float_type> antrp_relative_pos_0_;
        linalg::Mat_t antrp_mat_relative_pos_0_;
        std::vector<float_type> antrp_relative_pos_1_;
        linalg::Mat_t antrp_mat_relative_pos_1_;

        static const int max_1D_child_n = 2;
        static const int max_relative_pos = 2;

        std::array<std::vector<float_type>,max_1D_child_n*max_relative_pos> antrp_sub_;
        std::array<linalg::Mat_t, max_1D_child_n*max_relative_pos> antrp_mat_sub_;

    private:
        xt::xtensor<float_type, 1> nli_aux_1d_intrp;
        xt::xtensor<float_type, 2> nli_aux_2d_intrp;
        xt::xtensor<float_type, 3> nli_aux_3d_intrp;

        xt::xtensor<float_type, 2> nli_aux_2d_antrp;
        xt::xtensor<float_type, 3> nli_aux_3d_antrp;
        xt::xtensor<float_type, 1> nli_aux_1d_antrp_tmp;

        xt::xtensor<float_type, 3> child_combine_;

        std::array<xt::xtensor<float_type, 2>, max_relative_pos> antrp_mat_sub_simple_;
        std::array<xt::xtensor<float_type, 2>, max_1D_child_n*max_relative_pos> antrp_mat_sub_simple_sub_;
    };

}

#endif //IBLGF_INCLUDED_FMM_HPP
