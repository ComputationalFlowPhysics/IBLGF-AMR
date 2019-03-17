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
        Nli(size_t Nb_)
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
            nli_aux_1d_antrp_tmp(std::array<size_t, 1>{{ Nb_}})
        {
            antrp_mat_calc(antrp_mat_.data_, Nb_);

            antrp_mat_sub_[0].data_ =
                xt::view(antrp_mat_.data_,
                        xt::range(0, Nb_), xt::range( 0  , Nb_  ));

            antrp_mat_sub_[1].data_ =
                xt::view(antrp_mat_.data_,
                        xt::range(0, Nb_), xt::range( Nb_, 2*Nb_));
        }


    public: // functionalities

        template<class field,
            typename octant_t>
        void nli_intrp_node(octant_t parent, int mask_id)
            {
                auto& parent_linalg_data =
                    parent->data()->template get_linalg_data<field>();

                for (int i = 0; i < parent->num_children(); ++i)
                {
                    auto child = parent->child(i);
                    if (child == nullptr || !child->mask(mask_id) /*|| !child->locally_owned()*/) continue;


                    auto& child_linalg_data =
                        child ->data()->template get_linalg_data<field>();

                    nli_intrp_node(child_linalg_data, parent_linalg_data, i);
                }

            }


        template<typename linalg_data_t>
        void nli_intrp_node(linalg_data_t& child, linalg_data_t& parent,
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
                    // Row major
                    //xt::noalias(nli_aux_1d_intrp) =
                    //      view(parent, q, l, xt::all()) * 1.0;
                    //xt::noalias( view( nli_aux_2d_intrp, l, xt::all() )) =
                    //    xt::linalg::dot( nli_aux_1d_intrp, antrp_mat_sub_[idx_x].data_ );

                    // Column major
                    xt::noalias( view( nli_aux_2d_intrp, xt::all(), l )) =
                        xt::linalg::dot( view(parent, xt::all(), l, q),
                                         antrp_mat_sub_[idx_x].data_ );
                }

                for (int l=0; l<n; ++l)
                {
                    // For Y
                    // Row major
                    //xt::noalias(nli_aux_1d_intrp) =
                    //      view(nli_aux_2d_intrp, xt::all(), l) * 1.0;
                    //xt::noalias( view(nli_aux_3d_intrp, q, xt::all(), l) ) =
                    //    xt::linalg::dot(nli_aux_1d_intrp, antrp_mat_sub_[idx_y].data_);

                    // Column major
                    xt::noalias( view(nli_aux_3d_intrp, l, xt::all(), q) ) =
                        xt::linalg::dot( view(nli_aux_2d_intrp, l, xt::all()),
                                         antrp_mat_sub_[idx_y].data_);
                }
            }

            for (int p = 0; p <n; ++p)
            {
                for (int q = 0; q < n; ++q)
                {
                    // For Z
                    // Row major
                    //xt::noalias(nli_aux_1d_intrp) =
                    //      view(nli_aux_3d_intrp, xt::all(), p, q) ;

                    //xt::noalias( view(child, xt::all(), p, q) ) +=
                    //    xt::linalg::dot(nli_aux_1d_intrp, antrp_mat_sub_[idx_z].data_ ) ;

                    // Column major
                    xt::noalias( view(child, q, p, xt::all()) ) +=
                        xt::linalg::dot( view(nli_aux_3d_intrp, q, p, xt::all()),
                                         antrp_mat_sub_[idx_z].data_ );
                }
            }
        }


        template< class field, typename octant_t>
        void nli_antrp_node(octant_t parent, int mask_id)
            {
                auto& parent_linalg_data = parent->data()->template get_linalg_data<field>();

                for (int i = 0; i < parent->num_children(); ++i)
                {

                    auto child = parent->child(i);
                    if (child == nullptr ||
                        !child->locally_owned() ||
                        !child->mask(mask_id)) continue;

                    auto& child_linalg_data  =
                        child ->data()->template get_linalg_data<field>();

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
                    // For Z
                    //xt::noalias(nli_aux_1d_antrp_tmp)=
                    //      view(child, xt::all(), l, q);

                    //xt::noalias( view(nli_aux_2d_antrp, xt::all(), l) ) =
                    //    xt::linalg::dot( antrp_mat_sub_[idx_z].data_,
                    //                        nli_aux_1d_antrp_tmp );

                    // Column major
                    xt::noalias(nli_aux_1d_antrp_tmp)= view(child, q, l, xt::all()) ;
                    xt::noalias( view(nli_aux_2d_antrp, xt::all(), l) ) =
                        xt::linalg::dot( antrp_mat_sub_[idx_z].data_,
                                         nli_aux_1d_antrp_tmp );
                }

                for (int l=0; l<n; ++l)
                {
                    // For Y
                    // nli_aux_3d_antrp(:,l,q) = nli_aux_2d_antrp(l,:) X cmat

                    //xt::noalias( view(nli_aux_3d_antrp, q, l, xt::all()) ) =
                    //    xt::linalg::dot( antrp_mat_sub_[idx_y].data_,
                    //                        view(nli_aux_2d_antrp, l, xt::all()) );

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
                    //nli_aux_1d_antrp_tmp = view(nli_aux_3d_antrp, q,p,xt::all());

                    //xt::noalias( view(parent, p, q, xt::all()) ) +=
                    //    xt::linalg::dot( antrp_mat_sub_[idx_x].data_,
                    //                         nli_aux_1d_antrp_tmp);

                    // Column major
                    xt::noalias( view(parent, xt::all(), q, p) ) +=
                        xt::linalg::dot( antrp_mat_sub_[idx_x].data_,
                                         view(nli_aux_3d_antrp, q,p,xt::all()));
                }
            }
        }



    private:
        template<typename linalg_data_t>
        void antrp_mat_calc_fourier(linalg_data_t& antrp_mat_, int Nb_)
        {
            std::complex<double> II(0,1);

            double M(floor(Nb_));
            //M=16.0;
            xt::xtensor<std::complex<double>, 2>
                mat_cal_basis(std::array<size_t, 2>{{ Nb_, int(M)}});
            xt::xtensor<std::complex<double>, 2>
                mat_cal_intrp(std::array<size_t, 2>{{ 2*Nb_-1, int(M) }});

            std::vector<std::complex<double>> K((int)M);

            // wave numbers


            if ((int(M)% 2) == 1){
                for (int i = 0; i<(M+1)/2; i++) K[i] = double(i);
                for (int i = (M+1)/2; i<M; i++) K[i] = double( i - M );
            } else
            {
                for (int i = 0; i<M/2+1; i++) K[i] = double(i);
                for (int i = M/2+1; i<M; i++) K[i] = double( i - M );
            }

            // basis
            for (int j = 0; j < Nb_; ++j){
                for (int k = 0; k< M; ++k){
                    std::complex<double>  xj = (double)(j)/ (double)Nb_* M/2.0;
                    mat_cal_basis(j,k) = exp( (2.0*M_PI*II/M * xj * K[k])) / M;
                }
            }

            // intrp
            for (int j = 1; j < 2*Nb_-1; ++j){
                for (int k = 0; k< M; ++k){
                    std::complex<double>  xj = (double)(j)/ (double)Nb_* M/4.0;
                    mat_cal_intrp(j,k) = exp( (2.0*M_PI*II/M * xj * K[k])) / M;
                }
            }

            //for (int j = 1; j < 2*Nb_-1; ++j){
            //    for (int k = 0; k< Nb_; ++k){
            //        std::complex<double>  xj = (double)(j+0.5)/ (double)Nb_* M/2.0;
            //        if (k!=Nb_/2)
            //        mat_cal_intrp(j,k) = exp( (2.0*M_PI*II/M/2.0 * xj * K[k])) / M;
            //    }
            //}

            // copy
            auto wt = xt::transpose(xt::conj(mat_cal_basis));

            auto wtw_inv = xt::linalg::pinv( xt::linalg::dot(wt, mat_cal_basis));
            auto tmp = xt::linalg::dot(mat_cal_intrp,
                        xt::linalg::dot(wtw_inv, wt));

            for (int j = 1; j < 2*Nb_-1; ++j){
                for (int k = 0; k< Nb_; ++k){
                    antrp_mat_(k,j-1) = tmp(j,k).real();
                }
            }

            for (int c = Nb_*2 - 1; c>Nb_-1; --c){
                view(antrp_mat_, xt::all(),  c) =
                view(antrp_mat_, xt::all(),  c-2);
            }

        }



        template<typename linalg_data_t>
        void antrp_mat_calc(linalg_data_t& antrp_mat_, int Nb_)
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


    //private:
    public:
        const int pts_cap = 7;

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
    };

}

#endif //IBLGF_INCLUDED_FMM_HPP
