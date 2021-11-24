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

#ifndef IBLGF_INCLUDED_SOLVER_LINSYS_HELM_HPP
#define IBLGF_INCLUDED_SOLVER_LINSYS_HELM_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <complex>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>

#include <cstddef>
#include <iblgf/tensor/vector.hpp>

namespace iblgf
{
namespace solver
{
using namespace domain;

template<class Setup>
class LinSysSolver_helm
{
  public: //member types
    using simulation_type = typename Setup::simulation_t;
    using poisson_solver_t = typename Setup::poisson_solver_t;
    using domain_type = typename simulation_type::domain_type;
    using datablock_type = typename domain_type::datablock_t;
    using tree_t = typename domain_type::tree_t;
    using octant_t = typename tree_t::octant_type;
    using MASK_TYPE = typename octant_t::MASK_TYPE;
    using block_type = typename datablock_type::block_descriptor_type;
    using ib_t = typename domain_type::ib_t;
    using real_coordinate_type = typename domain_type::real_coordinate_type;
    //using force_type = typename ib_t::force_type;
    
    using coordinate_type = typename domain_type::coordinate_type;
    // FMM
    using Fmm_t = typename Setup::Fmm_t;
    using u_type = typename Setup::u_type;
    using r_i_type = typename Setup::r_i_type;
    using cell_aux2_type = typename Setup::cell_aux2_type;
    using face_aux_type = typename Setup::face_aux_type;
    using face_aux2_type = typename Setup::face_aux2_type;

    static constexpr std::size_t N_modes = Setup::N_modes; //number of complex modes


    using point_force_type = types::vector_type<float_type, 3 * 2 * N_modes>;
    using force_type = std::vector<point_force_type>;

  public:
    LinSysSolver_helm(simulation_type* simulation)
    : simulation_(simulation)
    , domain_(simulation->domain_.get())
    , ib_(&domain_->ib())
    , psolver_(simulation, N_modes - 1)
    {
        cg_threshold_ = simulation_->dictionary_->template get_or<float_type>("cg_threshold",1e-3);
        cg_max_itr_ = simulation_->dictionary_->template get_or<int>("cg_max_itr", 40);
        c_z = simulation_->dictionary()->template get_or<float_type>(
            "L_z", 1.0);
        /*float_type dx_base_ = domain_->dx_base();
        const int l_max = domain_->tree()->depth();
        const int l_min = domain_->tree()->base_level();
        const int nLevels = l_max - l_min;
        c_z = dx_base_*N_modes*2/std::pow(2.0, nLevels - 1);*/

        additional_modes = simulation_->dictionary()->template get_or<int>("add_modes", N_modes - 1);
        _compute_Modes.resize(N_modes);
        for (int i = 0; i < N_modes; i++) {
            this->_compute_Modes[i] = true;
        }
    }

    float_type test()
    {
        float_type alpha =  0.01;
        if (domain_->is_server())
            return 0;

        for (std::size_t i=0; i<ib_->size(); ++i)
            ib_->force(i,0)=1;

        //force_type uc(ib_->force());

        //this->template CG_solve<u_type>(uc, alpha);

        //force_type tmp(ib_->size(), (0.,0.,0.));
        //this->template ET_H_S_E<u_type>(ib_->force(), tmp, MASK_TYPE::IB2AMR, alpha);

        //ib_->communicator().compute_indices();
        //ib_->communicator().communicate(true, ib_->force());

        //if (comm_.rank()==1)
        //{
        //    printvec(ib_->force(), "forces");
        //    printvec(tmp, "u");
        //}

        domain::Operator::domainClean<face_aux_type>(domain_);
        domain::Operator::add_field_expression_complex_helmholtz<face_aux_type>(domain_, N_modes, simulation_->frame_vel(), 1.0);

        point_force_type tmp(0.0);

        force_type tmp2(ib_->size(), tmp);
        this->projection<face_aux_type>(tmp2);
        ib_->communicator().compute_indices();
        ib_->communicator().communicate(true, tmp2);


        

        if (comm_.rank()==1)
        {
            std::cout<< " Projection test" << std::endl;
            for (std::size_t i=0; i<ib_->size(); ++i)
                std::cout<<ib_->coordinate(i) << ", " << tmp2[i] << std::endl;
        }

        return 0;
    }

    template<class Field>
    void ib_solve(float_type alpha, float_type t)
    {
        // right hand side
        point_force_type tmp(0.0);
        force_type uc(ib_->size(), tmp);

        domain_->client_communicator().barrier();
        //std::cout<<"projection" << std::endl;
        this->projection<Field>(uc);
        domain_->client_communicator().barrier();
        //std::cout<<"subtract_boundary_vel" << std::endl;
        this->subtract_boundary_vel(uc, t);

        domain_->client_communicator().barrier();
        domain::Operator::domainClean<face_aux2_type>(domain_);
        this->template CG_solve<face_aux2_type>(uc, alpha);
    }

    template<class Field>
    void pressure_correction()
    {
        domain::Operator::domainClean<face_aux2_type>(domain_);
        domain::Operator::domainClean<cell_aux2_type>(domain_);
        this->smearing<face_aux2_type>(ib_->force());

        int l = domain_->tree()->depth()-1;
        domain::Operator::levelDivergence_helmholtz_complex<face_aux2_type, cell_aux2_type>(domain_, l, (1 + additional_modes), c_z);

        // apply L^-1
        psolver_.template apply_lgf_and_helm<cell_aux2_type,cell_aux2_type>(N_modes, 1, MASK_TYPE::IB2AMR);

        domain::Operator::add<cell_aux2_type, Field>(domain_, -1.0);
    }

    template<class ForceType>
    void subtract_boundary_vel(ForceType& uc, float_type t)
    {
        int sep = 2*N_modes;
        auto& frame_vel = simulation_->frame_vel();
        for (std::size_t i=0; i<uc.size(); ++i) {
            uc[i][0]    -=frame_vel(0, t, ib_->coordinate(i));
            uc[i][sep]  -=frame_vel(1, t, ib_->coordinate(i));
            uc[i][2*sep]-=frame_vel(2, t, ib_->coordinate(i));
            
        }
    }

    template<class ForceType>
    ForceType boundaryVel(ForceType x)
    {
        point_force_type tmp(0.0);
        tmp[0] = 1.0;
        return tmp;
    }

    template<class Ftmp, class UcType>
    void CG_solve(UcType& uc, float_type alpha)
    {
        auto& f = ib_->force();

        
        for (int i = 0; i < N_modes; i++) {
            this->_compute_Modes[i] = true;
        }

        point_force_type tmp(0.0);
        force_type Ax(ib_->size(), tmp);
        force_type r (ib_->size(), tmp);
        force_type Ap(ib_->size(), tmp);

        if (domain_->is_server())
            return;

        // Ax
        this->template ET_H_S_E<Ftmp>(f, Ax, alpha);
        //printvec(Ax, "Ax");

        //  res = uc - Ax
        for (int i=0; i<ib_->size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                r[i]=0;
            else
                r[i]=uc[i]-Ax[i];
        }

        // p = res
        auto p = r;

        // rold = r'* r;
        float_type rsold = dot(r, r);

        for (int k=0; k<cg_max_itr_; k++)
        {
            // Ap = A(p)
            this->template ET_H_S_E<Ftmp>(p, Ap, alpha );
            // alpha = rsold / p'*Ap
            float_type pAp = dot(p,Ap);
            if (pAp == 0.0)
            {
                return;
            }

            float_type alpha = rsold / dot(p, Ap);
            // f = f + alpha * p;
            add(f, p, 1.0, alpha);
            // r = r - alpha*Ap
            add(r, Ap, 1.0, -alpha);
            // rsnew = r' * r
            float_type rsnew = dot(r, r);
            float_type f2 = dot(f,f);
            //auto ModeError = dot_Mode(r,r);
            if (comm_.rank()==1)
                std::cout<< "residue square = "<< rsnew/f2<<std::endl;;
            if (sqrt(rsnew/f2)<cg_threshold_)
                break;

            // p = r + (rsnew / rsold) * p;
            add(p, r, rsnew/rsold, 1.0);
            rsold = rsnew;
        }
    }

    template<class Ftmp, class UcType>
    void BCG_solve(UcType& uc, float_type alpha_)
    {
        auto& f = ib_->force();

        
        for (int i = 0; i < N_modes; i++) {
            this->_compute_Modes[i] = true;
        }

        point_force_type tmp(0.0);
        force_type Ax(ib_->size(), tmp);
        force_type r(ib_->size(), tmp);
        //force_type v(ib_->size(), tmp); //sort of like a residue in QMR but not really
        //force_type v_prev(ib_->size(), tmp); //store previous v
        force_type Ap(ib_->size(), tmp);
        force_type Error(ib_->size(), tmp);

        if (domain_->is_server())
            return;

        // Ax
        this->template ET_H_S_E<Ftmp>(f, Ax, alpha_);
        //printvec(Ax, "Ax");

        //  res = uc - Ax
        for (int i=0; i<ib_->size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                r[i]=0;
            else
                r[i]=uc[i]-Ax[i];
        }

        //float_type w = dot(v,v);

        //std::complex<float_type> beta_sq = product(v,v);
        //std::complex<float_type> beta = std::sqrt(beta_sq);

        //std::complex<float_type> tao = beta*w;

        // p = res
        auto p = r;

        // rold = r'* r;
        std::complex<float_type> rsold = product(r, r);

        const std::complex<float_type> OneComp(1.0, 0.0);

        for (int k=0; k<cg_max_itr_; k++)
        {
            // Ap = A(p)
            this->template ET_H_S_E<Ftmp>(p, Ap, alpha_);
            // alpha = rsold / p'*Ap
            std::complex<float_type> pAp = product(p,Ap);
            if (std::abs(pAp) == 0.0 || std::abs(rsold) == 0.0)
            {
                return;
            }

            std::complex<float_type> alpha = rsold / pAp;
            // f = f + alpha * p;
            add_complex(f, p, OneComp, alpha);
            // r = r - alpha*Ap
            add_complex(r, Ap, OneComp, -alpha);
            // rsnew = r' * r
            std::complex<float_type> rsnew = product(r, r);
            float_type f2 = dot(f,f);
            float_type rs_mag = std::abs(rsnew);

            this->template ET_H_S_E<Ftmp>(f, Ax, alpha_);
            for (int i = 0; i < ib_->size(); ++i)
            {
                if (ib_->rank(i) != comm_.rank()) Error[i] = 0;
                else
                    Error[i] = uc[i] - Ax[i];
            }
            float_type errorMag = dot(Error, Error);
            if (comm_.rank()==1)
                std::cout<< "BCG residue square = "<< rs_mag/f2<< " Error is " << errorMag << std::endl;
            if (sqrt(rs_mag/f2)<cg_threshold_)
                break;

            // p = r + (rsnew / rsold) * p;
            add_complex(p, r, rsnew/rsold, OneComp);
            rsold = rsnew;
        }
    }

    template<class Ftmp, class UcType>
    void BCGstab_solve(UcType& uc, float_type alpha_)
    {
        auto& f = ib_->force();

        
        for (int i = 0; i < N_modes; i++) {
            this->_compute_Modes[i] = true;
        }

        point_force_type tmp(0.0);
        force_type Ax(ib_->size(), tmp);
        force_type As(ib_->size(), tmp);
        force_type r(ib_->size(), tmp);
        force_type v(ib_->size(), tmp);
        force_type p(ib_->size(), tmp);
        force_type r_old(ib_->size(), tmp);
        force_type r_hat(ib_->size(), tmp);
        //force_type v(ib_->size(), tmp); //sort of like a residue in QMR but not really
        //force_type v_prev(ib_->size(), tmp); //store previous v
        force_type Ap(ib_->size(), tmp);
        force_type Error(ib_->size(), tmp);

        if (domain_->is_server())
            return;

        // Ax
        this->template ET_H_S_E<Ftmp>(f, Ax, alpha_);
        //printvec(Ax, "Ax");

        //  res = uc - Ax
        for (int i=0; i<ib_->size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                r[i]=0;
            else
                r[i]=uc[i]-Ax[i];
        }

        r_hat = r;

        // rold = r'* r;
        float_type rsold = dot(r, r);
        float_type rho = 1;
        float_type rho_old = rho;
        float_type w = 1;
        float_type alpha = 1;

        for (int k=0; k<cg_max_itr_; k++)
        {
            rho = dot(r_hat, r);
            float_type beta = (rho/rho_old)*(alpha/w);
            add(p, v, 1.0, -w);
            add(p, r, beta, 1.0);
            // Ap = A(p)
            this->template ET_H_S_E<Ftmp>(p, v, alpha_);
            alpha = rho/dot(r_hat, v);

            add(f, p, 1.0, alpha);

            auto s = r;
            add(s, v, 1.0, -alpha);
            this->template ET_H_S_E<Ftmp>(s, As, alpha_);
            w = dot(As, s)/dot(As, As);
            add(f, s, 1.0, w);
            r = s;
            add(r, As, 1.0, -w);

            std::complex<float_type> rsnew = product(r, r);
            float_type f2 = dot(f,f);
            float_type rs_mag = std::abs(rsnew);

            /*this->template ET_H_S_E<Ftmp>(f, Ax, alpha_);
            for (int i = 0; i < ib_->size(); ++i)
            {
                if (ib_->rank(i) != comm_.rank()) Error[i] = 0;
                else
                    Error[i] = uc[i] - Ax[i];
            }
            float_type errorMag = dot(Error, Error);*/
            if (comm_.rank()==1)
                std::cout<< "BCGstab residue square = "<< rs_mag/f2/*<< " Error is " << errorMag*/ << std::endl;
            if (sqrt(rs_mag/f2)<cg_threshold_)
                break;

            // p = r + (rsnew / rsold) * p;
            rho_old = rho;
        }
    }

    template<class Ftmp, class UcType>
    void QMR_solve(UcType& uc, float_type alpha_)
    {
        auto& f = ib_->force();

        const std::complex<float_type> OneComp(1.0, 0.0);

        
        for (int i = 0; i < N_modes; i++) {
            this->_compute_Modes[i] = true;
        }

        point_force_type tmp(0.0);
        force_type Ax(ib_->size(), tmp);
        //force_type r(ib_->size(), tmp);
        force_type v(ib_->size(), tmp); //v_tilde sort of like a residue in QMR but not really
        force_type v_prev(ib_->size(), tmp); //store previous v_tilde
        force_type v_k(ib_->size(), tmp); //actual v
        force_type h(ib_->size(), tmp); //something to keep track of residue
        force_type v_k_old(ib_->size(), tmp);
        force_type Av(ib_->size(), tmp);

        if (domain_->is_server())
            return;

        // Ax
        this->template ET_H_S_E<Ftmp>(f, Ax, alpha_);
        //printvec(Ax, "Ax");

        //  res = uc - Ax
        for (int i=0; i<ib_->size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                v[i]=0;
            else
                v[i]=uc[i]-Ax[i];
        }


        std::complex<float_type> beta_sq = product(v,v);
        std::complex<float_type> beta = std::sqrt(beta_sq);


        auto tmp_v_k = v;
        scale_complex(tmp_v_k, OneComp/beta);
        float_type w = dot(tmp_v_k,tmp_v_k);
        w = std::sqrt(w);
        float_type w_old = w;
        float_type w_new = w;



        h = v;

        std::complex<float_type> tao_tilde = beta*w;
        std::complex<float_type> tao_tilde_old = beta*w;

        std::complex<float_type> s(0.0,0.0);

        std::complex<float_type> s_old = s;
        std::complex<float_type> s_oold = s; //s_{k-2}

        std::complex<float_type> c(1.0,0.0);
        std::complex<float_type> c_old = c;
        std::complex<float_type> c_oold = c;

        // p = res
        auto p = v_k;
        auto p_old = p;
        auto p_oold = p_old;

        std::complex<float_type> s_cum(1.0, 0.0);//value to find the accumulative product of s

        // rold = r'* r;
        

        

        for (int k=0; k<cg_max_itr_; k++)
        {
            if (std::abs(beta) == 0.0) {
                return;
            }
            v_k_old = v_k;
            v_k = v;
            scale_complex(v_k, OneComp/beta);
            //updating w, w_old
            w_old=w;
            w = w_new;
            //compute Av_k
            this->template ET_H_S_E<Ftmp>(v_k, Av, alpha_);
            std::complex<float_type> alpha = product(v_k, Av);
            //v_tilde = Av_k - alpha_k*v_k - beta_k*v_{k-1}
            v = Av;
            add_complex(v, v_k, OneComp, -alpha);
            add_complex(v, v_k_old, OneComp, -beta);
            std::complex<float_type> beta_new_sq = product(v, v);
            std::complex<float_type> beta_new    = std::sqrt(beta_new_sq);
            //updating w
            auto v_tmp = v; //use this v as the v for next step to compute w
            scale_complex(v_tmp, OneComp/beta_new);
            w_new = sqrt(dot(v_tmp, v_tmp));
            //compute theta, eta, xi, c, and s
            std::complex<float_type> theta_k = std::conj(s_oold)*beta*w_old;
            std::complex<float_type> eta     = c_old*c_oold*beta*w_old + std::conj(s_old)*alpha*w;
            std::complex<float_type> xi_tilde= c_old*alpha*w - s_old*c_oold*beta*w_old;
            float_type xi_mag = std::sqrt(std::abs(xi_tilde)*std::abs(xi_tilde)
                                        + w_new * w_new * std::abs(beta_new) * std::abs(beta_new));
            std::complex<float_type> xi(xi_mag, 0.0);
            if (std::abs(xi_tilde) != 0.0) {
                xi = xi_mag * xi_tilde / std::abs(xi_tilde);
            } 
            c = xi_tilde/xi;
            s = beta_new/xi*w_new;
            //updating c_oold c_old s_oold s_old
            c_oold = c_old;
            c_old = c;
            s_oold = s_old;
            s_old = s;

            auto p_tmp = v_k;
            add_complex(p_tmp, p_old, OneComp, -eta);
            add_complex(p_tmp, p_oold, OneComp, -theta_k);
            scale_complex(p_tmp, OneComp/xi);

            //updating p_old and p_oold
            p_oold = p_old;
            p_old = p;
            p = p_tmp;

            //tao_k = c_k tao_tilde_k
            std::complex<float_type> tao = c*tao_tilde;
            tao_tilde = -s*tao_tilde;
            //updating f
            add_complex(f, p, OneComp, tao);
            //updating beta and updating h and s_cum
            s_cum *= s;
            beta = beta_new;
            auto h_tmp = v;
            //get v_{k+1} = v_tilde/beta_new
            scale_complex(h_tmp, OneComp/beta_new);
            auto scaleVal = c*tao_tilde/(std::abs(s_cum)*std::abs(s_cum))*w_new;
            add_complex(h, h_tmp, OneComp, scaleVal);
            float_type f2 = dot(f,f);
            float_type rs_mag = (std::abs(s_cum)*std::abs(s_cum)) * (std::abs(s_cum)*std::abs(s_cum))*dot(h,h);

            if (comm_.rank()==1)
                std::cout<< "residue square = "<< rs_mag/f2<<std::endl;;
            if (sqrt(rs_mag/f2)<cg_threshold_)
                break;
        }
    }


    template <class F, class S>
    void printvec(F& f, S message)
    {
        std::cout<<"-- "<< message << std::endl;
        for (int i=0; i<f.size(); ++i)
            std::cout<<f[i][0] << " " << f[i][2*N_modes] << " " << f[i][4*N_modes] << std::endl;
    }


    //template <class VecType>
    //void ET_S_E(VecType& fin, VecType& fout, int fmm_type = MASK_TYPE::IB2IB)
    //{

    //    force_type ftmp(ib_->size(), (0.,0.,0.));

    //    if (domain_->is_server())
    //        return;

    //    this->smearing<u_type>(fin);
    //    this->projection<u_type>(fout);
    //    this->template apply_Schur<u_type, u_type>(fmm_type);
    //    this->projection<u_type>(ftmp);

    //    add(fout, ftmp, 1, -1);
    //}

    template <class Field, class VecType>
    void ET_H_S_E(VecType& fin, VecType& fout, float_type alpha)
    {
        auto client = domain_->decomposition().client();

        domain::Operator::domainClean<Field>(domain_);
        domain::Operator::domainClean<face_aux_type>(domain_);

        this->smearing<Field>(fin);

        //this->template apply_Schur<Field, face_aux_type>(MASK_TYPE::IB2xIB);

        //domain::Operator::add<face_aux_type, Field>(domain_, -1.0);

        //if (std::fabs(alpha)>1e-4)
        //    psolver_.template apply_lgf_IF<Field, Field>(alpha, MASK_TYPE::xIB2IB);

        if (std::fabs(alpha)>1e-12)
            psolver_.template apply_helm_if<Field, Field>(alpha, N_modes, c_z, 3, MASK_TYPE::IB2xIB);

        this->template apply_Schur<Field, face_aux_type>(MASK_TYPE::xIB2IB);

        domain::Operator::add<face_aux_type, Field>(domain_, -1.0);

        this->projection<Field>(fout);

    }


    template<class Source, class Target>
    void apply_Schur(int fmm_type)
    {

        // finest level only
        int l = domain_->tree()->depth()-1;

        // div
        domain::Operator::levelDivergence_helmholtz_complex<Source, cell_aux2_type>(domain_, l, (1 + additional_modes), c_z);

        // apply L^-1
        psolver_.template apply_lgf_and_helm<cell_aux2_type, cell_aux2_type>(N_modes, 1, fmm_type);

        // apply Gradient
        const int l_max = (fmm_type != MASK_TYPE::STREAM) ?
                    domain_->tree()->depth() : domain_->tree()->base_level()+1;

        const int l_min = (fmm_type !=  MASK_TYPE::IB2xIB && fmm_type !=  MASK_TYPE::xIB2IB) ?
                    domain_->tree()->base_level() : domain_->tree()->depth()-1;

        for (int l = l_min; l < l_max; ++l)
            domain::Operator::levelGradient_helmholtz_complex<cell_aux2_type, Target>(domain_, l, (1 + additional_modes), c_z);
    }

    template<class U, class ForceType,
        typename std::enable_if<(U::mesh_type() == MeshObject::face), void>::type* = nullptr>
    void smearing(ForceType& f, bool cleaning=true)
    {
        int sep = 2*N_modes; //distance between different components in the vector

        ib_->communicator().compute_indices();
        ib_->communicator().communicate(true, f);

        //cleaning
        if (cleaning)
            domain::Operator::domainClean<U>(domain_);

        for (std::size_t i=0; i<ib_->size(); ++i)
        {
            std::size_t oct_i=0;
            for (auto it: ib_->influence_list(i))
            {
                if (!it->locally_owned()) continue;
                auto ib_coord = ib_->scaled_coordinate(i, it->refinement_level());

                domain::Operator::ib_smearing_helmholtz<U>
                    (ib_coord, f[i], ib_->influence_pts(i, oct_i), ib_->delta_func(), N_modes, this->_compute_Modes);
                oct_i+=1;
            }
        }

    }

    template<class U, class ForceType,
        typename std::enable_if<(U::mesh_type() == MeshObject::face), void>::type* = nullptr>
    void projection(ForceType& f)
    {
        int sep = 2*N_modes;

        if (domain_->is_server())
            return;

        // clean f
        for (std::size_t i=0; i<ib_->size(); ++i)
            f[i]=0.0;

        for (std::size_t i=0; i<ib_->size(); ++i)
        {
            std::size_t oct_i=0;

            for (auto it: ib_->influence_list(i))
            {
                if (!it->locally_owned()) continue;
                auto ib_coord = ib_->scaled_coordinate(i, it->refinement_level());

                domain::Operator::ib_projection_helmholtz<U>
                    (ib_coord, f[i], ib_->influence_pts(i, oct_i), ib_->delta_func(), N_modes, this->_compute_Modes);

                oct_i+=1;
            }
        }

        ib_->communicator().compute_indices();
        ib_->communicator().communicate(false, f);

    }

    template<class VecType>
    float_type dot(VecType& a, VecType& b)
    {
        float_type s = 0;
        for (std::size_t i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            for (std::size_t d=0; d<a[0].size(); ++d)
                s+=a[i][d]*b[i][d];
        }

        float_type s_global=0.0;
        boost::mpi::all_reduce(domain_->client_communicator(), s,
                s_global, std::plus<float_type>());
        return s_global;
    }
    /*product of v^Tv for complex vectors*/
    template<class VecType>
    std::complex<float_type> product(VecType& a, VecType& b)
    {
        
        float_type real = 0;
        float_type imag = 0;
        for (std::size_t i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            int Num = a[i].size()/2;
            if (a[i].size() % 2 != 0) throw std::runtime_error("Number of elements in vector is not even (in QMR solve product)");
            for (std::size_t d=0; d<Num; ++d) {
                real+= (a[i][d*2]*b[i][d*2] - a[i][d*2 + 1]*b[i][d*2 + 1]);
                imag+= (a[i][d*2]*b[i][d*2 + 1] + b[i][d*2]*a[i][d*2 + 1]);
            }
        }
        float_type real_global = 0.0;
        float_type imag_global = 0.0;
        
        boost::mpi::all_reduce(domain_->client_communicator(), real,
                real_global, std::plus<float_type>());
        boost::mpi::all_reduce(domain_->client_communicator(), imag,
                imag_global, std::plus<float_type>());
        std::complex<float_type> s_global(real_global,imag_global);
        return s_global;
    }

    template<class VecType>
    std::vector<float_type> dot_Mode(VecType& a, VecType& b)
    {
        std::vector<float_type> s(N_modes, 0.0);
        for (std::size_t i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            for (std::size_t n = 0; n < N_modes; n++) {
                s[n] += (a[i][2*n] * b[i][2*n] + 
                         a[i][2*n + 1] * b[i][2*n + 1] +
                         a[i][2*N_modes + 2*n] * b[i][2*N_modes + 2*n] + 
                         a[i][2*N_modes + 2*n + 1] * b[i][2*N_modes + 2*n + 1] + 
                         a[i][4*N_modes + 2*n] * b[i][4*N_modes + 2*n] + 
                         a[i][4*N_modes + 2*n + 1] * b[i][4*N_modes + 2*n + 1]);
            }
        }

        std::vector<float_type> s_global(N_modes, 0.0);
        boost::mpi::all_reduce(domain_->client_communicator(), s,
                s_global, std::plus<std::vector<float_type>>());
        return s_global;
    }

    template <class VecType>
    void add(VecType& a, VecType& b,
            float_type scale1=1.0, float_type scale2=1.0)
    {
        for (int i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            for (int d=0; d<a[0].size(); ++d)
                a[i][d] = a[i][d]*scale1 + b[i][d]*scale2;
        }
    }

    template <class VecType>
    void add_complex(VecType& a, VecType& b,
            std::complex<float_type> scale1, std::complex<float_type> scale2)
    {
        for (int i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            int Num = a[i].size()/2;
            if (a[i].size() % 2 != 0) throw std::runtime_error("Number of elements in vector is not even (in QMR solve product)");
            VecType a_tmp = a;
            for (std::size_t d=0; d<Num; ++d) {
                float_type real = a[i][d*2];
                float_type imag = a[i][d*2+1];
                a[i][d*2]   = (real*scale1.real() - imag*scale1.imag()
                             + b[i][d*2]*scale2.real() - b[i][d*2 + 1]*scale2.imag());
                a[i][d*2+1] = (imag*scale1.real() + real*scale1.imag() 
                             + b[i][d*2 + 1]*scale2.real() + b[i][d*2]*scale2.imag());
            }
        }
    }


    template <class VecType>
    void scale_complex(VecType& a, std::complex<float_type> scale1)
    {
        for (int i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            int Num = a[i].size()/2;
            if (a[i].size() % 2 != 0) throw std::runtime_error("Number of elements in vector is not even (in QMR solve product)");
            VecType a_tmp = a;
            for (std::size_t d=0; d<Num; ++d) {
                float_type real = a[i][d*2];
                float_type imag = a[i][d*2+1];
                a[i][d*2]   = (real*scale1.real() - imag*scale1.imag());
                a[i][d*2+1] = (imag*scale1.real() + real*scale1.imag());
            }
        }
    }

  private:
    boost::mpi::communicator comm_;
    std::vector<bool> _compute_Modes;
    simulation_type* simulation_;
    domain_type*     domain_; ///< domain
    ib_t* ib_;
    poisson_solver_t psolver_;

    int additional_modes = 0;

    float_type c_z; //length in the homogeneous direction

    float_type cg_threshold_;
    int  cg_max_itr_;
};

} // namespace solver
} // namespace iblgf

#endif
