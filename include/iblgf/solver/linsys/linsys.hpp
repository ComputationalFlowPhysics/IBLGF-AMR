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

#ifndef IBLGF_INCLUDED_SOLVER_LINSYS_HPP
#define IBLGF_INCLUDED_SOLVER_LINSYS_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>

namespace iblgf
{
namespace solver
{
using namespace domain;

template<class Setup>
class LinSysSolver
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
    using force_type = typename ib_t::force_type;
    using coordinate_type = typename domain_type::coordinate_type;
    // FMM
    using Fmm_t = typename Setup::Fmm_t;
    using u_type = typename Setup::u_type;
    using r_i_type = typename Setup::r_i_type;
    using cell_aux2_type = typename Setup::cell_aux2_type;

  public:

    LinSysSolver(simulation_type* simulation)
    :simulation_(simulation),
    domain_(simulation->domain_.get()),
    ib_(domain_->ib_ptr()),
    psolver_(simulation)
    {

    }

    float_type test()
    {
        if (domain_->is_server())
            return 0;

        for (std::size_t i=0; i<ib_->size(); ++i)
            ib_->force(i,0)=1;

        auto uc = ib_->force_copy();

        //CG_solve(uc);

        force_type tmp(ib_->size(), (0.,0.,0.));
        //printvec(ib_->force(), "force before recover");
        ET_S_E(ib_->force(), tmp, MASK_TYPE::AMR2AMR);

        return 0;
    }

    template< class UcType>
    void CG_solve(UcType& uc, int fmm_type = MASK_TYPE::IB2IB)
    {
        auto& f = ib_->force();
        force_type Ax(ib_->size(), (0.,0.,0.));
        force_type r (ib_->size(), (0.,0.,0.));
        force_type Ap(ib_->size(), (0.,0.,0.));

        if (domain_->is_server())
            return;

        int Nitr = 20;
        float_type threshold=1e-5;


        // Ax
        ET_S_E(f, Ax, fmm_type);

        printvec(Ax, "Ax" );
        //  res = uc - Ax
        for (int i=0; i<ib_->size(); ++i)
            r[i]=uc[i]-Ax[i];
        // p = res
        printvec(r, "r" );

        auto p = r;
        // rold = r'* r;
        float_type rsold = dot(r, r);

        for (int k=0; k<Nitr; k++)
        {
            // Ap = A(p)
            ET_S_E(p, Ap, fmm_type );
            // alpha = rsold / p'*Ap
            float_type alpha = rsold / dot(p, Ap);
            // f = f + alpha * p;
            add(f, p, 1.0, alpha);
            // r = r - alpha*Ap
            add(r, Ap, 1.0, -alpha);
            // rsnew = r' * r
            float_type rsnew = dot(r, r);
            std::cout<< "residue square = "<< rsnew<<std::endl;;
            if (sqrt(rsnew)<threshold)
                break;

            // p = r + (rsnew / rsold) * p;
            add(p, r, rsnew/rsold, 1.0);
            rsold = rsnew;
        }
    }

    void recover_potential_flow()
    {
            ET_L_inv_S_E(p, Ap);
    }


    template <class F, class S>
    void printvec(F& f, S message)
    {
        std::cout<<"-- "<< message << std::endl;
        for (int i=0; i<f.size(); ++i)
            std::cout<<f[i]<< std::endl;
    }


    void H_S()
    {
    }

    template <class VecType>
    void ET_S_E(VecType& fin, VecType& fout, int fmm_type = MASK_TYPE::IB2IB)
    {

        force_type ftmp(ib_->size(), (0.,0.,0.));

        if (domain_->is_server())
            return;

        this->smearing<u_type>(fin);
        //this->projection<u_type>(fout);
        //this->template apply_Schur<u_type, u_type>(fmm_type);
        //this->projection<u_type>(ftmp);

        add(fout, ftmp, 1, -1);
    }

    template <class VecType>
    void ET_L_inv_S_E(VecType& fin, VecType& fout, int fmm_type = MASK_TYPE::IB2IB)
    {

        force_type ftmp(ib_->size(), (0.,0.,0.));

        if (domain_->is_server())
            return;

        this->smearing<u_type>(fin);
        psolver_.template apply_lgf<u_type,u_type>(fmm_type);
        this->projection<u_type>(fout);

        //this->smearing<u_type>(fin);
        this->template apply_Schur<u_type, u_type>(fmm_type);
        //psolver_.template apply_lgf<u_type,u_type>(MASK_TYPE::IB2IB);

        this->projection<u_type>(ftmp);
        add(fout, ftmp, 1, -1);
    }

    template<class Source, class Target>
    void apply_Schur(int fmm_type)
    {
        if (domain_->is_server())
            return;

        // finest level only
        int l = domain_->tree()->depth()-1;

        // div
        domain::Operator::levelDivergence<Source, cell_aux2_type>(domain_, l);

        // apply L^-1
        psolver_.template apply_lgf<cell_aux2_type,cell_aux2_type>(fmm_type);

        //// apply Gradient
        domain::Operator::levelGradient<cell_aux2_type, Target>(domain_, l);

    }

    template<class U, class ForceType,
        typename std::enable_if<(U::mesh_type() == MeshObject::face), void>::type* = nullptr>
    void smearing(ForceType& f)
    {
        if (domain_->is_server())
            return;

        //cleaning
        domain::Operator::domainClean<U>(domain_);

        for (std::size_t i=0; i<ib_->size(); ++i)
        {
            auto ib_coord = ib_->coordinate(i);
            std::cout<<ib_coord<<std::endl;
            for (auto it: ib_->influence_list(i))
            {
                if (!it->locally_owned())
                    continue;

                domain::Operator::ib_smearing<U>(ib_coord, f[i], it->data(), ib_->delta_func());
            }
        }

    }

    template<class U, class ForceType,
        typename std::enable_if<(U::mesh_type() == MeshObject::face), void>::type* = nullptr>
    void projection(ForceType& f)
    {
        // (scale * node + add) * ddf(x)

        if (domain_->is_server())
            return;

        // clean f
        for (std::size_t i=0; i<ib_->size(); ++i)
            f[i]=0.0;

        for (std::size_t i=0; i<ib_->size(); ++i)
        {
            auto ib_coord = ib_->coordinate(i);

            for (auto it: ib_->influence_list(i))
            {
                if (!it->locally_owned())
                    continue;

                domain::Operator::ib_projection<u_type>
                    (ib_coord, f[i], it->data(), ib_->delta_func());

            }
        }

    }

    template<class VecType>
    float_type dot(VecType& a, VecType& b)
    {
        float_type s = 0;
        for (int i=0; i<a.size(); ++i)
        {
            for (int d=0; d<a[0].size(); ++d)
                s+=a[i][d]*b[i][d];
        }

        return s;
    }

    template <class VecType>
    void add(VecType& a, VecType& b,
            float_type scale1=1.0, float_type scale2=1.0)
    {
        for (int i=0; i<a.size(); ++i)
        {
            for (int d=0; d<a[0].size(); ++d)
                a[i][d] = a[i][d]*scale1 + b[i][d]*scale2;
        }
    }




  private:
    simulation_type* simulation_;
    domain_type*     domain_; ///< domain
    ib_t* ib_;
    poisson_solver_t psolver_;
};


}
}

#endif
