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
    using face_aux_type = typename Setup::face_aux_type;
    using face_aux2_type = typename Setup::face_aux2_type;

  public:
    LinSysSolver(simulation_type* simulation)
    : simulation_(simulation)
    , domain_(simulation->domain_.get())
    , ib_(&domain_->ib())
    , psolver_(simulation)
    {
        cg_threshold_ = simulation_->dictionary_->template get_or<float_type>("cg_threshold",1e-3);
        cg_max_itr_ = simulation_->dictionary_->template get_or<int>("cg_max_itr", 40);
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
        domain::Operator::add_field_expression<face_aux_type>(domain_, simulation_->frame_vel(), 1.0);

	real_coordinate_type tmp_coord(0.0);

        force_type tmp2(ib_->size(), tmp_coord);
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
	real_coordinate_type tmp_coord(0.0);
        force_type uc(ib_->size(), tmp_coord);

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
        domain::Operator::levelDivergence<face_aux2_type, cell_aux2_type>(domain_, l);

        // apply L^-1
        psolver_.template apply_lgf<cell_aux2_type,cell_aux2_type>(MASK_TYPE::IB2AMR);

        domain::Operator::add<cell_aux2_type, Field>(domain_, -1.0);
    }

    template<class ForceType>
    void subtract_boundary_vel(ForceType& uc, float_type t)
    {
        auto& frame_vel = simulation_->frame_vel();
        for (std::size_t i=0; i<uc.size(); ++i)
            for (std::size_t idx=0; idx<uc[i].size(); ++idx)
                uc[i][idx]-=frame_vel(idx, t, ib_->coordinate(i));
    }

    template<class ForceType>
    ForceType boundaryVel(ForceType x)
    {
	if (domain_type::dims == 3) {
        return ForceType({1, 0, 0});
	}
	else {
	return ForceType({1, 0});
	}
    }

    template<class Ftmp, class UcType>
    void CG_solve(UcType& uc, float_type alpha)
    {
        auto& f = ib_->force();

	real_coordinate_type tmp_coord(0.0);
        force_type Ax(ib_->size(), tmp_coord);
        force_type r (ib_->size(), tmp_coord);
        force_type Ap(ib_->size(), tmp_coord);

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
            if (comm_.rank()==1)
                if (domain_type::dims == 3 || (domain_type::dims == 2 && (k % 20 == 0)))std::cout<< "residue square = "<< rsnew/f2<<std::endl;;
            if (sqrt(rsnew/f2)<cg_threshold_)
                break;

            // p = r + (rsnew / rsold) * p;
            add(p, r, rsnew/rsold, 1.0);
            rsold = rsnew;
        }
    }


    template <class F, class S>
    void printvec(F& f, S message)
    {
        std::cout<<"-- "<< message << std::endl;
        for (int i=0; i<f.size(); ++i)
            std::cout<<f[i]<< std::endl;
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

        if (std::fabs(alpha)>1e-4)
            psolver_.template apply_lgf_IF<Field, Field>(alpha, MASK_TYPE::IB2xIB);

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
        domain::Operator::levelDivergence<Source, cell_aux2_type>(domain_, l);

        // apply L^-1
        psolver_.template apply_lgf<cell_aux2_type, cell_aux2_type>(fmm_type);

        // apply Gradient
        const int l_max = (fmm_type != MASK_TYPE::STREAM) ?
                    domain_->tree()->depth() : domain_->tree()->base_level()+1;

        const int l_min = (fmm_type !=  MASK_TYPE::IB2xIB && fmm_type !=  MASK_TYPE::xIB2IB) ?
                    domain_->tree()->base_level() : domain_->tree()->depth()-1;

        for (int l = l_min; l < l_max; ++l)
            domain::Operator::levelGradient<cell_aux2_type, Target>(domain_, l);
    }

    template<class U, class ForceType,
        typename std::enable_if<(U::mesh_type() == MeshObject::face), void>::type* = nullptr>
    void smearing(ForceType& f, bool cleaning=true)
    {

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

                domain::Operator::ib_smearing<U>
                    (ib_coord, f[i], ib_->influence_pts(i, oct_i), ib_->delta_func());
                oct_i+=1;
            }
        }

    }


    template<class U, class ForceType,
        typename std::enable_if<(U::mesh_type() == MeshObject::cell), void>::type* = nullptr>
    void smearing(ForceType& f, bool cleaning=true)
    {

        real_coordinate_type tmp_coord(0.0);
        force_type tmp_f(ib_->size(), tmp_coord);
        for (std::size_t i=0; i<tmp_f.size(); ++i)
        {
            tmp_f[i][0]=f[i];
        }
        //needed for stability solver
        ib_->communicator().compute_indices();
        ib_->communicator().communicate(true, tmp_f);

        for (std::size_t i=0; i<tmp_f.size(); ++i)
        {
            f[i] = tmp_f[i][0];
        }

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

                domain::Operator::ib_smearing<U>
                    (ib_coord, f[i], ib_->influence_pts(i, oct_i), ib_->delta_func());
                oct_i+=1;
            }
        }

    }

    template<class U, class ForceType,
        typename std::enable_if<(U::mesh_type() == MeshObject::face), void>::type* = nullptr>
    void projection(ForceType& f)
    {

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

                domain::Operator::ib_projection<U>
                    (ib_coord, f[i], ib_->influence_pts(i, oct_i), ib_->delta_func());

                oct_i+=1;
            }
        }

        ib_->communicator().compute_indices();
        ib_->communicator().communicate(false, f);

    }


    template<class U, class ForceType,
        typename std::enable_if<(U::mesh_type() == MeshObject::cell), void>::type* = nullptr>
    void projection(ForceType& f)
    {
        //needed for stability solver
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

                domain::Operator::ib_projection<U>
                    (ib_coord, f[i], ib_->influence_pts(i, oct_i), ib_->delta_func());

                oct_i+=1;
            }
        }

        real_coordinate_type tmp_coord(0.0);


        force_type tmp_f(ib_->size(), tmp_coord);
        for (std::size_t i=0; i<tmp_f.size(); ++i)
        {
            tmp_f[i][0]=f[i];
        }
        //needed for stability solver
        ib_->communicator().compute_indices();
        ib_->communicator().communicate(false, tmp_f);

        for (std::size_t i=0; i<tmp_f.size(); ++i)
        {
            f[i] = tmp_f[i][0];
        }

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

  private:
    boost::mpi::communicator comm_;
    simulation_type* simulation_;
    domain_type*     domain_; ///< domain
    ib_t* ib_;
    poisson_solver_t psolver_;

    float_type cg_threshold_;
    int  cg_max_itr_;
};

} // namespace solver
} // namespace iblgf

#endif
