#ifndef IBLGF_INCLUDED_IFHERK_SOLVER_HPP
#define IBLGF_INCLUDED_IFHERK_SOLVER_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <array>

// IBLGF-specific
#include <global.hpp>
#include <simulation.hpp>
#include <domain/domain.hpp>
#include <IO/parallel_ostream.hpp>
#include <solver/poisson/poisson.hpp>

namespace solver
{

using namespace domain;

/** @brief Integrating factor 3-stage Runge-Kutta time integration
 * */
template<class Setup>
class Ifherk
{

public: //member types

    using simulation_type      = typename Setup::simulation_t;
    using domain_type          = typename simulation_type::domain_type;
    using datablock_type       = typename domain_type::datablock_t;
    using tree_t               = typename domain_type::tree_t;
    using octant_t             = typename tree_t::octant_type;
    using block_type           = typename datablock_type::block_descriptor_type;
    using real_coordinate_type = typename domain_type::real_coordinate_type;
    using coordinate_type      = typename domain_type::coordinate_type;
    using poisson_solver_t     = typename Setup::poisson_solver_t;


    //Fields
    //using coarse_target_sum = typename Setup::coarse_target_sum;
    //using source_tmp = typename Setup::source_tmp;

    //FMM
    using Fmm_t     = typename Setup::Fmm_t;

    using u   = typename Setup::u;
    using p   = typename Setup::p;
    using q_i = typename Setup::q_i;
    using r_i = typename Setup::r_i;
    using g_i = typename Setup::g_i;
    using d_i = typename Setup::d_i;

    using cell_aux = typename Setup::cell_aux;
    using face_aux = typename Setup::face_aux;
    using face_aux_2 = typename Setup::face_aux_2;
    using w_1      = typename Setup::w_1;
    using w_2      = typename Setup::w_2;
    using u_i      = typename Setup::u_i;

    static constexpr int lBuffer=1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer=1; ///< Lower left buffer for interpolation

    Ifherk(simulation_type* _simulation)
    :
    domain_(_simulation->domain_.get()),
    psolver(_simulation)
    {
        dx_      = domain_->dx_base();
        dt_      = _simulation->dictionary()->template get<float_type>("dt");
        nsteps_  = _simulation->dictionary()->template get<int>("nTimeSteps");
        cfl_max_ = _simulation->dictionary()->template get_or<float_type>("cfl_max",1000);
        Re_      = _simulation->dictionary()->template get<float_type>("Re");

        float_type tmp = Re_*dx_*dx_/dt_;

        alpha_[0]=(c_[1]-c_[0])/tmp;
        alpha_[1]=(c_[2]-c_[1])/tmp;
        alpha_[2]=(c_[3]-c_[2])/tmp;
    }

public:
    void time_march()
    {
        boost::mpi::communicator world;
        parallel_ostream::ParallelOstream pcout=parallel_ostream::ParallelOstream(world.size()-1);

        T_ = 0.0;
        pcout<<"Time marching with dt = " << dt_ << std::endl;
        pcout<<"                   nsteps = " << nsteps_ << std::endl;

        for (int i=0; i<nsteps_; ++i)
        {
            //mDuration_type ifherk_lgf(0);
            //TIME_CODE( ifherk_lgf, SINGLE_ARG(
            //psolver.template apply_lgf<cell_aux, d_i>();
            //));
            //std::cout<<ifherk_lgf.count()<< " on rank = "<< world.rank()<<std::endl;

            mDuration_type ifherk_if(0);
            TIME_CODE( ifherk_if, SINGLE_ARG(
            psolver.template apply_lgf_IF<u, u>(1.0/(Re_*dx_*dx_/dt_));
            ));
            pcout<<ifherk_if.count()<<std::endl;
            //time_step();
            pcout<<"T = " << T_ << " -----------------" << std::endl;
        }

    }

    void time_step()
    {
        // Initialize IFHERK
        // q_1 = u
        boost::mpi::communicator world;
        parallel_ostream::ParallelOstream pcout=parallel_ostream::ParallelOstream(world.size()-1);

         copy<u, q_i>();

         pcout<<"Stage 1"<< std::endl;
         // Stage 1
         // ******************************************************************
         clean<g_i>();
         clean<d_i>();
         clean<cell_aux>();
         clean<face_aux>();
         clean<face_aux_2>();

         // TODO nonlinear g_i
         // nonlin<u,g_i>();

         copy<q_i, r_i>();
         add<g_i, r_i>(coeff_a(1,1)*(-dt_));

         lin_sys_solve(alpha_[0]);

         pcout<<"Stage 2"<< std::endl;
         // Stage 2
         // ******************************************************************
         clean<r_i>();
         clean<d_i>();
         clean<cell_aux>();

         //cal wii
         //r_i = q_i + dt(a21 w21)
         //w11 = (1/a11)* dt (g_i - face_aux)

         add<g_i, face_aux>();
         copy<face_aux, w_1>(1.0/dt_/coeff_a(1,1));

         // TODO
         psolver.template apply_lgf_IF<q_i, q_i>(alpha_[0]);
         psolver.template apply_lgf_IF<w_1, w_1>(alpha_[0]);

         add<q_i, r_i>();
         add<w_1, r_i>(dt_*coeff_a(2,1));

         //nonlin<u_i,g_i>();
         add<g_i, r_i>( coeff_a(2,2)*(-dt_) );

         lin_sys_solve(alpha_[1]);

         pcout<<"Stage 3"<< std::endl;
         // Stage 3
         // ******************************************************************
         clean<d_i>();
         clean<cell_aux>();
         clean<w_2>();

         add<g_i, face_aux>();
         copy<face_aux, w_2>(1.0/dt_/coeff_a(2,2));
         copy<q_i, r_i>();
         add<w_1, r_i>(dt_*coeff_a(3,1));
         add<w_2, r_i>(dt_*coeff_a(3,2));

         psolver.template apply_lgf_IF<r_i, r_i>(alpha_[1]);

         //nonlin<u_i,g_i>();

         add<g_i, r_i>( coeff_a(3,3)*(-dt_) );

         lin_sys_solve(alpha_[2]);

         // ******************************************************************
         copy<u_i, u>();
         copy<d_i, p>(1.0/coeff_a(3,3)/dt_);
         // ******************************************************************
         // Add dt to Time
         T_ += dt_;
    }


private:
    void lin_sys_solve(float_type _alpha)
    {
         // TODO Poisson
         // Projection  with -1? need to check
         // cell_aux = G^T r_i
         // d_i = L^-1 cell_aux
         // note: defined as the negative of the one defined
         // in Liska's paper

         // cell_aux = G^T r_i
         // d_i = L^-1 cell_aux

         psolver.template apply_lgf<cell_aux, d_i>();

         // face_aux = G d_i
         add<face_aux, r_i>();

         if (std::fabs(_alpha)>1e-4)
             psolver.template apply_lgf_IF<r_i, u_i>(_alpha);
         else
             copy<r_i,u_i>();
    }

    float_type coeff_a(int i, int j)
    {return a_[i*(i-1)/2+j-1];}

    template <typename F>
    void clean()
    {
        for (auto it  = domain_->begin_leafs();
                  it != domain_->end_leafs(); ++it)
        {
            if(!it->locally_owned() || !it->data()) continue;
            for (std::size_t field_idx=0; field_idx<F::nFields; ++field_idx)
            {
                for(auto& e: it->data()->template get_data<F>(field_idx))
                    e=0.0;
            }
        }
    }

    template <typename From, typename To>
    void add(float_type scale=1.0)
    {
        static_assert (From::nFields == To::nFields, "number of fields doesn't match when add");
        for (auto it  = domain_->begin_leafs();
                  it != domain_->end_leafs(); ++it)
        {
            if(!it->locally_owned() || !it->data()) continue;
            for (std::size_t field_idx=0; field_idx<From::nFields; ++field_idx)
            {

                it->data()->template get_linalg<To>(field_idx).get()->
                    cube_noalias_view() +=
                     it->data()->template get_linalg_data<From>(field_idx) * scale;

            }
        }
    }

    template <typename From, typename To>
    void copy(float_type scale=1.0)
    {
        static_assert (From::nFields == To::nFields, "number of fields doesn't match when copy");

        for (auto it  = domain_->begin_leafs();
                  it != domain_->end_leafs(); ++it)
        {
            if(!it->locally_owned() || !it->data()) continue;
            for (std::size_t field_idx=0; field_idx<From::nFields; ++field_idx)
            {
                it->data()->template get_linalg<To>(field_idx).get()->
                    cube_noalias_view() =
                     it->data()->template get_linalg_data<From>(field_idx) * scale;

            }

        }
    }



private:
    domain_type*                      domain_;    ///< domain
    poisson_solver_t psolver;

    float_type T_;
    float_type dt_, dx_;
    float_type Re_;
    float_type cfl_max_;
    int nsteps_;
    std::array<float_type, 6> a_{{1.0/3, -1.0, 2.0, 0.0, 0.75, 0.25}};
    std::array<float_type, 4> c_{{0.0, 1.0/3, 1.0, 1.0}};
    std::array<float_type, 3> alpha_{{0.0,0.0,0.0}};

};

}

#endif // IBLGF_INCLUDED_POISSON_HPP
