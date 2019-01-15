#ifndef IBLGF_INCLUDED_POISSON_HPP
#define IBLGF_INCLUDED_POISSON_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fftw3.h>

// IBLGF-specific
#include <global.hpp>
#include <simulation.hpp>
#include <domain/domain.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <domain/octree/tree.hpp>
#include <IO/parallel_ostream.hpp>
#include <lgf/lgf.hpp>
#include <fmm/fmm.hpp>
#include <linalg/linalg.hpp>

#include<utilities/convolution.hpp>
#include<utilities/interpolation.hpp>
#include<solver/poisson/poisson.hpp>


const int Dim = 3;

using namespace domain;
using namespace octree;
using namespace types;
using namespace dictionary;
using namespace fft;

struct p_fmm_sin
{

    static constexpr int Dim = 3;

    //              name            type     lBuffer.  hBuffer
    make_field_type(phi_num          , float_type, 1,       1)
    make_field_type(source           , float_type, 1,       1)
    make_field_type(phi_exact        , float_type, 1,       1)
    make_field_type(error            , float_type, 1,       1)
    make_field_type(error_lap_source , float_type, 1,       1)

    // FIXME: temporarily here for FMM
    make_field_type(fmm_s            , float_type, 1,       1)
    make_field_type(fmm_t            , float_type, 1,       1)
    make_field_type(fmm_tmp          , float_type, 1,       1)
    make_field_type(phi_num_fmm      , float_type, 1,       1)


    // temporarily here for amr_laplace test
    make_field_type(amr_lap_source     , float_type, 1,       1)
    make_field_type(amr_lap_tmp        , float_type, 1,       1)

    using datablock_t = DataBlock<
        Dim, node,
        phi_num,
        source,
        phi_exact,
        error,
        fmm_s,
        fmm_t,
        fmm_tmp,
        phi_num_fmm,
        amr_lap_source,
        amr_lap_tmp,
        error_lap_source
        >;

    using coordinate_t       = typename datablock_t::coordinate_type;
    using domain_t           = domain::Domain<Dim,datablock_t>;
    using simulation_type    = Simulation<domain_t>;
    using node_type          = typename datablock_t::node_t;
    using node_field_type    = typename datablock_t::node_field_type;


    p_fmm_sin(Dictionary* _d)
    :simulation_(_d->get_dictionary("simulation_parameters")),
     domain_(simulation_.domain_)
    {
        pcout << "\n Setup:  LGF ViewTest \n" << std::endl;
        pcout << "Simulation: \n" << simulation_    << std::endl;
        this->initialize();
    }


    /** @brief Initialization of poisson problem.
     *  @detail Testing poisson with manufactured solutions: exp
     */
    void initialize()
    {
        this->pcout<<"Initializing"<<std::endl;
        auto bounding_box = domain_.bounding_box();
        auto b_max_0 = bounding_box.max()[0];
        auto b_max_1 = bounding_box.max()[1];
        auto b_max_2 = bounding_box.max()[2];

        std::cout<< bounding_box.min() << bounding_box.max() << std::endl;

        auto center = (domain_.bounding_box().max() -
                       domain_.bounding_box().min()-1) / 2.0 +
                       domain_.bounding_box().min();

        const int nRef = simulation_.dictionary_->
            template get_or<int>("nLevels",0);
        for(int l=0;l<nRef;++l)
        {
            for (auto it  = domain_.begin_leafs();
                    it != domain_.end_leafs(); ++it)
            {
                auto b=it->data()->descriptor();
                coordinate_t lower(2), upper(2);
                b.grow(lower, upper);
                if(b.is_inside( std::pow(2.0,l)*center )
                   && it->refinement_level()==l
                  )
                {
                    domain_.refine(it);
                }
            }
        }

        for (int lt = domain_.tree()->base_level();
                 lt < domain_.tree()->depth(); ++lt)
        {
            for (auto it  = domain_.begin(lt);
                      it != domain_.end(lt); ++it)
            {
                if(it->data())
                {
                    for(auto& e: it->data()->get_data<source>())
                        e=0.0;
                    for(auto& e: it->data()->get_data<phi_num>())
                        e=0.0;
                }
            }
        }

        center+=0.5/std::pow(2,nRef);
        const float_type a  = 10.;
        const float_type a2 = a*a;
        const float_type dx_base = domain_.dx_base();

        xt::xtensor<float_type, 3> soln(std::array<size_t, 3>{{b_max_0, b_max_1, b_max_2 }});
        xt::xtensor<float_type, 3> f_source(std::array<size_t, 3>{{b_max_0, b_max_1, b_max_2 }});

        double wave_number = 2.0;
        double alpha = 0.010;

        for (int i=0; i<b_max_0; ++i)
            for (int j=0; j<b_max_1; ++j)
                for (int k=0; k<b_max_2; ++k)
                {
                    float_type xx = (2* M_PI * wave_number * i) / (b_max_0-1) - M_PI* wave_number;
                    float_type yy = (2* M_PI * wave_number * j) / (b_max_1-1) - M_PI* wave_number;
                    float_type zz = (2* M_PI * wave_number * k) / (b_max_2-1) - M_PI* wave_number;

                    soln(i,j,k) =sin(xx) * sin(yy) * sin(zz) * exp(-alpha * (xx*xx + yy*yy + zz*zz));
                }


        xt::view(soln,0,xt::all(),xt::all()) *= 0;
        xt::view(soln, b_max_0-1, xt::all(),xt::all()) *= 0;
        xt::view(soln, xt::all(), 0, xt::all()) *= 0;
        xt::view(soln, xt::all(), b_max_1-1, xt::all()) *= 0;
        xt::view(soln, xt::all(), xt::all(), 0) *= 0;
        xt::view(soln, xt::all(), xt::all(), b_max_2-1) *= 0;

        std::cout<<f_source << std::endl;

        for (int i=0; i<b_max_0; ++i)
            for (int j=0; j<b_max_1; ++j)
                for (int k=0; k<b_max_2; ++k)
                {
                    f_source(i,j,k) = 0.0;

                    auto ii = i-1; auto jj = j; auto kk = k;
                    if ((ii>=0 && ii <b_max_0) && (jj>=0 && jj <b_max_1) && (kk>=0 && kk <b_max_2 ))
                        f_source(i,j,k) += soln(ii,jj,kk);

                    ii = i+1; jj = j; kk = k;
                    if ((ii>=0 && ii <b_max_0) && (jj>=0 && jj <b_max_1) && (kk>=0 && kk <b_max_2 ))
                        f_source(i,j,k) += soln(ii,jj,kk);

                    ii = i; jj = j-1; kk = k;
                    if ((ii>=0 && ii <b_max_0) && (jj>=0 && jj <b_max_1) && (kk>=0 && kk <b_max_2 ))
                        f_source(i,j,k) += soln(ii,jj,kk);

                    ii = i; jj = j+1; kk = k;
                    if ((ii>=0 && ii <b_max_0) && (jj>=0 && jj <b_max_1) && (kk>=0 && kk <b_max_2 ))
                        f_source(i,j,k) += soln(ii,jj,kk);

                    ii = i; jj = j; kk = k-1;
                    if ((ii>=0 && ii <b_max_0) && (jj>=0 && jj <b_max_1) && (kk>=0 && kk <b_max_2 ))
                        f_source(i,j,k) += soln(ii,jj,kk);

                    ii = i; jj = j; kk = k+1;
                    if ((ii>=0 && ii <b_max_0) && (jj>=0 && jj <b_max_1) && (kk>=0 && kk <b_max_2 ))
                        f_source(i,j,k) += soln(ii,jj,kk);

                    f_source(i,j,k) -= 6.0*soln(i,j,k);
                }


        for (auto it  = domain_.begin_leafs();
                it != domain_.end_leafs(); ++it)
        {

            auto dx_level =  dx_base/std::pow(2,it->refinement_level());
            auto scaling =  std::pow(2,it->refinement_level());


            auto view(it->data()->node_field().domain_view());
            auto& nodes_domain=it->data()->nodes_domain();

            for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
            {
               it2->get<source>() = 0.0;
               it2->get<phi_num>()= 0.0;
               const auto& coord=it2->level_coordinate();

               it2->get<source>() = f_source(coord[0], coord[1], coord[2])/ (dx_level * dx_level);
               it2->get<phi_exact>() = soln(coord[0], coord[1], coord[2]);

            }
        }


        for (auto it  = domain_.begin_leafs();
                it != domain_.end_leafs(); ++it)
        {
            auto a = it->data()->get_linalg_data<phi_exact>();
        }
    }


    /** @brief Run poisson test case, compute errors and write out.  */
    void run()
    {

        solver::PoissonSolver<simulation_type> psolver(&simulation_);
        psolver.solve<source, phi_num, fmm_s, fmm_t, phi_num_fmm, fmm_tmp, amr_lap_source, amr_lap_tmp>();
        compute_errors();
        simulation_.write("solution.vtk");
        pcout << "Writing solution " << std::endl;

    }


    /** @brief Compute L2 and LInf errors */
    void compute_errors()
    {
        auto L2   = 0.; auto LInf = -1.0; int count=0;
        for (auto it_t  = domain_.begin_leafs();
             it_t != domain_.end_leafs(); ++it_t)
        {

            auto& nodes_domain=it_t->data()->nodes_domain();
            for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
            {
                const float_type error_tmp =
                        it2->get<phi_num_fmm>() - it2->get<phi_exact>();

                it2->get<error>() = error_tmp;
                L2 += error_tmp*error_tmp;

                if ( std::fabs(error_tmp) > LInf)
                    LInf = std::fabs(error_tmp);

                ++count;
            }
        }
        pcout << "L2   = " << std::sqrt(L2/count)<< std::endl;
        pcout << "LInf = " << LInf << std::endl;
    }

private:

    Simulation<domain_t>              simulation_;
    domain_t&                         domain_;

    parallel_ostream::ParallelOstream pcout;
    lgf::LGF<lgf::Lookup>             lgf_;
};


#endif // IBLGF_INCLUDED_POISSON_HPP
