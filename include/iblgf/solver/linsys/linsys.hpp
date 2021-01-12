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
    using block_type = typename datablock_type::block_descriptor_type;
    using ib_t = typename domain_type::ib_t;
    using ib_points_t = typename ib_t::ib_points_type;
    using real_coordinate_type = typename domain_type::real_coordinate_type;
    using coordinate_type = typename domain_type::coordinate_type;
    // FMM
    using Fmm_t = typename Setup::Fmm_t;
    using u_type = typename Setup::u_type;
    using r_i_type = typename Setup::r_i_type;

  public:

    LinSysSolver(simulation_type* simulation)
    :simulation_(simulation),
    domain_(simulation->domain_.get()),
    ib_(domain_->get_ib_ptr()),
    psolver(simulation)
    {

    }

    float_type test()
    {
        this-> smearing<u_type>();
        return 0;
    }

    template<class U,
        typename std::enable_if<(U::mesh_type() == MeshObject::face), void>::type* = nullptr>
    void smearing()
    {
        if (domain_->is_server())
            return;

        auto ddf = ib_->ddf();

        constexpr auto u = U::tag();
        for (int i=0; i<ib_->ib_tot(); ++i)
        {
            auto ib_coord = ib_->get_ib_coordinate(i);

            std::cout<<ib_->get_ib_infl(i).size() << std::endl;
            for (auto it: ib_->get_ib_infl(i))
            {
                auto& block = it->data();
                for (auto& node : block)
                {
                    auto n_coord = node.level_coordinate();
                    auto dis = n_coord - ib_coord;

                    node(u, 0) = ib_->get_force(i, 0) * ddf(dis+(0, 0.5, 0.5));
                    node(u, 1) = ib_->get_force(i, 1) * ddf(dis+(0.5, 0, 0.5));
                    node(u, 2) = ib_->get_force(i, 2) * ddf(dis+(0.5, 0.5, 0));

                }
            }
        }

    }



  private:
    simulation_type* simulation_;
    domain_type*     domain_; ///< domain
    std::shared_ptr<ib_t> ib_;
    poisson_solver_t psolver;
};


}
}

#endif
