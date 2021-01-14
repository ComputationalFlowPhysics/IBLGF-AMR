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
    using real_coordinate_type = typename domain_type::real_coordinate_type;
    using coordinate_type = typename domain_type::coordinate_type;
    // FMM
    using Fmm_t = typename Setup::Fmm_t;
    using u_type = typename Setup::u_type;
    using r_i_type = typename Setup::r_i_type;

  public:
    LinSysSolver(simulation_type* simulation)
    : simulation_(simulation)
    , domain_(simulation->domain_.get())
    , ib_(domain_->ib_ptr())
    , psolver(simulation)
    {
    }

    float_type test()
    {
        //const bool send_locally_owned=true;
        //ib_->communicate_test(send_locally_owned);
        this->smearing<u_type>();
        //ib_->communicate_test(!send_locally_owned);
        this->projection<u_type>();
        return 0;
    }

    template<class U,
        typename std::enable_if<(U::mesh_type() == MeshObject::face),
            void>::type* = nullptr>
    void smearing()
    {
        if (domain_->is_server()) return;

        auto& ddf = ib_->delta_func();

        constexpr auto u = U::tag();

        float_type sum = 0;
        for (std::size_t i = 0; i < ib_->size(); ++i)
        {
            auto ib_coord = ib_->coordinate(i);

            //std::cout<<ib_->influence_list(i).size() << std::endl;
            for (auto it : ib_->influence_list(i))
            {
                if (!it->locally_owned()) continue;

                auto& block = it->data();
                for (auto& node : block)
                {
                    auto n_coord = node.level_coordinate();
                    auto dist = n_coord - ib_coord;

                    //FIXME: Make it dimension agnostic
                    real_coordinate_type off(0.5);
                    node(u, 0) =
                        ib_->force(i)[0] *
                        ddf(dist + real_coordinate_type({0, 0.5, 0.5}));
                    node(u, 1) =
                        ib_->force(i)[1] *
                        ddf(dist + real_coordinate_type({0.5, 0, 0.5}));
                    node(u, 2) =
                        ib_->force(i)[2] *
                        ddf(dist + real_coordinate_type({0.5, 0.5, 0}));
                    sum += node(u, 0);
                }
            }
        }
        std::cout << " total sum of u0 is " << sum << std::endl;
    }

    template<class U,
        typename std::enable_if<(U::mesh_type() == MeshObject::face),
            void>::type* = nullptr>
    void projection()
    {
        if (domain_->is_server()) return;

        auto& ddf = ib_->delta_func();

        // clean f
        for (std::size_t i = 0; i < ib_->size(); ++i) ib_->force(i) = 0.0;

        constexpr auto u = U::tag();
        for (std::size_t i = 0; i < ib_->size(); ++i)
        {
            auto ib_coord = ib_->coordinate(i);

            //std::cout<<ib_->influence_list(i).size() << std::endl;
            for (auto it : ib_->influence_list(i))
            {
                if (!it->locally_owned()) continue;

                auto& block = it->data();
                for (auto& node : block)
                {
                    auto n_coord = node.level_coordinate();
                    auto dist = n_coord - ib_coord;

                    //FIXME: Make it dimension agnostic
                    ib_->force(i)[0] +=
                        node(u, 0) *
                        ddf(dist + real_coordinate_type({0, 0.5, 0.5}));
                    ib_->force(i)[1] +=
                        node(u, 1) *
                        ddf(dist + real_coordinate_type({0.5, 0, 0.5}));
                    ib_->force(i)[2] +=
                        node(u, 2) *
                        ddf(dist + real_coordinate_type({0.5, 0.5, 0}));
                }
            }
        }
    }

  private:
    simulation_type* simulation_;
    domain_type*     domain_; ///< domain
    ib_t*            ib_;
    poisson_solver_t psolver;
};

} // namespace solver
} // namespace iblgf

#endif
