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

#ifndef IBLGF_INCLUDED_IBSOLVE_HPP
#define IBLGF_INCLUDED_IBSOLVE_HPP

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <vector>
#include <math.h>
#include <chrono>
#include <fftw3.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/domain/dataFields/dataBlock.hpp>
#include <iblgf/domain/dataFields/datafield.hpp>
#include <iblgf/domain/octree/tree.hpp>
#include <iblgf/IO/parallel_ostream.hpp>
#include <iblgf/lgf/lgf.hpp>
#include <iblgf/fmm/fmm.hpp>

#include <iblgf/utilities/convolution.hpp>
#include <iblgf/interpolation/interpolation.hpp>
#include <iblgf/solver/poisson/poisson.hpp>
#include <iblgf/solver/time_integration/ifherk.hpp>

#include "../../setups/setup_base.hpp"
#include <iblgf/operators/operators.hpp>

namespace iblgf
{
const int Dim = 3;

struct parameters
{
    static constexpr std::size_t Dim = 3;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
     (
        //name               type        Dim   lBuffer  hBuffer, storage type
         (u                , float_type, 3,    1,       1,     face,true ),
         (p                , float_type, 1,    1,       1,     cell,true )
    ))
    // clang-format on
};

struct ibsolve : public SetupBase<ibsolve, parameters>
{
    using super_type = SetupBase<ibsolve, parameters>;
    using real_coordinate_type = typename types::vector_type<float_type, Dim>;

    ibsolve(Dictionary* _d)
    : super_type(_d, [this](auto _d, auto _domain) {
        return this->initialize_domain(_d, _domain);
    })
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);

        Re_ = 1.0;
        domain_->init_refine(_d->get_dictionary("simulation_parameters")
                                 ->template get_or<int>("nLevels", 0),
            global_refinement_);

        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();

        this->initialize();

        boost::mpi::communicator world;
        if (world.rank() == 0)
            std::cout << "on Simulation: \n" << simulation_ << std::endl;
    }

    float_type run()
    {
        linsys_solver_t ibsolve(&this->simulation_);
        ibsolve.test();

        if (!communication_locally_owned_test())
        {
            std::cout << "FAIL: communication_locally_owned_test()"
                      << std::endl;
        }
        if (!communication_ghost_test())
        { std::cout << "FAIL: communication_ghost_test()" << std::endl; }

        simulation_.write("ibtest.hdf5");
        return 0;
    }

    /** @brief Test of send of locally_owned forces and assign in ghosts */
    bool communication_locally_owned_test()
    {
        bool pass = true;

        auto ib_ptr = domain_->ib_ptr();

        boost::mpi::communicator world;
        const auto               my_rank = world.rank();
        for (std::size_t i = 0; i < ib_ptr->size(); ++i)
        {
            if (ib_ptr->rank(i) == world.rank())
            {
                ib_ptr->force(i) =
                    real_coordinate_type(float_type(world.rank()));
            }
        }

        bool send_locally_owned = true;
        ib_ptr->communicator().compute_indices();
        ib_ptr->communicator().communicate(send_locally_owned);

        for (std::size_t i = 0; i < ib_ptr->size(); ++i)
        {
            if (ib_ptr->rank(i) != my_rank)
            {
                for (auto& infl_block : ib_ptr->influence_list(i))
                {
                    if (infl_block->rank() == my_rank)
                    {
                        if (std::fabs(ib_ptr->rank(i) - ib_ptr->force(i).x()) >
                            1e-8)
                        { pass = false; }
                    }
                }
            }
        }

        return pass;
    }

    /** @brief Test of send of ghost and add in locally owned */
    bool communication_ghost_test()
    {
        bool pass = true;

        auto ib_ptr = domain_->ib_ptr();

        boost::mpi::communicator world;
        const auto               my_rank = world.rank();
        for (std::size_t i = 0; i < ib_ptr->size(); ++i)
        {
            if (ib_ptr->rank(i) != world.rank())
            {
                for (auto& infl_block : ib_ptr->influence_list(i))
                {
                    if (infl_block->rank() == my_rank)
                    {
                        ib_ptr->force(i) = real_coordinate_type(
                            float_type(infl_block->rank()));
                    }
                }
            }
            else
            {
                ib_ptr->force(i) = real_coordinate_type(float_type(0.0));
            }
        }

        bool send_locally_owned = false;
        ib_ptr->communicator().compute_indices();
        ib_ptr->communicator().communicate(send_locally_owned);

        for (std::size_t i = 0; i < ib_ptr->size(); ++i)
        {
            if (ib_ptr->rank(i) == my_rank)
            {
                float_type    acc_force = 0.;
                std::set<int> unique_inflRanks;
                for (auto& infl_block : ib_ptr->influence_list(i))
                {
                    //unique ranks
                    if (infl_block->rank() != my_rank)
                    { unique_inflRanks.insert(infl_block->rank()); }
                }

                for (auto& ur : unique_inflRanks) { acc_force += ur; }

                if (std::fabs(acc_force - ib_ptr->force(i).x()) > 1e-8)
                { pass = false; }
            }
        }
        return pass;
    }

    void initialize()
    {
        if (domain_->is_server()) return;

        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned()) continue;

            for (auto& node : it->data())
            {
                node(u, 0) = 0;
                node(u, 1) = 0;
                node(u, 2) = 0;
            }
        }
    }

    /** @brief  Initialization of the domain blocks. This is registered in the
     *          domain through the base setup class, passing it to the domain ctor.
     */
    std::vector<coordinate_t> initialize_domain(
        Dictionary* _d, domain_t* _domain)
    {
        auto res =
            _domain->construct_basemesh_blocks(_d, _domain->block_extent());

        domain_->read_parameters(_d);

        return res;
    }

  private:
    boost::mpi::communicator client_comm_;
    int                      global_refinement_ = 0;
    float_type               Re_;
};

} // namespace iblgf

#endif
