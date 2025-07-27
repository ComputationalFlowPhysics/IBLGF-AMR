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

#ifndef IBLGF_INCLUDED_POD2D_HPP
#define IBLGF_INCLUDED_POD2D_HPP

#include <iostream>
#include <iblgf/dictionary/dictionary.hpp>

#include "../../setups/setup_base.hpp"
#include <iblgf/operators/operators.hpp>
#include <iblgf/solver/modal_analysis/pod_petsc.hpp>
#include <slepceps.h>
#include <slepcsys.h>
namespace iblgf
{
using namespace domain;
using namespace types;
using namespace dictionary;

const int Dim = 2;

struct parameters
{
    static constexpr std::size_t Dim = 2;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
        (
            //name, type, nFields, l/h-buf,mesh_obj, output(optional)
            (tlevel,        float_type, 1, 1, 1, cell, true),
            (u,             float_type, 2, 1, 1, face, true),
            (p,             float_type, 1, 1, 1, cell, true),
            (test,          float_type, 1, 1, 1, cell,true ),
            (idx_u,         float_type, 2, 1, 1, face, true)

        )
    )
    // clang-format on
};
struct POD2D : public SetupBase<POD2D, parameters>
{
    using super_type = SetupBase<POD2D, parameters>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;
    using key_id_t = typename domain_t::tree_t::key_type::value_type;

    POD2D(Dictionary* _d, std::string restart_tree_dir, std::string restart_field_dir)
    : super_type(
          _d, [this](auto _d, auto _domain) { return this->initialize_domain(_d, _domain); }, restart_tree_dir)
    , restart_tree_dir_(restart_tree_dir)
    , restart_field_dir_(restart_field_dir)
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);
        nLevelRefinement_ = simulation_.dictionary_->template get_or<int>("nLevels", 0);
        // std::cout << "Number of refinement levels: " << nLevelRefinement_ << std::endl;
        // std::cout << "Restarting list construction..." << std::endl;
        // domain_->register_adapt_condition() =
        //     [this](std::vector<float_type> source_max, auto& octs, std::vector<int>& level_change)
        // { return this->template adapt_level_change(source_max, octs, level_change); };

        domain_->register_refinement_condition() = [this](auto octant, int diff_level) { return false; };
        // domain_->init_refine(nLevelRefinement_, 0, 0);

        domain_->restart_list_construct();
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        simulation_.template read_h5<u_type>(restart_field_dir, "u");

        // simulation_.read(restart_tree_dir,"tree");
        // simulation_.read(restart_field_dir,"fields");
    }

    float_type run(int argc, char* argv[])
    {
        boost::mpi::communicator world;
        PetscCall(SlepcInitialize(&argc, &argv, (char*)0, NULL));
        simulation_.write("init");
        solver::POD<SetupBase> pod(&this->simulation_);
        pod.run_vec_test();
        pod.run_POD();
        PetscCall(SlepcFinalize());
        simulation_.write("final");

        return 0.0;

    }
    
    void initialize()
    {
        boost::mpi::communicator world;
        if (domain_->is_server()) return;
        auto center =
            (domain_->bounding_box().max() - domain_->bounding_box().min() + 1) / 2.0 + domain_->bounding_box().min();

        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned()) continue;
            int ref_level_ = it->refinement_level();
            for (auto& n : it->data())
            {
                n(tlevel) = ref_level_ + 0.5;
                n(u, 0) = ref_level_ + 0.5;
                n(u, 1) = ref_level_ + 0.5;
            }
        }
    }

    std::vector<coordinate_t> initialize_domain(Dictionary* _d, domain_t* _domain)
    {
        auto res = _domain->construct_basemesh_blocks(_d, _domain->block_extent());
        domain_->read_parameters(_d);

        return res;
    }

  private:
    int                      nLevelRefinement_ = 0; // Number of refinement levels
    boost::mpi::communicator client_comm_;
    std::vector<key_id_t>    ref_keys_;  //referfecne keys local to rank
    std::vector<int>         ref_leafs_; //reference leafs local to rank
    std::string              restart_tree_dir_;
    std::string              restart_field_dir_;
};
} // namespace iblgf
#endif