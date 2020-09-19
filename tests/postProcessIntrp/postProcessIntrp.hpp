#ifndef IBLGF_INCLUDED_POSTPROCESS_INTRP_HPP
#define IBLGF_INCLUDED_POSTPROCESS_INTRP_HPP

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
        //IF-HERK
         (u                , float_type, 3,    1,       1,     face,true ),
         (f_tmp            , float_type, 1,    1,       1,     cell,true ),
         (p                , float_type, 1,    1,       1,     cell,true )
    ))
    // clang-format on
};

struct PostProcessIntrp : public SetupBase<PostProcessIntrp, parameters>
{
    using super_type =SetupBase<PostProcessIntrp,parameters>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

    PostProcessIntrp(   Dictionary* _d,
                        std::string restart_tree_dir,
                        std::string restart_field_dir
                    )
    :super_type(_d,
            [this](auto _d, auto _domain){
                return this->initialize_domain(_d, _domain); },
                restart_tree_dir),
    restart_tree_dir_(restart_tree_dir),
    restart_field_dir_(restart_field_dir)
    {

        if(domain_->is_client())client_comm_=client_comm_.split(1);
        else client_comm_=client_comm_.split(0);

        pcout << "\n PostProcessing : "<<restart_field_dir << std::endl;
        domain_->restart_list_construct();
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        simulation_.template read_h5<u_type>(restart_field_dir,"u");
        simulation_.template read_h5<edge_aux_type>(restart_field_dir,"edge_aux");
    }

    void run( std::string output_name="PostTest")
    {
        boost::mpi::communicator world;

        time_integration_t ifherk(&this->simulation_);
        auto client = domain_->decomposition().client();

        if(domain_->is_client())
        {
            ifherk.template up_and_down<u_type>();
            ifherk.template up_and_down<edge_aux_type>();


            for (int l = domain_->tree()->base_level();
                    l < domain_->tree()->depth(); ++l)
            {
                client->template buffer_exchange<u_type>(l);
                client->template buffer_exchange<edge_aux_type>(l);
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned() || !it->has_data()) continue;

                    domain::Operator::cell_center_average<u_type, f_tmp_type>(it->data());
                    domain::Operator::cell_center_average<edge_aux_type, f_tmp_type>(it->data());
                }
            }

        }
        simulation_.write(output_name);
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

    std::string restart_tree_dir_;
    std::string restart_field_dir_;
};


} // namespace iblgf

#endif
