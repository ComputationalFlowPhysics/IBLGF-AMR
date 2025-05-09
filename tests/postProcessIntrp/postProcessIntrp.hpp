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
         (p                , float_type, 1,    1,       1,     cell,true ),
          (test                , float_type, 1,    1,       1,     cell,true )
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

    void run(int NoStep)
    {
        std::string output_name="postProc_"+std::to_string(NoStep);
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

        // Calculating statistics
        std::vector<std::vector<float_type>> stats(5);
        // hydrodynamic impulse I = 0.5 \int x cross omega
        stats[0].resize(3);
        // kinetic energy
        stats[1].resize(1);
        // enstrophy
        stats[2].resize(1);
        // Saffman-centroid
        stats[3].resize(3);
        //tmp
        stats[4].resize(1);

        for (auto& s:stats)
            for (auto& element:s)
                element=0;

        if(domain_->is_client())
        {
            const float_type dx_base = domain_->dx_base();
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves(); ++it)
            {
                if(!it->locally_owned()) continue;

                auto dx_level =  dx_base/std::pow(2,it->refinement_level());
                auto dx3 = dx_level*dx_level*dx_level;

                for (auto& node : it->data())
                {

                    const auto& coord = node.level_coordinate();

                    // cell center coordinates
                    float_type x0 = static_cast<float_type>
                        (coord[0]+0.5)*dx_level;
                    float_type x1 = static_cast<float_type>
                        (coord[1]+0.5)*dx_level;
                    float_type x2 = static_cast<float_type>
                        (coord[2]+0.5)*dx_level;

                    // 1 hydrodynamic impulse I = 0.5 \int x cross omega
                    std::vector<float_type> xXw(3);

                    xXw[0] = x1*node(edge_aux,2)-x2*node(edge_aux,1);
                    xXw[1] = x2*node(edge_aux,0)-x0*node(edge_aux,2);
                    xXw[2] = x0*node(edge_aux,1)-x1*node(edge_aux,0);
                    stats[0][0] += 0.5*xXw[0]*dx3;
                    stats[0][1] += 0.5*xXw[1]*dx3;
                    stats[0][2] += 0.5*xXw[2]*dx3;
                    // 2 kinetic energy
                    //stats[1][0] += 0.5*(node(u,0)*node(u,0) + node(u,1)*node(u,1) + node(u,2)*node(u,2)) * dx3;
                    stats[1][0] += (node(u,0)*xXw[0] + node(u,1)*xXw[1] + node(u,2)*xXw[2]) * dx3;

                    // 3 enstrophy
                    stats[2][0] += 0.5*(node(edge_aux,0)*node(edge_aux,0) + node(edge_aux,1)*node(edge_aux,1) + node(edge_aux,2)*node(edge_aux,2)) * dx3;

                    // 4 Saffman-centroid
                    //float tmp0 = 0.5*(x1*node(edge_aux,2)-x2*node(edge_aux,1)) * dx3;
                    //float tmp1 = 0.5*(x2*node(edge_aux,0)-x0*node(edge_aux,2)) * dx3;
                    //float tmp2 = 0.5*(x2*node(edge_aux,0)-x0*node(edge_aux,2)) * dx3;
                }
            }
        }

        for (auto& s:stats)
            for (auto& element:s)
            {
                float tmp_s=0;
                float tmp = element;
                boost::mpi::all_reduce(world, tmp, tmp_s, std::plus<float_type>());
                world.barrier();
                element=tmp_s;
            }

        // 4 Saffman-centroid / cause it uses I
        if(domain_->is_client())
        {
            const float_type dx_base = domain_->dx_base();
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves(); ++it)
            {
                if(!it->locally_owned()) continue;

                auto dx_level =  dx_base/std::pow(2,it->refinement_level());
                auto dx3 = dx_level*dx_level*dx_level;

                for (auto& node : it->data())
                {

                    const auto& coord = node.level_coordinate();

                    // cell center coordinates
                    float_type x0 = static_cast<float_type>
                        (coord[0]+0.5)*dx_level;
                    float_type x1 = static_cast<float_type>
                        (coord[1]+0.5)*dx_level;
                    float_type x2 = static_cast<float_type>
                        (coord[2]+0.5)*dx_level;

                    std::vector<float_type> xXw(3);

                    xXw[0] = x1*node(edge_aux,2)-x2*node(edge_aux,1);
                    xXw[1] = x2*node(edge_aux,0)-x0*node(edge_aux,2);
                    xXw[2] = x0*node(edge_aux,1)-x1*node(edge_aux,0);

                    // 4 Saffman-centroid
                    float I2 = stats[0][0]*stats[0][0] + stats[0][1]*stats[0][1] + stats[0][2]*stats[0][2];
                    float tmp = (xXw[0]*stats[0][0]+xXw[1]*stats[0][1]+xXw[2]*stats[0][2])/I2;
                    stats[3][0]+=0.5*tmp*x0*dx3;
                    stats[3][1]+=0.5*tmp*x1*dx3;
                    stats[3][2]+=0.5*tmp*x2*dx3;
                }
            }
        }

        for (auto& element:stats[3])
        {
            float tmp_s=0;
            float tmp = element;
            boost::mpi::all_reduce(world, tmp, tmp_s, std::plus<float_type>());
            world.barrier();
            element=tmp_s;
        }

        /// output

        if(!domain_->is_client())
        {
            std::vector<int> leafC(domain_->tree()->depth()-domain_->tree()->base_level());
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves(); ++it)
            {
                leafC[it->refinement_level()]+=1;
            }
            std::cout<<"LeafCount= ";
            for (auto c: leafC)
                std::cout<<c<<" ";
            std::cout<<std::endl;
        }

        if(!domain_->is_client())
        {
            std::ofstream outfile;
            int width=20;

            outfile.open("stats.txt", std::ios_base::app); // append instead of overwrite
            outfile <<std::setw(width) << NoStep<<std::setw(width)<<std::scientific<<std::setprecision(9);
            for (auto& s:stats)
            {
                for (auto& element:s)
                {
                    outfile<<element<<std::setw(width);
                }
            }
            outfile<<std::endl;
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

    std::string restart_tree_dir_;
    std::string restart_field_dir_;
};


} // namespace iblgf

#endif
