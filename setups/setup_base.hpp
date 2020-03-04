#ifndef IBLGF_INCLUDED_SETUP_BASE_HPP
#define IBLGF_INCLUDED_SETUP_BASE_HPP

#include <global.hpp>
#include <utilities/crtp.hpp>
#include <domain/domain.hpp>
#include <simulation.hpp>
#include <lgf/lgf.hpp>
#include <fmm/fmm.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <solver/poisson/poisson.hpp>
#include <solver/time_integration/ifherk.hpp>
#include <IO/parallel_ostream.hpp>

using namespace domain;
using namespace octree;
using namespace types;
using namespace fmm;
using namespace dictionary;

/**  @brief Base class for a setup. Provides all the neccessary default fields
 *          and aliases for datablock, domain and simulation.
 */
template<class Setup, class SetupTraits>
class SetupBase : private crtp::Crtps<Setup,SetupBase<Setup,SetupTraits>>,
                  public SetupTraits
{

const int n_ifherk_stage = 3;
const int n_ifherk_ij = 6;

public:
    using SetupTraits::Dim;

public: //default fields
    REGISTER_FIELDS
    (Dim,
    (
      (coarse_target_sum,   float_type,  1,  1,  1,  cell,false),
      (source_tmp,          float_type,  1,  1,  1,  cell,false),
      (correction_tmp,      float_type,  1,  1,  1,  cell,true),
      (corr_lap_tmp,        float_type,  1,  1,  1,  cell,false),
      (source_correction_tmp,float_type, 1,  1,  1,  cell,false),
      (target_tmp,          float_type,  1,  1,  1,  cell,false),
      (fmm_s,               float_type,  1,  1,  1,  cell,false),
      (fmm_t,               float_type,  1,  1,  1,  cell,false),
      //flow variables
      (q_i,                 float_type,  3,  1,  1,  face,false),
      (u_i,                 float_type,  3,  1,  1,  face,false),
      (d_i,                 float_type,  1,  1,  1,  cell,false),
      (g_i,                 float_type,  3,  1,  1,  face,false),
      (r_i,                 float_type,  3,  1,  1,  face,false),
      (w_1,                 float_type,  3,  1,  1,  face,false),
      (w_2,                 float_type,  3,  1,  1,  face,false),
      (cell_aux,            float_type,  1,  1,  1,  cell,true),
      (face_aux,            float_type,  3,  1,  1,  face,false),
      (stream_f,            float_type,  3,  1,  1,  edge,true),
      (correction,          float_type,  1,  1,  1,  cell,false),
      (edge_aux,            float_type,  3,  1,  1,  edge,true)
    ))

    using field_tuple=fields_tuple_t;

public: //datablock
    template<class... DataFieldType>
    using db_template = domain::DataBlock<Dim, node,
                                DataFieldType...>;
    template<class userFields>
    using datablock_template_t =
        typename domain::tuple_utils::make_from_tuple
        <
          db_template,
          typename domain::tuple_utils::concat
              <field_tuple,userFields>::type
        >::type;


public: //Trait types to be used by others
    using user_fields        = typename SetupTraits::fields_tuple_t;
    using datablock_t        = datablock_template_t<user_fields>;
    using block_descriptor_t = typename datablock_t::block_descriptor_type;
    using extent_t           = typename block_descriptor_t::extent_t;
    using coordinate_t       = typename datablock_t::coordinate_type;
    using domain_t           = domain::Domain<Dim,datablock_t>;
    using domaint_init_f     = typename domain_t::template block_initialze_fct<Dictionary*>;
    using simulation_t       = Simulation<domain_t>;
    using fcoord_t           = coordinate_type<float_type, Dim>;


    using Fmm_t = Fmm<SetupBase>;
    using fmm_mask_builder_t = FmmMaskBuilder<domain_t>;
    using poisson_solver_t = solver::PoissonSolver<SetupBase>;
    using time_integration_t = solver::Ifherk<SetupBase>;

public: //Ctors
    SetupBase(Dictionary* _d)
    :simulation_(_d->get_dictionary("simulation_parameters")),
     domain_(simulation_.domain())
    {
        domain_->initialize(simulation_.dictionary()->get_dictionary("domain"));
    }
    SetupBase(Dictionary* _d,domaint_init_f _fct)
    :simulation_(_d->get_dictionary("simulation_parameters")),
     domain_(simulation_.domain())
    {
        domain_->initialize(
            simulation_.dictionary()->get_dictionary("domain").get(),
            _fct
        );

        if(domain_->is_client())client_comm_=client_comm_.split(1);
        else client_comm_=client_comm_.split(0);

        nLevels_=simulation_.dictionary_->
            template get_or<int>("nLevels",0);
        global_refinement_=simulation_.dictionary_->
            template get_or<int>("global_refinement",0);
    }


public: //memebers


    /** @brief Compute L2 and LInf errors */
    template<class Numeric, class Exact, class Error>
    void compute_errors(std::string _output_prefix="", int field_idx = 0)
    {
        const float_type dx_base=domain_->dx_base();
        float_type L2   = 0.; float_type LInf = -1.0; int count=0;
        float_type L2_exact = 0; float_type LInf_exact = -1.0;

        std::vector<float_type> L2_perLevel(nLevels_+1+global_refinement_,0.0);
        std::vector<float_type> L2_exact_perLevel(nLevels_+1+global_refinement_,0.0);
        std::vector<float_type> LInf_perLevel(nLevels_+1+global_refinement_,0.0);
        std::vector<float_type> LInf_exact_perLevel(nLevels_+1+global_refinement_,0.0);

        std::vector<int> counts(nLevels_+1+global_refinement_,0);

        std::ofstream ofs,ofs_global;
        parallel_ostream::ParallelOstream pofs(io::output().dir()+"/"+
            _output_prefix+"level_error.txt",1,ofs);
        parallel_ostream::ParallelOstream pofs_global(io::output().dir()+"/"+
            _output_prefix+"global_error.txt",1,ofs_global);

        if(domain_->is_server())  return;

        for (auto it_t  = domain_->begin_leafs();
                it_t != domain_->end_leafs(); ++it_t)
        {
            if(!it_t->locally_owned() || !it_t->data())continue;
            if(it_t->is_correction()) continue;

            int refinement_level = it_t->refinement_level();
            double dx = dx_base/std::pow(2.0,refinement_level);

            auto& nodes_domain=it_t->data()->nodes_domain();
            for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
            {
                float_type tmp_exact = it2->template get<Exact>(field_idx);
                float_type tmp_num   = it2->template get<Numeric>(field_idx);

                //if(std::isnan(tmp_num))
                //    std::cout<<"this is nan at level = " << it_t->level()<<std::endl;

                float_type error_tmp = tmp_num - tmp_exact;

                it2->template get<Error>(field_idx) = error_tmp;

                L2 += error_tmp*error_tmp * (dx*dx*dx);
                L2_exact += tmp_exact*tmp_exact*(dx*dx*dx);

                L2_perLevel[refinement_level]+=error_tmp*error_tmp* (dx*dx*dx);
                L2_exact_perLevel[refinement_level]+=tmp_exact*tmp_exact*(dx*dx*dx);
                ++counts[refinement_level];

                if ( std::fabs(tmp_exact) > LInf_exact)
                    LInf_exact = std::fabs(tmp_exact);

                if ( std::fabs(error_tmp) > LInf)
                    LInf = std::fabs(error_tmp);

                if ( std::fabs(error_tmp) > LInf_perLevel[refinement_level] )
                    LInf_perLevel[refinement_level]=std::fabs(error_tmp);

                if ( std::fabs(tmp_exact) > LInf_exact_perLevel[refinement_level] )
                    LInf_exact_perLevel[refinement_level]=std::fabs(tmp_exact);

                ++count;
            }
        }

        float_type L2_global(0.0);
        float_type LInf_global(0.0);

        float_type L2_exact_global(0.0);
        float_type LInf_exact_global(0.0);

        boost::mpi::all_reduce(client_comm_,L2, L2_global, std::plus<float_type>());
        boost::mpi::all_reduce(client_comm_,L2_exact, L2_exact_global, std::plus<float_type>());

        boost::mpi::all_reduce(client_comm_,LInf, LInf_global,[&](const auto& v0,
                               const auto& v1){return v0>v1? v0  :v1;} );
        boost::mpi::all_reduce(client_comm_,LInf_exact, LInf_exact_global,[&](const auto& v0,
                               const auto& v1){return v0>v1? v0  :v1;} );

        pcout_c << "Glabal "<<_output_prefix<<"L2_exact = " << std::sqrt(L2_exact_global)<< std::endl;
        pcout_c << "Global "<<_output_prefix<<"LInf_exact = " << LInf_exact_global << std::endl;

        pcout_c << "Glabal "<<_output_prefix<<"L2 = " << std::sqrt(L2_global)<< std::endl;
        pcout_c << "Global "<<_output_prefix<<"LInf = " << LInf_global << std::endl;

        ofs_global<< std::sqrt(L2_exact_global)<<" "<<LInf_exact_global<<" "
                  << std::sqrt(L2_global) << " " <<LInf_global<<std::endl;

        //Level wise errros
        std::vector<float_type> L2_perLevel_global(nLevels_+1+global_refinement_,0.0);
        std::vector<float_type> LInf_perLevel_global(nLevels_+1+global_refinement_,0.0);

        std::vector<float_type> L2_exact_perLevel_global(nLevels_+1+global_refinement_,0.0);
        std::vector<float_type> LInf_exact_perLevel_global(nLevels_+1+global_refinement_,0.0);


        //files

        std::vector<int> counts_global(nLevels_+1+global_refinement_,0);
        for(std::size_t i=0;i<LInf_perLevel_global.size();++i)
        {
            boost::mpi::all_reduce(client_comm_,counts[i],
                                   counts_global[i], std::plus<float_type>());
            boost::mpi::all_reduce(client_comm_,L2_perLevel[i],
                                   L2_perLevel_global[i], std::plus<float_type>());
            boost::mpi::all_reduce(client_comm_,LInf_perLevel[i],
                                   LInf_perLevel_global[i],[&](const auto& v0,
                                       const auto& v1){return v0>v1? v0  :v1;});

            boost::mpi::all_reduce(client_comm_,L2_exact_perLevel[i],
                                   L2_exact_perLevel_global[i], std::plus<float_type>());
            boost::mpi::all_reduce(client_comm_,LInf_exact_perLevel[i],
                                   LInf_exact_perLevel_global[i],[&](const auto& v0,
                                       const auto& v1){return v0>v1? v0  :v1;});

            pcout_c<<_output_prefix<<"L2_"<<i<<" "<<std::sqrt(L2_perLevel_global[i])<<std::endl;
            pcout_c<<_output_prefix<<"LInf_"<<i<<" "<<LInf_perLevel_global[i]<<std::endl;
            pcout_c<<"count_"<<i<<" "<<counts_global[i]<<std::endl;


            pofs<<i<<" "<<std::sqrt(L2_perLevel_global[i])<<" "<<LInf_perLevel_global[i]<<std::endl;
        }
    }


protected:
    simulation_t                        simulation_;     ///< simulation
    std::shared_ptr<domain_t>           domain_=nullptr; ///< Domain reference for convience
    boost::mpi::communicator            client_comm_;    ///< Communicator for clients only
    boost::mpi::communicator            world_;          ///< World Communicator
    parallel_ostream::ParallelOstream   pcout;           ///< parallel cout on master
    parallel_ostream::ParallelOstream   pcout_c=parallel_ostream::ParallelOstream(1);

    int nLevels_=0;
    int global_refinement_;
};

#endif // IBLGF_INCLUDED_SETUP_BASE_HPP


