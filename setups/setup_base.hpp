#ifndef IBLGF_INCLUDED_SETUP_BASE_HPP
#define IBLGF_INCLUDED_SETUP_BASE_HPP

#include <global.hpp>

#include <domain/domain.hpp>
#include <utilities/crtp.hpp>
#include <simulation.hpp>
#include <fmm/fmm.hpp>
#include <solver/poisson/poisson.hpp>
#include <solver/time_integration/ifherk.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>

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
      (coarse_target_sum,  float_type,  1,   1,  1,  cell),
      (source_tmp,         float_type,  1,   1,  1,  cell),
      (correction_tmp,     float_type,  1,   1,  1,  cell),
      (target_tmp,         float_type,  1,   1,  1,  cell),
      (fmm_s,              float_type,  1,   1,  1,  cell),
      (fmm_t,              float_type,  1,   1,  1,  cell),
      //flow variables
      (q_i,                float_type,  3,  1,  1,  face),
      (u_i,                float_type,  3,  1,  1,  face),
      (d_i,                float_type,  1,  1,  1,  cell),
      (g_i,                float_type,  3,  1,  1,  face),
      (r_i,                float_type,  3,  1,  1,  face),
      (w_1,                float_type,  3,  1,  1,  face),
      (w_2,                float_type,  3,  1,  1,  face),
      (face_aux,           float_type,  3,  1,  1,  face),
      (face_aux_2,         float_type,  3,  1,  1,  face),
      (cell_aux,           float_type,  1,  1,  1,  cell),
      (edge_aux,           float_type,  1,  1,  1,  edge)
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
    }



protected:
    simulation_t                        simulation_;     ///< simulation
    std::shared_ptr<domain_t>           domain_=nullptr; ///< Domain reference for convience
    parallel_ostream::ParallelOstream   pcout;           ///< parallel cout on master
    parallel_ostream::ParallelOstream   pcout_c=parallel_ostream::ParallelOstream(1);
};

#endif // IBLGF_INCLUDED_SETUP_BASE_HPP


