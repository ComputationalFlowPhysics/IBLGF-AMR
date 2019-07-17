#ifndef IBLGF_INCLUDED_SETUP_BASE_HPP
#define IBLGF_INCLUDED_SETUP_BASE_HPP

#include <global.hpp>

#include <domain/domain.hpp>
#include <utilities/crtp.hpp>
#include <simulation.hpp>
#include <fmm/fmm.hpp>
#include<solver/poisson/poisson.hpp>
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

public:
    using SetupTraits::Dim;

public: //default fields
    REGISTER_FIELDS
    (Dim,
    (
      (coarse_target_sum,  float_type,  1,  1,  cell),
      (source_tmp,         float_type,  1,  1,  cell),
      (fmm_s,              float_type,  1,  1,  cell),
      (fmm_t,              float_type,  1,  1,  cell),
      //flow variables
      (u,  float_type,  1,  1,  face),
      (v,  float_type,  1,  1,  face),
      (w,  float_type,  1,  1,  face),
      (p,  float_type,  1,  1,  cell)
    ))

    using field_tuple=fields_tuple_t;
    using velocity_tuple = std::tuple<u&,v&,w&>;

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


