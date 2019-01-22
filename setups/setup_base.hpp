#ifndef IBLGF_INCLUDED_SETUP_BASE_HPP
#define IBLGF_INCLUDED_SETUP_BASE_HPP

#include <global.hpp>

#include <domain/domain.hpp>
#include <simulation.hpp>
#include <fmm/fmm.hpp>
#include<solver/poisson/poisson.hpp>

using namespace domain;
using namespace octree;
using namespace types;
using namespace fmm;
using namespace dictionary;

/**  @brief Base class for a setup. Provides all the neccessary default fields
 *          and aliases for datablock, domain and simulation.
 */
template<class Setup, class SetupTraits>
class SetupBase : public SetupTraits
{

public: 
    using SetupTraits::Dim;

public: //default fields
    REGISTER_FIELDS
    (Dim,
    (
      (coarse_target_sum, float_type, 1, 1),
      (source_tmp       , float_type, 1, 1),
      (fmm_s,             float_type, 1, 1), 
      (fmm_t,             float_type, 1, 1)
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
    using user_fields   = typename SetupTraits::fields_tuple_t;
    using datablock_t   = datablock_template_t<user_fields>;
    using coordinate_t  = typename datablock_t::coordinate_type;
    using domain_t      = domain::Domain<Dim,datablock_t>;
    using simulation_t  = Simulation<domain_t>;

    using Fmm_t = Fmm<SetupBase>;
    using poisson_solver_t = solver::PoissonSolver<SetupBase>;

public: //Ctors
    SetupBase(Dictionary* _d)
    :simulation_(_d->get_dictionary("simulation_parameters")),
     domain_(simulation_.domain_)
    {
        pcout << "\n Setup:  LGF ViewTest \n" << std::endl;
        pcout << "Simulation: \n" << simulation_    << std::endl;
    }


protected:

    simulation_t                        simulation_; ///< simulation
    domain_t&                           domain_;     ///< Domain reference 
    parallel_ostream::ParallelOstream   pcout;       ///< parallel cout
};


#endif // IBLGF_INCLUDED_POISSON_HPP
