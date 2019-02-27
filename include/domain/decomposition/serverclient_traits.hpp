#ifndef DOMAIN_INCLUDED_SERVERCLIENT_TRAITS_HPP
#define DOMAIN_INCLUDED_SERVERCLIENT_TRAITS_HPP

#include <vector>
#include <stdexcept>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/mpi/status.hpp>

#include <global.hpp>
#include <domain/decomposition/compute_task.hpp>

#include <domain/mpi/task_manager.hpp>
#include <domain/mpi/query_registry.hpp>
#include <domain/mpi/task.hpp>

namespace domain
{

using namespace sr_mpi;

template<class Domain>
struct ServerClientTraits
{

    using domain_t = Domain;
    using key_t  = typename  domain_t::key_t;

    using key_query_t  = Task<tags::key_query,std::vector<key_t>>;
    using rank_query_t = Task<tags::key_query,std::vector<int>>;

    template< template<class> class BufferPolicy >
    using mask_query_t = Task<tags::field_query,
                                       bool, BufferPolicy>;

    template< template<class> class BufferPolicy >
    using induced_fields_task_t = Task<tags::field_query,
                                       std::vector<double>, BufferPolicy>;

    using task_manager_t = TaskManager<key_query_t,
                                       rank_query_t,
                                       mask_query_t<OrAssignRecv>,
                                       induced_fields_task_t<CopyAssign>,
                                       induced_fields_task_t<AddAssignRecv>
                                       >;
};


}





#endif
