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

#ifndef DOMAIN_INCLUDED_SERVERCLIENT_TRAITS_HPP
#define DOMAIN_INCLUDED_SERVERCLIENT_TRAITS_HPP

#include <vector>
#include <stdexcept>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/mpi/status.hpp>

#include <iblgf/global.hpp>
#include <iblgf/domain/decomposition/compute_task.hpp>

#include <iblgf/domain/mpi/task_manager.hpp>
#include <iblgf/domain/mpi/query_registry.hpp>
#include <iblgf/domain/mpi/task.hpp>

namespace domain
{
using namespace sr_mpi;

template<class Domain>
struct ServerClientTraits
{
    using domain_t = Domain;
    using key_t = typename domain_t::key_t;

    using key_query_t = Task<tags::key_query, std::vector<key_t>>;
    using rank_query_t = Task<tags::key_query, std::vector<int>>;

    using octant_t = typename domain_t::octant_t;
    using fmm_mask_type = typename octant_t::fmm_mask_type;
    using flag_list_type = typename octant_t::flag_list_type;

    using flag_query_send_t = Task<tags::flags, std::vector<key_t>>;
    using flag_query_recv_t = Task<tags::flags, std::vector<flag_list_type>>;

    using mask_init_query_send_t = Task<tags::mask_init, std::vector<key_t>>;
    using mask_init_query_recv_t =
        Task<tags::mask_init, std::vector<fmm_mask_type>>;

    template<template<class> class BufferPolicy>
    using mask_query_t = Task<tags::mask_query, bool, BufferPolicy>;

    template<template<class> class BufferPolicy>
    using induced_fields_task_t =
        Task<tags::field_query, std::vector<double>, BufferPolicy, octant_t>;

    using acc_induced_fields_task_t = Task<tags::accumulated_field_query,
        std::vector<double>, Inplace, octant_t>;

    using halo_task_t =
        Task<tags::halo, std::vector<double>, Inplace, octant_t>;

    using balance_task =
        Task<tags::balance, std::vector<double>, Inplace, octant_t>;

    using task_manager_t = TaskManager<key_query_t, rank_query_t,
        flag_query_send_t, flag_query_recv_t, mask_init_query_send_t,
        mask_init_query_recv_t, mask_query_t<OrAssignRecv>,
        induced_fields_task_t<InfluenceFieldBuffer>,
        induced_fields_task_t<CopyAssign>, induced_fields_task_t<AddAssignRecv>,
        acc_induced_fields_task_t, halo_task_t, balance_task>;
};

} // namespace domain

#endif
