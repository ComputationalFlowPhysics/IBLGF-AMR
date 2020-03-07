#ifndef INCLUDED_TAGS_HPP
#define INCLUDED_TAGS_HPP


namespace sr_mpi
{

namespace tags
{
enum type: int
{
    key_answer,
    key_query,
    mask_query,
    flags,
    mask_init,
    request,
    task_type,
    field_query,
    accumulated_field_query,
    halo,
    balance,
    idle,
    connection,
    confirmation,
    disconnect,
    nTags
};

} //Tags namespace

} //namespace sr_mpi

#endif
