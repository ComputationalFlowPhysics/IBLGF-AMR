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
    request,
    task_type,
    idle,
    connection,
    confirmation,
    disconnect,
    nTags
};

} //Tags namespace

} //namespace sr_mpi

#endif 
