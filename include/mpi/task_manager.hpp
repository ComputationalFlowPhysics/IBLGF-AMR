#ifndef INCLUDED_TASK_MANAGER__HPP
#define INCLUDED_TASK_MANAGER__HPP

#include<vector>
#include<array>
#include<list>
#include<queue>

#include <mpi/tags.hpp>
#include <mpi/task.hpp>
#include <mpi/tag_generator.hpp>
#include <mpi/task_communicator.hpp>

namespace sr_mpi
{

template<class... TasksType>
class TaskManager
{
public:


    template<class TaskType>
    using send_comm_t = SendTaskCommunicator<TaskType>;

    template<class TaskType>
    using recv_comm_t = RecvTaskCommunicator<TaskType>;

    using send_comm_tuple_t = std::tuple<send_comm_t<TasksType>...>;
    using recv_comm_tuple_t = std::tuple<recv_comm_t<TasksType>...>;

public: //Ctor
    TaskManager()
    {
    }

public: //Memebers:

    template<class TaskType>
    send_comm_t<TaskType>& send_communicator() noexcept
    {
        return std::get<send_comm_t<TaskType>>(send_comms_);
    } 
    
    template<class TaskType>
    const send_comm_t<TaskType>& send_communicator()const noexcept
    {
        return std::get<send_comm_t<TaskType>>(send_comms_);
    }
    
    template<class TaskType>
    const recv_comm_t<TaskType>& recv_communicator()const  noexcept
    {
        return std::get<recv_comm_t<TaskType>>(recv_comms_);
    }
    template<class TaskType>
    recv_comm_t<TaskType>& recv_communicator() noexcept
    {
        return std::get<recv_comm_t<TaskType>>(recv_comms_);
    }

    template<class TaskType>
    bool done()const noexcept{ return send_communicator<TaskType>.done()&&
                                      recv_communicator<TaskType>.done(); }

    bool all_done() const noexcept
    {
        bool all_done_=true;
        tuple_utils::for_each(send_comms_,
                [&](const auto& _comm ){
                    if(!_comm.done()) all_done_=false;
                }
        );
        tuple_utils::for_each(recv_comms_,
                [&](const auto& _comm ){
                    if(!_comm.done()) all_done_=false;
                }
        );
        return all_done_;
    }


private:

private:

    boost::mpi::communicator comm_;

    send_comm_tuple_t send_comms_;
    recv_comm_tuple_t recv_comms_;

    int nActive_tasks_=0;
 };

}

#endif 
