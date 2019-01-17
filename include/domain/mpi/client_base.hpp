#ifndef INCLUDED_CLIENT_HPP
#define INCLUDED_CLIENT_HPP

#include "task_manager.hpp"

namespace sr_mpi
{

template<class Traits>
class ClientBase
{

public: // aliases
    using task_manager_t = typename Traits::task_manager_t;

public: // ctors

	ClientBase(const ClientBase&) = default;
	ClientBase(ClientBase&&) = default;
	ClientBase& operator=(const ClientBase&) & = default;
	ClientBase& operator=(ClientBase&&) & = default;
    ~ClientBase()=default;

    ClientBase(int _server_rank=0):server_rank_(_server_rank){}

public:

    
    template<class QueryType>
    auto wait(QueryType& _q)
    {
        using send_task_t = typename QueryType::send_task_t;
        using recv_task_t = typename QueryType::recv_task_t;

        auto& send_comm=
            task_manager_.template send_communicator<send_task_t>();
        auto& recv_comm=
            task_manager_.template recv_communicator<recv_task_t>();
        
        while(true)
        {
            send_comm.start_communication();
            auto finished_tasks=send_comm.finish_communication();
            for(auto& e : finished_tasks )
            {
                auto answer=recv_comm.post_answer(e,
                        _q.recvDataPtr(0));
            }
            if(send_comm.done())
                break;
        }
        while(!recv_comm.done())
        {
            recv_comm.start_communication();
            auto ft=recv_comm.finish_communication();
            for(auto& e : ft)
            {
                std::cout<<"Received answer on rank"<<comm_.rank()<<": \n";
                for(auto& d: e->data()) std::cout<<d<<"  ";
                std::cout<<std::endl;
            }
        }
    }



    void disconnect()
    {
        const auto tag=tag_gen().get<tags::connection>(comm_.rank());
        comm_.send(server_rank_,tag, false);
    }

    const int& server() const noexcept{return server_rank_;}
    int& server() noexcept{return server_rank_;}

protected:
    boost::mpi::communicator comm_;
    task_manager_t task_manager_;
    int server_rank_;

};


} //namespace  sr_mpi
#endif 
