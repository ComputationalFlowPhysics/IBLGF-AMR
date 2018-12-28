#ifndef INCLUDED_SERVER_HPP
#define INCLUDED_SERVER_HPP

#include<set>
#include<optional>
#include<vector>
#include<memory>
#include<list>
#include <boost/serialization/vector.hpp>

#include "task_manager.hpp"

namespace sr_mpi
{

template<class Traits>
class ServerBase
{
    
public:
    using task_manager_t = typename Traits::task_manager_t;

public: // ctors

	ServerBase(const ServerBase&) = default;
	ServerBase(ServerBase&&) = default;
	ServerBase& operator=(const ServerBase&) & = default;
	ServerBase& operator=(ServerBase&&) & = default;
    ~ServerBase()=default;
    ServerBase()=default;



protected: //helper struct

    class ClientInfo
    {
    public: //Ctor:

        ClientInfo(const ClientInfo&) = delete;
        ClientInfo(ClientInfo&&) = default;
        ClientInfo& operator=(const ClientInfo&) & = delete;
        ClientInfo& operator=(ClientInfo&&) & = default;
        ClientInfo(int _rank):rank(_rank){}

    public: //Operators:
        bool operator< (const ClientInfo &other) const
        {
            return rank < other.rank;
        }

    public: //members:
        int rank=-1;
    };

public: //members

    void initialize()
    {
        nConnections_=comm_.size()-1;
        clients_.clear();
        for(int i =1;i<=nConnections_;++i)
        {
            clients_.emplace_back(i);
        }
    }

    template<class QueryType>
    void run_query(QueryType& _q)
    {
        initialize();
        while (true)
        {
            auto tasks=get_tasks < QueryType > ( _q);
            do_tasks<QueryType>(tasks, _q);
            update_client_status();

            if(task_manager_.all_done() && !connected())
                break;
        }
    }

protected:


    template<class QueryType>
    std::vector<std::shared_ptr<typename QueryType::recv_task_t>> 
    get_tasks(QueryType& _q, int  nChecks=1)
    {
        using recv_task_t = typename QueryType::recv_task_t;
        auto& recv_comm=
            task_manager_.template recv_communicator<recv_task_t>();

        recv_comm.start_communication();
        auto finished_tasks=recv_comm.finish_communication();

        //for(auto& t  : finished_tasks)
        //{
        //    std::cout<<"Received query: \n";
        //    for(auto& e : t->data()) { std::cout<<e<<"\n"; }
        //    std::cout<<std::endl;
        //}

        for(int i=0;i<nChecks;++i)
        {
            for(auto& client: clients_)
            {
                auto tag= tag_gen().get<recv_task_t::tag()>( client.rank );
                if(comm_.iprobe(client.rank, tag))
                {
                    auto t= recv_comm.post_task(
                                _q.recvDataPtr(client.rank),
                                client.rank);
                }
            }
        }
        return finished_tasks;
    }

    template<class QueryType, class T >
    void do_tasks( T&  _tasks, QueryType& _q)
    {
        using  send_t =typename QueryType::send_task_t;
        
        //Complete tasks
        for(auto& t: _tasks) 
        {
            _q.complete(t);
            if(_q.sendDataPtr(t->rank_other()))
            {
                auto& send_comm=
                    task_manager_.template send_communicator<send_t>();
                auto task=send_comm.post_answer(t, 
                        _q.sendDataPtr(t->rank_other()));

            }
        }

        //Send answers
        auto& send_comm=
            task_manager_.template send_communicator<send_t>();
        send_comm.start_communication();
        send_comm.finish_communication();
    }
   

    void update_client_status()
    {
        for(auto& client : clients_)
        {
            const auto tag=tag_gen().get<tags::connection>(client.rank);
            if(auto ostatus= comm_.iprobe(client.rank,tag ) )
            {
                const auto status=*ostatus;
                bool conn;
                comm_.recv(status.source(),tag,conn);
                --nConnections_;
            }
        }
    }

    template<class TaskType>
    auto& client(const TaskType& _t)noexcept
    {return clients_[_t->rank_other()-1];}

    template<class TaskType>
    const auto& client(const TaskType& _t)const noexcept
    {return clients_[_t->rank_other()-1];}

    bool connected(){return nConnections_>0;}

protected:

    boost::mpi::communicator comm_;
    int nConnections_=0;
    std::vector<ClientInfo> clients_;
    task_manager_t task_manager_;

};

}

#endif 
