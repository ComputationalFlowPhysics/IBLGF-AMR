#ifndef INCLUDED_SERVER_HPP
#define INCLUDED_SERVER_HPP

#include<set>
#include<optional>
#include<vector>
#include<unordered_set>
#include<memory>
#include<list>
#include <boost/serialization/vector.hpp>

#include "task_manager.hpp"
#include "serverclient_base.hpp"

namespace sr_mpi
{

template<class Traits>
class ServerBase : public ServerClientBase<Traits>
{
    
public:
    using task_manager_t = typename Traits::task_manager_t;
    using super_type  =ServerClientBase<Traits>;
protected:

    using super_type::comm_;
    using super_type::task_manager_;

public: // ctors

    using super_type::ServerClientBase;

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
        bool operator== (const ClientInfo &other) const
        {
            return rank == other.rank;
        }

        struct chash
        {
            std::size_t operator()(const ClientInfo& c1) const
            {
                return std::hash<std::size_t>()(c1.rank);
            }

        };

    public: //members:
        int rank=-1;
    };

public: //members
    using client_set_t= std::unordered_set<ClientInfo, 
                                           typename ClientInfo::chash>; 


public: //members

    virtual void initialize()
    {
        nConnections_=this->comm_.size()-1;
        clients_.clear();
        for(int i =0;i<comm_.size();++i)
        {
            if(i==comm_.rank())continue;
            clients_.emplace(i);
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

            if(task_manager_->all_done() && !connected())
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
            task_manager_->template recv_communicator<recv_task_t>();

        recv_comm.start_communication();
        auto finished_tasks=recv_comm.finish_communication();

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
                    task_manager_->template send_communicator<send_t>();
                auto task=send_comm.post_answer(t, 
                        _q.sendDataPtr(t->rank_other()));

            }
        }

        //Send answers
        auto& send_comm=
            task_manager_->template send_communicator<send_t>();
        send_comm.start_communication();
        send_comm.finish_communication();
    }
   

    void update_client_status()
    {
        for(int i =0;i<comm_.size();++i)
        {
            const auto tag=tag_gen().get<tags::connection>(i);
            if(auto ostatus= comm_.iprobe(i,tag ) )
            {
                bool connect=false;
                comm_.recv(i,tag,connect);
                if(!connect)
                {
                    std::cout<<"DisConnected from "<<i<<std::endl;
                    --nConnections_;
                }
            }
        }
    }

    template<class TaskType>
    auto& client(const TaskType& _t)
    {
        return 
            clients_.find(_t->rank_other());
    }

    template<class TaskType>
    const auto& client(const TaskType& _t)const 
    {  clients_.find(_t->rank_other()); }

    bool connected(){return nConnections_>0;}

protected:
    int nConnections_=0;
    client_set_t clients_;

};

}

#endif 
