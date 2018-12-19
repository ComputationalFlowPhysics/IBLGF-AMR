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

//TODO: Make in place:
//          Buffer should be optional 
//          else do inpace 
//          For receive it should be posibble to complete task
//          directly from buffer
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

    //TODO: implement inplace tasks:
    template<class TaskType >
    void run_query()
    {
        //std::vector<typename TaskType::data_type> 
        //    client_recvdata_vector(comm_.size());
        //std::vector<typename TaskType::data_type> 
        //    client_senddata_vector(comm_.size());

        //const auto receiveDataMap = [&](auto& _client){ 
        //    return &client_recvdata_vector[_client.rank]; };
        //const auto sendDataMap = [&](auto& _task){ 
        //    return &client_senddata_vector[_task->rank_other()]; };
        
        std::vector<std::vector<std::shared_ptr<typename TaskType::data_type>>>
            client_recvdata_vector(comm_.size());
        std::vector<std::vector<std::shared_ptr<typename TaskType::data_type>>>
            client_senddata_vector(comm_.size());

        const auto recvData_ptr = [&](auto& _client){ 
            auto buffer=std::make_shared<typename TaskType::data_type>();
            client_recvdata_vector[_client.rank].push_back(buffer);
            return buffer.get();
            //return &client_recvdata_vector[_client.rank]; 
            };
        const auto sendData_ptr = [&](auto& _task){ 

            auto buffer=std::make_shared<typename TaskType::data_type>();
            client_senddata_vector[_task->rank_other()].push_back(buffer);
            return buffer.get();
            //return &client_senddata_vector[_task->rank_other()]; };
            };


        std::cout<<"\n\nStarting up server ...\n"<<std::endl;
        initialize();

        while (true)
        {
            auto tasks=check_clients<TaskType>(recvData_ptr);
            answer<TaskType>(tasks, sendData_ptr);
            update_client_status();

            if(task_manager_.all_done() && !connected())
                break;
        }
        std::cout<<"\n\nShutting down server ...\n"<<std::endl;
    }

protected:


    template<class TaskType, class Function>
    std::vector<std::shared_ptr<TaskType>> check_clients(Function& _getData)
    {
        auto& recv_comm=
            task_manager_.template recv_communicator<TaskType>();

        recv_comm.receive();
        auto finished_tasks=recv_comm.check();

        std::cout<<"number of finished receives "<<finished_tasks.size()<<std::endl;
        for(auto& t  : finished_tasks)
        {
            std::cout<<"Received query: ";
            for(auto& e : t->data()) { std::cout<<e<<" "; }
            std::cout<<std::endl;
        }

        //Check for new messages
        for(int i=0;i<2;++i)
        {
            for(auto& client: clients_)
            {
                auto tag= tag_gen().get<TaskType::tag()>( client.rank );
                if(comm_.iprobe(client.rank, tag))
                {
                    auto t= recv_comm.post(
                                _getData(client),
                                client.rank);
                }
            }

        }
        return finished_tasks;
    }

    template<class TaskType, class Function>
    void answer( std::vector<std::shared_ptr<TaskType>>&  _tasks,  
                 Function& _getData)
    {
        //Complete tasks
        for(auto& t: _tasks) 
        {
            auto send_data_ptr=_getData(t);
            t->complete(&t->data(),send_data_ptr);
            auto& send_comm=
                task_manager_.template send_communicator<TaskType>();
            auto task=send_comm.post_answer(t, send_data_ptr);
        }

        //Send answers
        auto& send_comm=
            task_manager_.template send_communicator<TaskType>();
        send_comm.send();
        send_comm.check();
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
