#ifndef INCLUDED_SERVER_HPP
#define INCLUDED_SERVER_HPP

#include <boost/serialization/vector.hpp>
#include<set>
#include<optional>
#include<vector>
#include<memory>
#include<list>

#include <mpi/tags.hpp>
#include <mpi/tag_generator.hpp>
#include <mpi/task_manager.hpp>

namespace sr_mpi
{
class Server
{
    

public:

    using key_query_t = Task<tags::key_query,std::vector<int> >;
    using task_manager_t = TaskManager<key_query_t>;

    template<class TaskType>
    using answer_task_type  = typename TaskType::answer_task_type;

    template<class TaskType>
    using answer_data_type  = typename TaskType::answer_data_type;

public: // ctors

	Server(const Server&) = default;
	Server(Server&&) = default;
	Server& operator=(const Server&) & = default;
	Server& operator=(Server&&) & = default;
    ~Server()=default;
    Server()=default;

private: //helper struct

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
        
        for(int i =1;i<=nConnections_;++i)
        {
            clients_.emplace_back(i);
        }
        client_recvdata_vector.clear();
        client_recvdata_vector.resize(comm_.size());
        client_senddata_vector.resize(comm_.size());

    }

    void run()
    {
        std::cout<<"\n\nStarting up server ...\n"<<std::endl;
        initialize();
        while (true)
        {
            auto tasks=check_clients<key_query_t>();
            post_answers<key_query_t>(tasks);
            send_answers<key_query_t>();
            update_client_status();

            if(task_manager_.all_done() && !connected())
                break;

        }
        
        std::cout<<"\n\nShutting down server ...\n"<<std::endl;
    }

    template<class TaskType>
    std::vector<std::shared_ptr<TaskType>> check_clients()
    {
        auto& recv_comm=task_manager_.recv_communicator<TaskType>();

        recv_comm.receive();
        auto finished_tasks=recv_comm.check();

        for(auto& t  : finished_tasks)
        {
            std::cout<<"Received query: ";
            for(auto& e : t->data()) { std::cout<<e<<" "; }
            std::cout<<std::endl;
        }

        //Check for new messages
        for(auto& client: clients_)
        {
            auto tag= tag_gen().get<TaskType::tag()>( client.rank );
            if(comm_.iprobe(client.rank, tag))
            {
                auto t= recv_comm.post(
                            &client_recvdata_vector[client.rank], 
                            client.rank);
            }
        }
        return finished_tasks;
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

private:

    template<class TaskType>
    void post_answers( std::vector<std::shared_ptr<TaskType>>&  _tasks)
    {
        for(auto& t: _tasks) 
        {
            t->complete(&t->data(),     
                        &client_senddata_vector[t->rank_other()]);
            auto& send_comm=task_manager_.send_communicator<TaskType>();
            auto task=send_comm.post_answer(t,
                            &client_senddata_vector[t->rank_other()]);
        }
    }

    template<class TaskType>
    void send_answers()
    {
        auto& send_comm=task_manager_.send_communicator<TaskType>();
        send_comm.send();
        send_comm.check();
    }
    
private:

    boost::mpi::communicator comm_;
    int nConnections_=0;
    std::vector<ClientInfo> clients_;
    task_manager_t task_manager_;
    std::vector<std::vector<int>> client_recvdata_vector;
    std::vector<std::vector<int>> client_senddata_vector;

};

}

#endif 
