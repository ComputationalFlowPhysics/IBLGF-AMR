#ifndef INCLUDED_CLIENT_XX_HPP
#define INCLUDED_CLIENT_XX_HPP

#include <domain/mpi/task_manager.hpp>
#include <domain/mpi/client_base.hpp>

using namespace sr_mpi;

struct ClientTraits 
{
    using key_query_t = Task<tags::key_query,std::vector<int>>;
    using task_manager_t = TaskManager<key_query_t>;
};

class Client : public ClientBase<ClientTraits>
{

public: // aliases

    using trait_t =  ClientTraits;
    using super_type  = ClientBase<trait_t>;
    using key_query_t = typename trait_t::key_query_t;
    using task_manager_t = typename trait_t::task_manager_t;

public: // ctors

	Client(const Client&) = default;
	Client(Client&&) = default;
	Client& operator=(const Client&) & = default;
	Client& operator=(Client&&) & = default;
    ~Client()=default;

    Client()=default;

public:

    void test_query()
    {
        std::vector<int> task_dat(3,comm_.rank());
        std::vector<int> task_dat2(3,comm_.rank());

        recv_dat.resize(3);


        auto& send_comm=
            task_manager_.template send_communicator<key_query_t>();
        auto& recv_comm=
            task_manager_.template recv_communicator<key_query_t>();

        //Send random queries:
        auto task= send_comm.post_task(&task_dat, 0);
        if(comm_.rank()==1)
        {
            auto task= send_comm.post_task(&task_dat2, 0);
        }

        int count=0;
        while(true)
        {
            send_comm.start_communication();
            auto finished_tasks=send_comm.finish_communication();
            for(auto& e : finished_tasks )
            {
                recv_dat.push_back(std::vector<int>(3,0));
                auto answer=recv_comm.post_answer(e,&recv_dat[count++]);
            }
            if(send_comm.done())
                break;
        }
        
        //Busy wait:
        while(!recv_comm.done())
        {
            recv_comm.start_communication();
            auto ft=recv_comm.finish_communication();
            for(auto& e : ft)
            {
            std::cout<<"Received answer: ";
                for(auto& d: e->data()) std::cout<<d<<" ";
                std::cout<<std::endl;
            }
        }
    }

    void disconnect()
    {
        const auto tag=tag_gen().get<tags::connection>(comm_.rank());
        comm_.send(0,tag, false);
    }

private:
    boost::mpi::communicator comm_;
    task_manager_t task_manager_;
    std::vector<std::vector<int>> recv_dat;

};
#endif 
