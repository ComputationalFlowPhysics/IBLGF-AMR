#ifndef INCLUDED_CLIENT_HPP
#define INCLUDED_CLIENT_HPP

#include "task_manager.hpp"

namespace sr_mpi
{

template<class Traits>
class ClientBase
{

public: // aliases
    using key_query_t = Task<tags::key_query,std::vector<int>>;
    using key_answer_t = Task<tags::key_answer,std::vector<int> >;
    using task_manager_t = typename Traits::task_manager_t;

public: // ctors

	ClientBase(const ClientBase&) = default;
	ClientBase(ClientBase&&) = default;
	ClientBase& operator=(const ClientBase&) & = default;
	ClientBase& operator=(ClientBase&&) & = default;
    ~ClientBase()=default;

    ClientBase()=default;

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
        auto task= send_comm.post_new(&task_dat, 0);
        if(comm_.rank()==1)
        {
            auto task= send_comm.post_new(&task_dat2, 0);
        }

        int count=0;
        while(true)
        {
            send_comm.send();
            auto finished_tasks=send_comm.check();
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
            recv_comm.receive();
            auto ft=recv_comm.check();
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
}
#endif 
