#ifndef INCLUDED_CLIENT_XX_HPP
#define INCLUDED_CLIENT_XX_HPP

#include <domain/mpi/task_manager.hpp>
#include <domain/mpi/client_base.hpp>
#include <domain/mpi/query_registry.hpp>

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

    void test()
    {
        auto& send_comm=
            this->task_manager_.template send_communicator<key_query_t>();

        std::vector<int> task_dat(3,comm_.rank());
        std::vector<int> recvData;
        auto task= send_comm.post_task(&task_dat, 0);
        QueryRegistry<key_query_t, key_query_t> mq;
        mq.register_recvMap([&recvData](int i){return &recvData;} );
        wait(mq);
    }


private:
    boost::mpi::communicator comm_;

};
#endif 
