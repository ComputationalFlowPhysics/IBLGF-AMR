#ifndef INCLUDED_CLIENT_TEST_HPP
#define INCLUDED_CLIENT_TEST_HPP

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

 protected:
    using ClientBase<ClientTraits>::comm_;
    using ClientBase<ClientTraits>::task_manager_;
public: // ctors

	Client(const Client&) = default;
	Client(Client&&) = default;
	Client& operator=(const Client&) & = default;
	Client& operator=(Client&&) & = default;
    ~Client()=default;

    using super_type::ClientBase;

public:

    void test()
    {
        auto& send_comm=
            task_manager_->template send_communicator<key_query_t>();

        std::vector<int> task_dat(3,comm_.rank());
        std::vector<int> recvData;
        auto task= send_comm.post_task(&task_dat, this->server_rank_);
        QueryRegistry<key_query_t, key_query_t> mq;
        mq.register_recvMap([&recvData](int i){return &recvData;} );
        wait(mq);
    }



};
#endif 
