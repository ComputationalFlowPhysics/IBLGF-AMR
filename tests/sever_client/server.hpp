#ifndef INCLUDED_SERVER_XX_HPP
#define INCLUDED_SERVER_XX_HPP

#include<set>
#include<optional>
#include<vector>
#include<memory>
#include<list>
#include <boost/serialization/vector.hpp>

#include <domain/mpi/task_manager.hpp>
#include <domain/mpi/server_base.hpp>
#include <domain/mpi/query_registry.hpp>

using namespace sr_mpi;
struct ServerTraits
{
    using  key_query_t =Task<tags::key_query, std::vector<int>, Inplace>;
    using task_manager_t = TaskManager<key_query_t>;
};
class Server : public ServerBase<ServerTraits>
{
    
public:

    using trait_t =  ServerTraits;
    using super_type =ServerBase<ServerTraits>;
    using key_query_t = typename trait_t::key_query_t;
    using task_manager_t = typename trait_t::task_manager_t;

public: // ctors

	Server(const Server&) = default;
	Server(Server&&) = default;
	Server& operator=(const Server&) & = default;
	Server& operator=(Server&&) & = default;
    ~Server()=default;
    Server()=default;

public: //members

    void test()
    {
        InlineQueryRegistry<key_query_t, key_query_t> mq(comm_.size());

        mq.register_completeFunc([](auto _task, auto answerData)
        {
                    std::vector<int> ans(10, -1.0*_task->rank_other());
                    *answerData=ans;
        });

        run_query(mq);
    }

private:
    boost::mpi::communicator comm_;
};


#endif 
