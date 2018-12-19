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

using namespace sr_mpi;
struct ServerTraits
{
    using key_query_t = Task<tags::key_query,std::vector<int>>;
    using task_manager_t = TaskManager<key_query_t>;
};
class Server : public ServerBase<ServerTraits>
{
    
public:

    using trait_t =  ServerTraits;
    using super_type =ServerBase<ServerTraits>;
    using key_query_t = typename trait_t::key_query_t;
    using task_manager_t = typename trait_t::task_manager_t;

    template<class TaskType>
    using answer_data_type  = typename TaskType::answer_data_type;

public: // ctors

	Server(const Server&) = default;
	Server(Server&&) = default;
	Server& operator=(const Server&) & = default;
	Server& operator=(Server&&) & = default;
    ~Server()=default;
    Server()=default;

public: //members

    void run()
    {
        this->run_query<key_query_t>();
    }

private:
    boost::mpi::communicator comm_;

};


#endif 
