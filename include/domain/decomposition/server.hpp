#ifndef DOMAIN_INCLUDED_SERVER_HPP
#define DOMAIN_INCLUDED_SERVER_HPP

#include <vector>
#include <stdexcept>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/mpi/status.hpp>

#include <global.hpp>
#include <domain/decomposition/compute_task.hpp>

#include <domain/mpi/task_manager.hpp>
#include <domain/mpi/server_base.hpp>
#include <domain/mpi/query_registry.hpp>

namespace domain
{

template<class Domain>
struct ServerTraits
{
    using domain_t = Domain;
    using key_t  = typename  domain_t::key_t;
    using key_query_t = Task<tags::key_query,std::vector<key_t>>;
    using rank_query_t = Task<tags::key_query,std::vector<int>>;
    using task_manager_t = TaskManager<key_query_t, rank_query_t>;
};

/** @brief ProcessType Server 
 *  Master/Server process.
 *  Stores the full tree structure without the data.
 *  Responsible for load balancing and listens to 
 *  the client/worker processes.
 */
template<class Domain>
class Server : public ServerBase<ServerTraits<Domain>>
{

public:
    using domain_t = Domain;
    using communicator_type  = typename  domain_t::communicator_type;
    using octant_t  = typename  domain_t::octant_t;
    using key_t  = typename  domain_t::key_t;
    using ctask_t = ComputeTask<key_t>;

    using trait_t =  ServerTraits<Domain>;
    using super_type = ServerBase<trait_t>;
     
    using rank_query_t = typename trait_t::rank_query_t;
    using key_query_t = typename trait_t::key_query_t;
    using task_manager_t =typename trait_t::task_manager_t;
public:

    Server(const Server&  other) = default;
    Server(      Server&& other) = default;
    Server& operator=(const Server&  other) & = default;
    Server& operator=(      Server&& other) & = default;
    ~Server() = default;

    Server(Domain* _d, communicator_type _comm)
    :domain_(_d), comm_(_comm)
    {
        boost::mpi::communicator world;
        std::cout<<"I am the server on rank: "<<world.rank()<<std::endl;
    }

          
public:

    auto compute_distribution()
    {
        std::cout<<"Computing domain decomposition "<<std::endl;
        float_type total_load=0.0;
        for( auto it = domain_->begin_df(); it!= domain_->end_df();++it )
        {
            total_load+=it->load();
        }
        
        auto nProcs=comm_.size()-1;
        const float_type ideal_load=total_load/nProcs;

        float_type total_load_perProc=0;
        int procCount=0;
        std::vector<std::vector<ctask_t>> tasks_perProc(nProcs);
        for( auto it = domain_->begin_df(); it!= domain_->end_df();++it )
        {
            it->rank()=procCount+1;
            auto load= it->load();
            ctask_t task(it->key(), it->rank(), load);
            
            if(total_load_perProc+load<ideal_load || 
               (procCount == nProcs-1))
            {
                tasks_perProc[procCount].push_back(task);
                total_load_perProc+=load;
            }
            else 
            {
                procCount++;
                task.rank()=procCount+1;
                tasks_perProc[procCount].push_back(task);
                total_load_perProc=load;
            }
        }

        std::vector<int> total_loads_perProc(nProcs,0);
        std::ofstream ofs("load_balance.txt");
        for(int i =0; i<nProcs;++i)
        {
            for(auto t:  tasks_perProc[i] )
            {
                total_loads_perProc[i]+=t.load();
            }
            ofs<< tasks_perProc[i][0].rank()<<" "
                <<total_loads_perProc[i]<<std::endl;
        }

        //TODO: Iterate to balance/diffuse load 
        
        return tasks_perProc;
    }

    void send_keys()
    {
        auto tasks=compute_distribution();
        for(auto& t:tasks)
        {
            std::vector<key_t> keys;
            for(auto tt: t ) keys.push_back(tt.key());
            comm_.send(t[0].rank(),0, keys );
        }
    }

    void test()
    {
        InlineQueryRegistry<rank_query_t, key_query_t> mq(comm_.size());
        mq.register_completeFunc([this](auto _task, auto _answerData)
        {
            this->get_octant_rank(_task, _answerData);
        });

        this->run_query(mq);
    }

    template<class TaskPtr, class OutPtr>
    void get_octant_rank(TaskPtr _task, OutPtr _out)
    {
        std::vector<int> ranks(10, _task->rank_other());
        *_out=ranks;
    }



private:
    Domain* domain_;
    communicator_type comm_;

};

}






#endif
