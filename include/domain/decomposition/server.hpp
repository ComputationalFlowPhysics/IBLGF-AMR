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
#include "serverclient_traits.hpp"

namespace domain
{


/** @brief ProcessType Server
 *  Master/Server process.
 *  Stores the full tree structure without the data.
 *  Responsible for load balancing and listens to
 *  the client/worker processes.
 */
template<class Domain>
class Server : public ServerBase<ServerClientTraits<Domain>>
{

public:
    using domain_t = Domain;
    using communicator_type  = typename  domain_t::communicator_type;
    using octant_t  = typename  domain_t::octant_t;
    using key_t  = typename  domain_t::key_t;
    using ctask_t = ComputeTask<octant_t>;

    using trait_t =  ServerClientTraits<Domain>;
    using super_type = ServerBase<trait_t>;

    using rank_query_t   = typename trait_t::rank_query_t;
    using key_query_t    = typename trait_t::key_query_t;
    using task_manager_t = typename trait_t::task_manager_t;

public: //Ctors

    using super_type::ServerBase;

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


        domain_->tree()->construct_neighbor_lists();
        domain_->tree()->construct_influence_lists();

        float_type total_load=0.0;
        int c=0;
        for( auto it = domain_->begin_df(); it!= domain_->end_df();++it )
        {
            total_load+=it->load();
            ++c;
        }
        std::cout<<"Total number of octants "<<c<<std::endl;

        auto nProcs=comm_.size()-1;
        const float_type ideal_load=total_load/nProcs;

        float_type total_load_perProc=0;
        int procCount=0;

        std::vector<std::list<ctask_t>> tasks_perProc(nProcs);
        for( auto it = domain_->begin_df(); it!= domain_->end_df();++it )
        {
            it->rank()=procCount+1;

            auto load= it->load();
            ctask_t task(it.ptr(), it->rank(), load);

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
                it->rank()=procCount+1;
                tasks_perProc[procCount].push_back(task);
                total_load_perProc=load;
            }
        }

        float_type max_load=-1;
        float_type min_load=total_load+10;
        std::vector<int> total_loads_perProc(nProcs,0);
        for(int i =0; i<nProcs;++i)
        {
            for(auto t:  tasks_perProc[i] )
            {
                total_loads_perProc[i]+=t.load();
            }
            if(total_loads_perProc[i] < min_load) min_load=total_loads_perProc[i];
            if(total_loads_perProc[i] > max_load) max_load=total_loads_perProc[i];
        }

        //Iterate the redistribute load ..
        float_type imbalance=max_load-min_load;
        while(true)
        {
            float_type max_load=-1;
            float_type min_load=total_load+10;

            for(int i =0; i<nProcs-1;++i)
            {
                const float_type dx_load=
                    total_loads_perProc[i+1]-total_loads_perProc[i];

                //Shuffle around
                if(dx_load>0 && tasks_perProc[i+1].size() > 1)
                {
                    auto task_to_move =tasks_perProc[i+1].front();
                    const auto load_to_move = task_to_move.load();

                    const auto new_total_np1= total_loads_perProc[i+1]-load_to_move;
                    const auto new_total = total_loads_perProc[i]+load_to_move;

                    task_to_move.rank()=tasks_perProc[i].back().rank();
                    task_to_move.octant()->rank()=task_to_move.rank();
                    tasks_perProc[i].push_back(task_to_move);
                    tasks_perProc[i+1].pop_front();

                    total_loads_perProc[i+1]=new_total_np1;
                    total_loads_perProc[i]=new_total;
                }
                else if(dx_load<0 && tasks_perProc[i].size()>1)
                {
                    auto task_to_move =tasks_perProc[i].back();
                    const auto load_to_move = task_to_move.load();

                    const auto new_total_np1= total_loads_perProc[i+1]+load_to_move;
                    const auto new_total = total_loads_perProc[i]-load_to_move;

                    task_to_move.rank()=tasks_perProc[i+1].front().rank();
                    task_to_move.octant()->rank()=task_to_move.rank();
                    tasks_perProc[i+1].push_front(task_to_move);
                    tasks_perProc[i].pop_back();

                    total_loads_perProc[i+1]=new_total_np1;
                    total_loads_perProc[i]=new_total;

                }

                if(total_loads_perProc[i] < min_load) 
                    min_load=total_loads_perProc[i];
                if(total_loads_perProc[i] > max_load)
                    max_load=total_loads_perProc[i];
            }

            //Update last:
            if(total_loads_perProc[nProcs-1] < min_load) 
                min_load=total_loads_perProc[nProcs-1];
            if(total_loads_perProc[nProcs-1] > max_load) 
                max_load=total_loads_perProc[nProcs-1];

            const auto imbalance_new = max_load-min_load;
            if((  imbalance_new >= imbalance  ) ) break;
            imbalance=imbalance_new;
        }


        //Iterate to balance/diffuse load
        std::vector<int> total_loads_perProc2(nProcs,0);
        std::ofstream ofs("load_balance.txt");
        for(int i =0; i<nProcs;++i)
        {
            int rank=0;
            for(auto t:  tasks_perProc[i] )
            {
                total_loads_perProc2[i]+=t.load();
                rank=t.rank();
            }
            ofs<<rank<<" "
                <<total_loads_perProc2[i]<<std::endl;
        }

        std::cout<<"Done with initial load balancing"<<std::endl;
        return tasks_perProc;
    }



    void send_keys()
    {
        auto tasks=compute_distribution();
        for(auto& t:tasks)
        {
            std::vector<key_t> keys;
            for(auto tt: t ) keys.push_back(tt.key());
            comm_.send(t.front().rank(),0, keys );
        }
    }

    void rank_query()
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
        _out->resize(_task->data().size());
        int count=0;
        for(auto& key :  _task->data())
        {
            //std::cout<<key<<std::endl;
            auto oct =domain_->tree()->find_octant(key);
            if(oct)
                (*_out)[count++]=oct->rank();
            else
                (*_out)[count++]=-1;
        }
    }


private:
    Domain* domain_;
    communicator_type comm_;

};

}






#endif
