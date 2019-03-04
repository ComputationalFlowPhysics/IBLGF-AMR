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

    auto compute_distribution()const noexcept
    {
        return compute_distribution_1();
    }
    auto compute_distribution_1() const noexcept
    {
        std::cout<<"Computing domain decomposition "<<std::endl;

        domain_->tree()->construct_neighbor_lists();
        domain_->tree()->construct_influence_lists();

        float_type total_load=0.0;
        int nOctants=0;
        for( auto it = domain_->begin_df(); it!= domain_->end_df();++it )
        {
            total_load+=it->load();
            ++nOctants;
        }
        std::cout<<"Total number of octants "<<nOctants<<std::endl;
        auto nProcs=comm_.size()-1;
        const float_type ideal_load=total_load/nProcs;

        std::vector<std::list<ctask_t>> tasks_perProc(nProcs);
        // Load distribution assuming equal loads
        float_type chunks=static_cast<float_type>(nOctants)/nProcs;
        //auto it = domain_->begin_df();
        auto it = domain_->begin_bf();
        for ( int i=0; i<nProcs;++i )
        {
            size_t start= (i*chunks);
            size_t end= ((i+1)*chunks);
            const int nlocal=end-start;
            for(int j=0;j<nlocal;++j)
            {
                it->rank()=i+1;
                auto load= it->load();
                ctask_t task(it.ptr(), it->rank(), load);
                tasks_perProc[i].push_back(task);
                ++it;
            }
        }

        //Iterate to balance/diffuse load
        std::vector<float_type> total_loads_perProc2(nProcs,0);
        float_type max_load=-1;
        float_type min_load=total_load+10;
        std::ofstream ofs("load_balance.txt");
        for(int i =0; i<nProcs;++i)
        {
            for(auto& t:  tasks_perProc[i] )
            {
                total_loads_perProc2[i]+=t.load();
            }
            if(total_loads_perProc2[i]>max_load) max_load=total_loads_perProc2[i];
            if(total_loads_perProc2[i]<min_load) min_load=total_loads_perProc2[i];
            ofs<<total_loads_perProc2[i]<<std::endl;
        }
        ofs<<"max/min load: "<<max_load<<"/"<<min_load<<" = "<<max_load/min_load<<std::endl;
        ofs<<"total_load: "<<total_load<<std::endl;
        ofs<<"ideal_load: "<<ideal_load<<std::endl;

        std::cout<<"Done with initial load balancing"<<std::endl;
        return tasks_perProc;
    }

    auto compute_distribution_2() const noexcept
    {
        std::cout<<"Computing domain decomposition "<<std::endl;


        domain_->tree()->construct_neighbor_lists();
        domain_->tree()->construct_influence_lists();

        float_type total_load=0.0;
        int nOctants=0;
        for( auto it = domain_->begin_df(); it!= domain_->end_df();++it )
        {
            total_load+=it->load();
            ++nOctants;
        }
        std::cout<<"Total number of octants "<<nOctants<<std::endl;
        auto nProcs=comm_.size()-1;

        const float_type ideal_load=total_load/nProcs;
        float_type total_load_perProc=0;

        int procCount=0;
        int overhead_count=0;
        std::vector<std::list<ctask_t>> tasks_perProc(nProcs);
        for( auto it = domain_->begin_bf(); it!= domain_->end_bf();++it )
        {
            it->rank()=procCount+1;

            auto load= it->load();
            ctask_t task(it.ptr(), it->rank(), load);

            if(total_load_perProc+load<ideal_load ||
                    (procCount == nProcs-1))
            {
                tasks_perProc[procCount].push_back(task);
                total_load_perProc+=load;

                if(procCount == nProcs-1 && total_load_perProc >ideal_load ) 
                {
                    ++overhead_count;
                }
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

        //std::cout<<" Initial overhead_count "<<overhead_count<<std::endl;
        //propagate overhead backward
        for(int k =0;k<overhead_count;++k)
        {
            //Get maximum imbalance 
            int stop_idx=-1;
            float_type max_imbalance=-1;;
            std::vector<int> total_loads_perProc(nProcs,0);
            for(int i =0; i<nProcs;++i)
            {
                for(auto t:  tasks_perProc[i] )
                {
                    total_loads_perProc[i]+=t.load();
                }
                if( ideal_load - total_loads_perProc[i] > max_imbalance )
                {
                    max_imbalance= ideal_load - total_loads_perProc[i];
                    stop_idx=i;
                }
            }

            //Backpropagate the imbalance:
            for(int i =nProcs-1; i>=1;--i)
            {
                //take the front and put it in the back
                auto task_to_move =tasks_perProc[i].front();
                const auto load_to_move = task_to_move.load();

                const auto new_total= total_loads_perProc[i]-load_to_move;
                const auto new_total_m1 = total_loads_perProc[i-1]+load_to_move;
                
                task_to_move.octant()->rank()=tasks_perProc[i-1].back().rank();
                task_to_move.rank()=tasks_perProc[i-1].back().rank();

                tasks_perProc[i-1].push_back(task_to_move);
                tasks_perProc[i].pop_front();

                total_loads_perProc[i]=new_total;
                total_loads_perProc[i-1]=new_total_m1;

                if(i-1 == stop_idx)
                    break;
            }
        }

        std::vector<float_type> total_loads_perProc2(nProcs,0);
        float_type max_load=-1;
        float_type min_load=total_load+10;
        std::ofstream ofs("load_balance.txt");
        for(int i =0; i<nProcs;++i)
        {
            for(auto& t:  tasks_perProc[i] )
            {
                total_loads_perProc2[i]+=t.load();
            }
            if(total_loads_perProc2[i]>max_load) max_load=total_loads_perProc2[i];
            if(total_loads_perProc2[i]<min_load) min_load=total_loads_perProc2[i];
            ofs<<total_loads_perProc2[i]<<std::endl;
        }
        ofs<<"max/min load: "<<max_load<<"/"<<min_load<<" = "<<max_load/min_load<<std::endl;
        ofs<<"total_load: "<<total_load<<std::endl;
        ofs<<"ideal_load: "<<ideal_load<<std::endl;

        std::cout<<"Done with initial load balancing"<<std::endl;
        return tasks_perProc;
    }



    void send_keys()
    {
        auto tasks=compute_distribution();
        for(auto& t:tasks)
        {
            std::vector<key_t> keys;
            for(auto& tt: t ) keys.push_back(tt.key());
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
