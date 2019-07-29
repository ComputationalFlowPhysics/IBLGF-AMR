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

    using mask_init_query_send_t   = typename trait_t::mask_init_query_send_t;
    using mask_init_query_recv_t   = typename trait_t::mask_init_query_recv_t;

    using leaf_query_send_t   = typename trait_t::leaf_query_send_t;
    using leaf_query_recv_t   = typename trait_t::leaf_query_recv_t;

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


    auto compute_distribution() const noexcept
    {
        std::cout<<"Computing domain decomposition for "<<comm_.size()<<" processors" <<std::endl;

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

        //auto it = domain_->begin_df();
        auto it = domain_->begin_bf();
        float_type current_load= 0.;
        for(int crank=0;crank<nProcs;++crank)
        {
            float_type target_load= (static_cast<float_type>(crank+1)/nProcs)*total_load;
            target_load=std::min(target_load,total_load);
            float_type octant_load=0.;
            int count=0;
            while(current_load<=target_load || count ==0)
            {
                it->rank()=crank+1;
                auto load= it->load();

                ctask_t task(it.ptr(), it->rank(), load);
                tasks_perProc[crank].push_back(task);
                current_load+=load;
                octant_load+=load;
                ++it;
                ++count;
                if(it==domain_->end_bf()) break;
                //if(it==domain_->end_df()) break;
            }
            if(it==domain_->end_bf()) break;
            //if(it==domain_->end_df()) break;
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
        ofs<<"max/ideal load: "<<max_load<<"/"<<ideal_load<<" = "<<max_load/ideal_load<<std::endl;
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
        //Send global tree depth
        for(int i=1;i<comm_.size();++i)
        {
            comm_.send(i,0, domain_->tree()->depth()) ;
        }
    }

    void mask_query()
    {
        InlineQueryRegistry<mask_init_query_recv_t, mask_init_query_send_t> mq(comm_.size());
        mq.register_completeFunc([this](auto _task, auto _answerData)
        {
            this->get_octant_mask(_task, _answerData);
        });

        this->run_query(mq);
    }

    void leaf_query()
    {
        InlineQueryRegistry<leaf_query_recv_t, leaf_query_send_t> mq(comm_.size());
        mq.register_completeFunc([this](auto _task, auto _answerData)
        {
            this->get_octant_leaf(_task, _answerData);
        });

        this->run_query(mq);
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
    void get_octant_mask(TaskPtr _task, OutPtr _out)
    {
        _out->resize(_task->data().size());
        int count=0;
        for(auto& key : _task->data())
        {
            auto oct =domain_->tree()->find_octant(key);
            if(oct)
            { (*_out)[count++]=(oct->fmm_mask());
            }
            else
            {
                throw std::runtime_error(
                        "can't find octant for mask query");
            }
        }
    }

    template<class TaskPtr, class OutPtr>
    void get_octant_leaf(TaskPtr _task, OutPtr _out)
    {
        _out->resize(_task->data().size());
        int count=0;
        for(auto& key : _task->data())
        {
            auto oct =domain_->tree()->find_octant(key);
            if(oct)
            { (*_out)[count++]=(oct->is_leaf());
            }
            else
            {
                (*_out)[count++]=false;
            }
        }
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
