#ifndef INCLUDED_TASK_COMMUNICATOR__HPP
#define INCLUDED_TASK_COMMUNICATOR__HPP

#include<vector>
#include<array>
#include<list>
#include<queue>


namespace bla
{


struct SendMode{};
struct RecvMode{};

/**
 * @brief: Task communicator.  Manages posted messages, buffers them and calls 
 *         non-blocking send/recvs and queries their status.
 *         Can be used in both Send and Recv mode.
 *
 *         TODO:  Post_answer function ...just as insert
 */
template<class TaskType, class Mode>
class TaskCommunicator
{
public:

    using task_t = TaskType;
    using task_ptr_t = std::shared_ptr<task_t>;
    using vector_task_ptr_t = std::vector<task_ptr_t>;
    using id_type = typename task_t::id_type;
    using task_vector_t =std::vector<task_ptr_t>;
    using task_data_t = typename TaskType::data_type;
    using buffer_queue_t = std::queue<task_ptr_t>;
    using buffer_container_t =typename task_t::buffer_container_type;

    using request_list_t = std::list<boost::mpi::request>;
    using query_arr_t =std::list<boost::mpi::request>;

public: //Ctor

    TaskCommunicator( int _nBuffers=2 )  
    {
        buffer_.init(_nBuffers);
    }

    //No copy or Assign 
	TaskCommunicator(const TaskCommunicator&) = delete;
	TaskCommunicator& operator=(const TaskCommunicator&) & = delete;
    ~TaskCommunicator()=default;

public:

    /** * @brief Insert task into buffer queue */
    task_ptr_t insert(const id_type& _taskId ,
                      task_data_t* _data, 
                      int _rank_other
                      ) noexcept
    {
        ++nActive_tasks_;
        auto task_ptr = std::make_shared<task_t>(_taskId);
        task_ptr->attach_data(_data);
        task_ptr->rank_other()=_rank_other;
        buffer_queue_.push( task_ptr );

        return task_ptr;
    }

    /** * @brief Post an answer of this task */
    task_ptr_t post_answer(task_ptr_t _task_ptr, task_data_t* _data) noexcept
    {
        return insert(_task_ptr->id(), _data, _task_ptr->rank_other());
    }


    /** * @brief Check if all tasks are done and nothing is in the queue */
    bool done() const noexcept
    {
        return tasks_.size()==0  && 
               buffer_queue_.size()==0 && 
               unconfirmed_tasks_.size()==0;
    }

    /********************************************************************/
    //Send mode:
    
    /** * @brief Post a send task: Insert into queue */
    template
    < 
        class M=Mode,
        std::enable_if_t< std::is_same<M,SendMode>::value, int > =0
    >
    task_ptr_t post(  task_data_t* _dat, int _dest, bool request_confirm =true)
    {
        auto tag= tag_gen().get<task_t::tag()>( comm_.rank() );
        auto task_ptr=insert(tag, _dat, _dest);

        //std::cout<<"Post Send:    Rank:"<<comm_.rank()<<" "
        //         <<"to " <<task_ptr->rank_other()<<" "
        //         <<"ID: "<<task_ptr->id()<<std::endl;
        tag= tag_gen().generate<task_t::tag()>( comm_.rank() );
        return task_ptr;
    }

    /** * @brief Try sending when buffer is free */
    template
    < 
        class M=Mode,
        std::enable_if_t< std::is_same<M,SendMode>::value, int > =0
    >
    void send()
    {
        auto assigned_tasks= data_to_buffer() ;
        for(auto& t :  assigned_tasks)
        {
            tasks_.push_back(t);
            t->isend(comm_);
        }
    }

    /** * @brief Get finished tasks */
    template
    < 
        class M=Mode,
        std::enable_if_t< std::is_same<M,SendMode>::value, int > =0 
    >
    task_vector_t check()
    {
        task_vector_t finished_;
        for(auto it=tasks_.begin();it!=tasks_.end();)
        {
            auto& t=*it;
            if( auto status_opt = t->test() )
            {
                finished_.push_back(t);
                t->deattach_buffer();

                insert_unconfirmed_tasks(t);

                it=tasks_.erase(it);
                --nActive_tasks_;
            }
            else { ++it; }
        }

        check_unconfirmed_tasks();
        return finished_;
    }

    /** * @brief Wait for all posted tasks to finish */
    template
    < 
        class M=Mode,
        std::enable_if_t< std::is_same<M,SendMode>::value, int > =0 
    >
    void wait_all()
    {
        while(true)
        {
            this->send();
            this->check();
            if(this->done())
                break;
        }
    }

    /********************************************************************/
    //Receive mode:
    
    /** * @brief Post a receive task */
    template 
    < 
        class M=Mode,
        std::enable_if_t< std::is_same<M,RecvMode>::value, int > =0 
    >
    task_ptr_t post(  task_data_t* _dat, int _src  )
    {
        auto tag= tag_gen().get<task_t::tag()>(_src );
        auto task_ptr=insert(tag, _dat, _src);
        tag= tag_gen().generate<task_t::tag()>(_src);
        return task_ptr;

    }
    
    /** * @brief Buffer and receive */
    template 
    < 
        class M=Mode,
        std::enable_if_t< std::is_same<M,RecvMode>::value, int > =0 
    >
    task_vector_t receive()
    {
        task_vector_t res;
        while(buffer_.is_free() && !buffer_queue_.empty())
        {
            auto task =buffer_queue_.front();
            auto ptr = buffer_.get_free_buffer();
            task->attach_buffer( ptr );
            task->irecv(comm_);

            //std::cout
            //         <<"Post Receive: Rank:"<<comm_.rank()<<" "
            //         <<"from " <<task->rank_other()<<" "
            //         <<"ID: "<<task->id()<<std::endl;

            tasks_.push_back(task);

            res.push_back(task);
            buffer_queue_.pop();
        }
        return res;
    }

    /** * @brief Get finished receives */
    template 
    < 
        class M=Mode,
        std::enable_if_t< std::is_same<M,RecvMode>::value, int > =0 
    >
    task_vector_t check()
    {
        task_vector_t finished_;
        for(auto it=tasks_.begin();it!=tasks_.end();)
        {
            auto& t=*it;
            if( auto status_opt = t->test() )
            {
                finished_.push_back(t);
                t->assign_buffer2data();
                t->deattach_buffer();

                insert_unconfirmed_tasks(t);

                it=tasks_.erase(it);
                --nActive_tasks_;
            }
            else { ++it; }
        }
        check_unconfirmed_tasks();
        return finished_;
    }

    /** * @brief Wait for all posted tasks to finish */
    template
    < 
        class M=Mode,
        std::enable_if_t< std::is_same<M,RecvMode>::value, int > =0 
    >
    task_vector_t wait_all()
    {
        vector_task_ptr_t res_all;
        while(!this->done())
        {
            this->receive();
            auto ft=this->check();
            res_all.insert(res_all.end(), ft.begin(), ft.end());
        }
        return res_all;
    }


private:

    /** * @brief Assign data to buffer if possible */
    task_vector_t data_to_buffer()
    {
        task_vector_t res;
        while(buffer_.is_free() && !buffer_queue_.empty())
        {
            auto task =buffer_queue_.front();
            auto ptr = buffer_.get_free_buffer();
            task->attach_buffer( ptr );
            task->assign_data2buffer();
            res.push_back(task);
            buffer_queue_.pop();
        }
        return res;
    }

    void insert_unconfirmed_tasks(task_ptr_t& t) noexcept
    {
        if(t->requires_confirmation())
        {
            auto confirmed=t->confirmed();
            if(!confirmed)
            {
                unconfirmed_tasks_.push_back(t);
            }
        }
    }
    void check_unconfirmed_tasks() noexcept
    {
        for(auto it=unconfirmed_tasks_.begin(); it!=unconfirmed_tasks_.end();)
        {
            if((*it)->confirmed())
            {
                it=unconfirmed_tasks_.erase(it);
            }
            else { ++it; }
        }
    }


private:

    boost::mpi::communicator comm_; ///< Mpi communicator.
    int nActive_tasks_=0;           ///< Number of active tasks.

    task_vector_t tasks_;             ///< Taks that currently are being send.
    buffer_container_t buffer_;       ///< Data buffer to be send for each task.
    buffer_queue_t buffer_queue_;     ///< Queue of tasks to fill the send buffer
    task_vector_t unconfirmed_tasks_; ///< Unconfirmed tasks

};

template<class TaskType>
using SendTaskCommunicator = TaskCommunicator<TaskType,SendMode>;
    
template<class TaskType>
using RecvTaskCommunicator = TaskCommunicator<TaskType,RecvMode>;
}

#endif 
