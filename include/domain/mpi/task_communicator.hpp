#ifndef INCLUDED_TASK_COMMUNICATOR__HPP
#define INCLUDED_TASK_COMMUNICATOR__HPP

#include<vector>
#include<array>
#include<list>
#include<queue>


namespace sr_mpi
{


struct SendMode{};
struct RecvMode{};

/**
 * @brief: Task communicator.  Manages posted messages, buffers them and calls 
 *         non-blocking send/recvs and queries their status.
 *         Can be used in both Send and Recv mode.
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
    /** * @brief Wait for all tasks to be finished */
    task_vector_t wait_all()
    {
        vector_task_ptr_t res_all;
        while(!this->done())
        {
            this->receive();
            auto ft=this->finalize();
            res_all.insert(res_all.end(), ft.begin(), ft.end());
        }
        return res_all;
    }

    /** * @brief Check if all tasks are done and nothing is in the queue */
    bool done() const noexcept
    {
        return tasks_.size()==0  && 
               buffer_queue_.size()==0 && 
               unconfirmed_tasks_.size()==0;
    }

    /** * @brief Check if all tasks are done and nothing is in the queue */
    task_ptr_t post_task(task_data_t* _dat, int _rank_other)
    {
        auto tag= tag_gen().get<task_t::tag()>(tag_rank(_rank_other));
        auto task_ptr=insert(tag, _dat, _rank_other);
        tag= tag_gen().generate<task_t::tag()>(tag_rank(_rank_other));
        return task_ptr;
    }

    /** * @brief Start communication (send or receive for this task)*/
    task_vector_t start_communication()
    {
        task_vector_t res;
        while(buffer_.is_free() && !buffer_queue_.empty())
        {
            auto task =buffer_queue_.front();
            auto ptr = buffer_.get_free_buffer();
            task->attach_buffer( ptr );
            to_buffer(task);
            sendRecv(task);

            tasks_.push_back(task);
            res.push_back(task);
            buffer_queue_.pop();
        }
        return res;
    }

    /** * @brief Finish communication (send or receive for this task)
     *           Task will also be completed at the same time
     * */
    template<class... Args>
    task_vector_t finish_communication(Args&&... args)
    {
        task_vector_t finished_;
        for(auto it=tasks_.begin();it!=tasks_.end();)
        {
            auto& t=*it;
            if( auto status_opt = t->test() )
            {
                finished_.push_back(t);
                from_buffer(t);
                //TODO: Complete the task
                //t.complete(std::forward<Args>(args)...);

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

    /********************************************************************/
    /********************************************************************/
    //Tag rank
    template < class M=Mode,
        std::enable_if_t< std::is_same<M,SendMode>::value, int > =0 >
    int tag_rank(int rank_other) const noexcept { return comm_.rank(); }

    template < class M=Mode,
        std::enable_if_t< std::is_same<M,RecvMode>::value, int > =0 >
    int tag_rank(int rank_other) const noexcept {return rank_other;  }
    
    //Communicate
    template < class M=Mode,
        std::enable_if_t< std::is_same<M,SendMode>::value, int > =0 >
    void sendRecv(task_ptr_t _t) const noexcept {_t->isend(comm_);}
    template < class M=Mode,
        std::enable_if_t< std::is_same<M,RecvMode>::value, int > =0 >
    void sendRecv(task_ptr_t _t) const noexcept {_t->irecv(comm_);}


    //To Buffer:
    template < class M=Mode,
        std::enable_if_t< std::is_same<M,SendMode>::value, int > =0 >
    void to_buffer(task_ptr_t _t) const noexcept 
    {
        _t->assign_data2buffer();
    }
    template < class M=Mode,
        std::enable_if_t< std::is_same<M,RecvMode>::value, int > =0 >
    void to_buffer(task_ptr_t _t) const noexcept 
    { }


    //From Buffer:
    template < class M=Mode,
        std::enable_if_t< std::is_same<M,SendMode>::value, int > =0 >
    void from_buffer(task_ptr_t _t) const noexcept 
    { }
    template < class M=Mode,
        std::enable_if_t< std::is_same<M,RecvMode>::value, int > =0 >
    void from_buffer(task_ptr_t _t) const noexcept 
    {
        _t->assign_buffer2data();
    }


private:

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
