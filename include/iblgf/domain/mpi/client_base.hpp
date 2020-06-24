//      ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
//      ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌
//          ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄
//          ▐░▌     ▐░░░░░░░░░░▌ ▐░▌          ▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌
//          ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌
//      ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌
//      ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀

#ifndef INCLUDED_CLIENT_HPP
#define INCLUDED_CLIENT_HPP

#include "task_manager.hpp"
#include "serverclient_base.hpp"
#include "server_base.hpp"

namespace iblgf
{
namespace sr_mpi
{
template<class Traits>
class ClientBase : public ServerClientBase<Traits>
{
  public: // aliases
    using task_manager_t = typename Traits::task_manager_t;

  protected:
    using ServerClientBase<Traits>::comm_;
    using ServerClientBase<Traits>::task_manager_;

  public: // ctors
    ClientBase(const ClientBase&) = default;
    ClientBase(ClientBase&&) = default;
    ClientBase& operator=(const ClientBase&) & = default;
    ClientBase& operator=(ClientBase&&) & = default;
    ~ClientBase() = default;

    ClientBase(int _server_rank = 0)
    : server_rank_(_server_rank)
    {
    }

  public:
    template<class QueryType>
    auto wait(QueryType& _q)
    {
        while (!done_sending(_q)) {}
        while (!done_receiving(_q)) {}
    }

    template<class QueryType>
    bool done_receiving(QueryType& _q)
    {
        using recv_task_t = typename QueryType::recv_task_t;
        auto& recv_comm =
            task_manager_->template recv_communicator<recv_task_t>();

        recv_comm.start_communication();
        auto ft = recv_comm.finish_communication();
        //for(auto& e : ft)
        //{
        //    std::cout<<"Received answer on rank"<<comm_.rank()<<": \n";
        //    for(auto& d: e->has_data()) std::cout<<d<<"  ";
        //    std::cout<<std::endl;
        //}
        if (recv_comm.done()) return true;
        return false;
    }

    template<class QueryType>
    bool done_sending(QueryType& _q)
    {
        using send_task_t = typename QueryType::send_task_t;
        using recv_task_t = typename QueryType::recv_task_t;
        auto& send_comm =
            task_manager_->template send_communicator<send_task_t>();
        auto& recv_comm =
            task_manager_->template recv_communicator<recv_task_t>();

        send_comm.start_communication();
        auto finished_tasks = send_comm.finish_communication();
        for (auto& e : finished_tasks)
        {
            if (_q.recvDataPtr(server_rank_))
            {
                auto answer =
                    recv_comm.post_answer(e, _q.recvDataPtr(server_rank_));
            }
        }
        if (send_comm.done()) return true;
        return false;
    }

    void disconnect(int _server_rank)
    {
        const auto tag = tag_gen().get<tags::connection>(comm_.rank());
        comm_.send(_server_rank, tag, false);
    }
    void disconnect() { disconnect(server_rank_); }

    void connect()
    {
        const auto tag = tag_gen().get<tags::connection>(comm_.rank());
        comm_.send(server_rank_, tag, true);
    }

    const int& server() const noexcept { return server_rank_; }
    int&       server() noexcept { return server_rank_; }

  protected:
    int server_rank_;
};

} //namespace  sr_mpi
} // namespace iblgf
#endif
