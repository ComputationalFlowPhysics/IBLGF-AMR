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

#ifndef INCLUDED_QUERY_REGISTRY_HPP
#define INCLUDED_QUERY_REGISTRY_HPP

#include "tags.hpp"
#include "task_buffer.hpp"

#include <iblgf/utilities/crtp.hpp>

namespace iblgf
{
namespace sr_mpi
{
template<class SendTask, class RecvTask = SendTask>
class QueryRegistry
{
  public:
    using send_task_t = SendTask;
    using recv_task_t = RecvTask;
    using send_data_t = typename SendTask::data_type;
    using recv_data_t = typename RecvTask::data_type;

    using sendMap_functor_t = std::function<send_data_t*(int)>;
    using recvMap_functor_t = std::function<recv_data_t*(int)>;

    using complete_functor_t =
        std::function<void(recv_task_t* _t, send_data_t* _outDat)>;

  public: //Ctor
    QueryRegistry(const QueryRegistry&) = default;
    QueryRegistry(QueryRegistry&&) = default;
    QueryRegistry& operator=(const QueryRegistry&) & = default;
    QueryRegistry& operator=(QueryRegistry&&) & = default;
    ~QueryRegistry() = default;
    QueryRegistry() = default;

  public: //Registry
    void register_sendMap(sendMap_functor_t _f) noexcept { sendMap_ = _f; }

    void register_recvMap(recvMap_functor_t _f) noexcept { recvMap_ = _f; }

    void register_completeFunc(complete_functor_t _f) { complete_ = _f; }

    virtual send_data_t* sendDataPtr(int _rank = 0) const noexcept
    {
        return sendMap_ ? sendMap_(_rank) : nullptr;
    }
    virtual recv_data_t* recvDataPtr(int _rank = 0) const noexcept
    {
        return recvMap_ ? recvMap_(_rank) : nullptr;
    }

    virtual send_data_t& sendData(int _rank = 0) const noexcept
    {
        return *sendMap_(_rank);
    }
    virtual recv_data_t& recvData(int _rank = 0) const noexcept
    {
        return *recvMap_(_rank);
    }

    auto complete(std::shared_ptr<recv_task_t> _t) const noexcept
    {
        if (complete_)
            return complete_(_t.get(), sendDataPtr(_t->rank_other()));
    }

  private:
    sendMap_functor_t  sendMap_;
    recvMap_functor_t  recvMap_;
    complete_functor_t complete_;
};

template<class SendTask, class RecvTask>
class InlineQueryRegistry : public QueryRegistry<SendTask, RecvTask>
{
  public:
    using send_task_t = SendTask;
    using recv_task_t = RecvTask;
    using send_data_t = typename SendTask::data_type;
    using recv_data_t = typename RecvTask::data_type;

  public: //Ctor
    InlineQueryRegistry(const InlineQueryRegistry&) = default;
    InlineQueryRegistry(InlineQueryRegistry&&) = default;
    InlineQueryRegistry& operator=(const InlineQueryRegistry&) & = default;
    InlineQueryRegistry& operator=(InlineQueryRegistry&&) & = default;
    ~InlineQueryRegistry() = default;
    InlineQueryRegistry() = default;

    InlineQueryRegistry(int _size)
    : send_data_(_size)
    , recv_data_(_size)
    {
    }
    InlineQueryRegistry(int _send_size, int _recv_size)
    : send_data_(_send_size)
    , recv_data_(_recv_size)
    {
    }

  public:
    send_data_t* sendDataPtr(int _rank = 0) const noexcept override
    {
        return &send_data_[_rank];
    }

    recv_data_t* recvDataPtr(int _rank = 0) const noexcept override
    {
        return &recv_data_[_rank];
    }

    send_data_t& sendData(int _rank = 0) const noexcept override
    {
        return send_data_[_rank];
    }

    recv_data_t& recvData(int _rank = 0) const noexcept override
    {
        return recv_data_[_rank];
    }

  private:
    mutable std::vector<send_data_t> send_data_;
    mutable std::vector<recv_data_t> recv_data_;
};

} // namespace sr_mpi
} // namespace iblgf
#endif
