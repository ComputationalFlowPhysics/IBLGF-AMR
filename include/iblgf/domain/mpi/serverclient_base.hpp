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

#ifndef INCLUDED_SERVERCLIENTBASE_HPP
#define INCLUDED_SERVERCLIENTBASE_HPP

#include <set>
#include <vector>
#include <unordered_set>
#include <memory>
#include <list>
#include <boost/serialization/vector.hpp>

#include "task_manager.hpp"

namespace iblgf
{
namespace sr_mpi
{
template<class Traits>
class ServerClientBase
{
  public:
    using task_manager_t = typename Traits::task_manager_t;

  public: // ctors
    ServerClientBase(const ServerClientBase&) = default;
    ServerClientBase(ServerClientBase&&) = default;
    ServerClientBase& operator=(const ServerClientBase&) & = default;
    ServerClientBase& operator=(ServerClientBase&&) & = default;
    ~ServerClientBase() = default;
    ServerClientBase()
    : task_manager_(std::make_shared<task_manager_t>())
    {
    }

    ServerClientBase(std::shared_ptr<task_manager_t> _task_manager)
    : task_manager_(_task_manager)
    {
    }

  public:
    const auto& task_manager() const noexcept { return task_manager_; }
    auto&       task_manager() noexcept { return task_manager_; }

    const auto& task_manager_vec() const noexcept { return task_manager_vec; }
    auto&       task_manager_vec() noexcept { return task_manager_vec; }
#ifdef USE_OMP
    void resizing_manager_vec(int n) {
      for (int i = 0; i < n; i++) {
        task_manager_vec_.emplace_back(new task_manager_t());
      }
    }
#endif

  protected:
    boost::mpi::communicator        comm_;
    std::shared_ptr<task_manager_t> task_manager_ = nullptr;

    std::vector<std::shared_ptr<task_manager_t>> task_manager_vec_;

};

} // namespace sr_mpi
} // namespace iblgf

#endif
