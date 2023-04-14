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

#ifndef INCLUDED_HALOCOMMUNICATOR_HPP
#define INCLUDED_HALOCOMMUNICATOR_HPP

#include "tags.hpp"
#include "task_buffer.hpp"

#include <iblgf/utilities/crtp.hpp>

namespace iblgf
{
namespace sr_mpi
{
/** @brief Communicate field buffers/halos for a given fields and a task type
 */
template<class TaskType, class Field, class Domain>
class HaloCommunicator
{
  public: //Ctor
    HaloCommunicator(const HaloCommunicator&) = default;
    HaloCommunicator(HaloCommunicator&&) = default;
    HaloCommunicator& operator=(const HaloCommunicator&) & = default;
    HaloCommunicator& operator=(HaloCommunicator&&) & = default;
    ~HaloCommunicator() = default;
    HaloCommunicator()
    {
        boost::mpi::communicator world;
        inter_send_interface.resize(world.size());
        inter_recv_interface.resize(world.size());
        send_fields_.resize(world.size());
        recv_fields_.resize(world.size());
        send_tasks_.resize(world.size());
        recv_tasks_.resize(world.size());
    }

    using field_t = Field;
    using view_type = typename field_t::view_type;
    using octant_t = typename Domain::octant_t;

  private:
    struct interface
    {
        octant_t* src;       //sending octant
        octant_t* dest;      //recving octant
        view_type view;      //field view of overlap
        int       field_idx; //field view of overlap
    };

  public: //members
    void extrapolate_to_buffer()
    {
        //
    }

    /** @brief Compute and store communication task for halo echange of fields*/
    void compute_tasks(
        Domain* _domain, int _level, bool axis_neighbors_only = false) noexcept
    {
        for (std::size_t j = 0; j < Field::nFields(); ++j)
        {
            for (auto it = _domain->begin(_level); it != _domain->end(_level);
                 ++it)
            {
                if (!it->locally_owned()) continue;
                if (!it->has_data() || !it->data().is_allocated()) continue;

                for (std::size_t i = 0; i < it->num_neighbors(); ++i)
                {
                    auto it2 = it->neighbor(i);
                    if (!it2) continue;
                    if (!it2->has_data()) continue;

                    auto& field = it->data_r(Field::tag(),j);
                    auto& field2 = it2->data_r(Field::tag(),j);

                    //send view
                    if (auto overlap_opt = field.send_view(field2))
                    {
                        interface sif;
                        sif.src = *it; //it.ptr();
                        sif.dest = it2;
                        sif.field_idx = j;
                        sif.view = std::move(overlap_opt.value());

                        if (axis_neighbors_only &&
                            (static_cast<int>(sif.view.size()) <=
                                field.extent()[0]))
                            continue;

                        if (it2->locally_owned())
                            intra_send_interface.emplace_back(std::move(sif));
                        else
                            inter_send_interface[it2->rank()].emplace_back(
                                std::move(sif));
                    }
                    //recv view
                    if (auto overlap_opt = field.recv_view(field2))
                    {
                        interface sif;
                        sif.src = it2;
                        sif.dest = *it; //it.ptr();
                        sif.field_idx = j;
                        sif.view = std::move(overlap_opt.value());

                        if (axis_neighbors_only &&
                            (static_cast<int>(sif.view.size()) <=
                                field.extent()[0]))
                            continue;

                        if (!it2->locally_owned())
                        {
                            inter_recv_interface[it2->rank()].emplace_back(
                                std::move(sif));
                        }
                    }
                }
            }
        }

        //sort & combine tasks
        for (int rank_other = 0;
             rank_other < static_cast<int>(inter_send_interface.size());
             ++rank_other)
        {
            auto compare = [&](const auto& c0, const auto& c1) {
                if (c0.dest->key().id() == c1.dest->key().id())
                {
                    if (c0.src->key().id() == c1.src->key().id())
                    { return c0.field_idx < c1.field_idx; }
                    else
                    {
                        return c0.src->key().id() < c1.src->key().id();
                    }
                }
                else
                {
                    return c0.dest->key().id() < c1.dest->key().id();
                }
                return (c0.dest->key().id() == c1.dest->key().id())
                           ? (c0.src->key().id() < c1.src->key().id())
                           : (c0.dest->key().id() < c1.dest->key().id());
            };

            //send:
            {
                auto& tasks = inter_send_interface[rank_other];
                std::sort(tasks.begin(), tasks.end(), compare);
                if (tasks.empty())
                {
                    send_tasks_[rank_other] = nullptr;
                    continue;
                }
                boost::mpi::communicator c;
                int                      idx = tasks[0].src->idx();
                idx = c.rank();
                auto task_ptr = std::make_shared<TaskType>(idx);
                task_ptr->attach_data(&send_fields_[rank_other]);
                task_ptr->rank_other() = rank_other;
                task_ptr->requires_confirmation() = false;
                send_tasks_[rank_other] = task_ptr;
            }

            //recv:
            {
                auto& tasks = inter_recv_interface[rank_other];
                std::sort(tasks.begin(), tasks.end(), compare);

                if (tasks.empty())
                {
                    recv_tasks_[rank_other] = nullptr;
                    continue;
                }
                int idx = tasks[0].src->idx();
                idx = rank_other;
                auto task_ptr = std::make_shared<TaskType>(idx);
                task_ptr->attach_data(&recv_fields_[rank_other]);
                task_ptr->rank_other() = rank_other;
                task_ptr->requires_confirmation() = false;
                recv_tasks_[rank_other] = task_ptr;
            }
        }
    }

    /** @brief Fill the buffers and copy locally_owned fields into halo buffer*/
    void pack_messages()
    {
        //Copy locally owned fields to buffers:
#ifdef USE_OMP
        #pragma omp parallel for
        for (int i = 0; i < intra_send_interface.size(); i++)
        {
            auto& sf = intra_send_interface[i];
            auto& dfield =
                sf.dest->data_r(Field::tag(),sf.field_idx);
            auto dest_view = dfield.view(sf.view);
            dest_view.assign_toView(sf.view);
        }
#else
        for (auto& sf : intra_send_interface)
        {
            auto& dfield =
                sf.dest->data_r(Field::tag(),sf.field_idx);
            auto dest_view = dfield.view(sf.view);
            dest_view.assign_toView(sf.view);
        }
#endif

        //Copy data into buffer:
        for (int rank_other = 0;
             rank_other < static_cast<int>(inter_send_interface.size());
             ++rank_other)
        {
            auto& tasks = inter_send_interface[rank_other];
            auto& rtasks = inter_recv_interface[rank_other];

            if (tasks.empty())
            {
                send_tasks_[rank_other] = nullptr;
                continue;
            }
            if (rtasks.empty())
            {
                recv_tasks_[rank_other] = nullptr;
                continue;
            }

            std::size_t size = 0;
            for (auto& interfc : inter_send_interface[rank_other])
            {
                for (auto it = interfc.view.begin(); it != interfc.view.end();
                     ++it)
                { ++size; }
            }
            if (size != send_fields_[rank_other].size())
                send_fields_[rank_other].resize(size);
            int count = 0;
            for (auto& interfc : inter_send_interface[rank_other])
            {
                interfc.view.iterate([&](const auto& val) {
                    send_fields_[rank_other][count++] = val;
                });
            }
            send_tasks_[rank_other]->attach_data(&send_fields_[rank_other]);
            recv_tasks_[rank_other]->attach_data(&recv_fields_[rank_other]);
        }
    }

    /** @brief Assign received buffers into field views*/
    void unpack_messages()
    {
        //Copy received messages from buffer into views
        for (int rank_other = 0;
             rank_other < static_cast<int>(inter_recv_interface.size());
             ++rank_other)
        {
            int count = 0;
            for (auto& interfc : inter_recv_interface[rank_other])
            {
                auto& recv_view = interfc.view;
                recv_view.iterate([&](auto& val) {
                    val = recv_fields_[rank_other][count++];
                });
            }
        }
    }





    /** @brief Fill the buffers and copy locally_owned fields of an speciic index into halo buffer*/
    void pack_messages(int _field_idx)
    {
        //Copy locally owned fields to buffers:
        for (auto& sf : intra_send_interface)
        {
            if ((sf.field_idx != _field_idx)) continue;
            auto& dfield =
                sf.dest->data_r(Field::tag(),sf.field_idx);
            auto dest_view = dfield.view(sf.view);
            dest_view.assign_toView(sf.view);
        }

        //Copy data into buffer:
        for (int rank_other = 0;
             rank_other < static_cast<int>(inter_send_interface.size());
             ++rank_other)
        {
            auto& tasks = inter_send_interface[rank_other];
            auto& rtasks = inter_recv_interface[rank_other];

            if (tasks.empty())
            {
                send_tasks_[rank_other] = nullptr;
                continue;
            }
            if (rtasks.empty())
            {
                recv_tasks_[rank_other] = nullptr;
                continue;
            }

            std::size_t size = 0;
            for (auto& interfc : inter_send_interface[rank_other])
            {
                if ((interfc.field_idx != _field_idx)) continue;
                for (auto it = interfc.view.begin(); it != interfc.view.end();
                     ++it)
                { ++size; }
            }
            if (size != send_fields_[rank_other].size())
                send_fields_[rank_other].resize(size);
            int count = 0;
            for (auto& interfc : inter_send_interface[rank_other])
            {
                if ((interfc.field_idx != _field_idx)) continue;
                interfc.view.iterate([&](const auto& val) {
                    send_fields_[rank_other][count++] = val;
                });
            }
            send_tasks_[rank_other]->attach_data(&send_fields_[rank_other]);
            recv_tasks_[rank_other]->attach_data(&recv_fields_[rank_other]);
        }
    }

    /** @brief Assign received buffers into field views*/
    void unpack_messages(int _field_idx)
    {
        //Copy received messages from buffer into views
        for (int rank_other = 0;
             rank_other < static_cast<int>(inter_recv_interface.size());
             ++rank_other)
        {
            int count = 0;
            for (auto& interfc : inter_recv_interface[rank_other])
            {
                if ((interfc.field_idx != _field_idx)) continue;
                auto& recv_view = interfc.view;
                recv_view.iterate([&](auto& val) {
                    val = recv_fields_[rank_other][count++];
                });
            }
        }
    }

    auto&       send_tasks() noexcept { return send_tasks_; }
    const auto& send_tasks() const noexcept { return send_tasks_; }

    auto&       recv_tasks() noexcept { return recv_tasks_; }
    const auto& recv_tasks() const noexcept { return recv_tasks_; }
    void        clear()
    {
        inter_send_interface.clear();
        inter_recv_interface.clear();
        send_fields_.clear();
        recv_fields_.clear();
        send_tasks_.clear();
        recv_tasks_.clear();

        boost::mpi::communicator world;
        inter_send_interface.resize(world.size());
        inter_recv_interface.resize(world.size());
        send_fields_.resize(world.size());
        recv_fields_.resize(world.size());
        send_tasks_.resize(world.size());
        recv_tasks_.resize(world.size());
    }

  private:
    //send/recv interfaces per processor
    std::vector<std::vector<interface>> inter_send_interface;
    std::vector<std::vector<interface>> inter_recv_interface;

    std::vector<std::shared_ptr<TaskType>> send_tasks_;
    std::vector<std::shared_ptr<TaskType>> recv_tasks_;

    std::vector<std::vector<float_type>> send_fields_;
    std::vector<std::vector<float_type>> recv_fields_;

    //intra processor send/recv interfaces
    std::vector<interface> intra_send_interface;
    std::vector<interface> intra_recv_interface;
};





template<class TaskType, class Field, class Domain>
class HaloCommunicator_idx
{
    //halo communicator for single index
  public: //Ctor
    HaloCommunicator_idx(const HaloCommunicator_idx&) = default;
    HaloCommunicator_idx(HaloCommunicator_idx&&) = default;
    HaloCommunicator_idx& operator=(const HaloCommunicator_idx&) & = default;
    HaloCommunicator_idx& operator=(HaloCommunicator_idx&&) & = default;
    ~HaloCommunicator_idx() = default;
    HaloCommunicator_idx(int _field_idx)
    {
        boost::mpi::communicator world;
        inter_send_interface.resize(world.size());
        inter_recv_interface.resize(world.size());
        send_fields_.resize(world.size());
        recv_fields_.resize(world.size());
        send_tasks_.resize(world.size());
        recv_tasks_.resize(world.size());
        field_idx = _field_idx;
    }

    using field_t = Field;
    using view_type = typename field_t::view_type;
    using octant_t = typename Domain::octant_t;

  private:
    struct interface
    {
        octant_t* src;       //sending octant
        octant_t* dest;      //recving octant
        view_type view;      //field view of overlap
        int       field_idx; //field view of overlap
    };

  public: //members
    void extrapolate_to_buffer()
    {
        //
    }

    /** @brief Compute and store communication task for halo echange of fields*/
    void compute_tasks(
        Domain* _domain, int _level, bool axis_neighbors_only = false) noexcept
    {
        
        for (auto it = _domain->begin(_level); it != _domain->end(_level);
                ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->has_data() || !it->data().is_allocated()) continue;

            for (std::size_t i = 0; i < it->num_neighbors(); ++i)
            {
                auto it2 = it->neighbor(i);
                if (!it2) continue;
                if (!it2->has_data()) continue;

                auto& field = it->data_r(Field::tag(),field_idx);
                auto& field2 = it2->data_r(Field::tag(),field_idx);

                //send view
                if (auto overlap_opt = field.send_view(field2))
                {
                    interface sif;
                    sif.src = *it; //it.ptr();
                    sif.dest = it2;
                    sif.field_idx = field_idx;
                    sif.view = std::move(overlap_opt.value());

                    if (axis_neighbors_only &&
                        (static_cast<int>(sif.view.size()) <=
                            field.extent()[0]))
                        continue;

                    if (it2->locally_owned())
                        intra_send_interface.emplace_back(std::move(sif));
                    else
                        inter_send_interface[it2->rank()].emplace_back(
                            std::move(sif));
                }
                //recv view
                if (auto overlap_opt = field.recv_view(field2))
                {
                    interface sif;
                    sif.src = it2;
                    sif.dest = *it; //it.ptr();
                    sif.field_idx = field_idx;
                    sif.view = std::move(overlap_opt.value());

                    if (axis_neighbors_only &&
                        (static_cast<int>(sif.view.size()) <=
                            field.extent()[0]))
                        continue;

                    if (!it2->locally_owned())
                    {
                        inter_recv_interface[it2->rank()].emplace_back(
                            std::move(sif));
                    }
                }
            }
        }
        

        //sort & combine tasks
        for (int rank_other = 0;
             rank_other < static_cast<int>(inter_send_interface.size());
             ++rank_other)
        {
            auto compare = [&](const auto& c0, const auto& c1) {
                if (c0.dest->key().id() == c1.dest->key().id())
                {
                    if (c0.src->key().id() == c1.src->key().id())
                    { return c0.field_idx < c1.field_idx; }
                    else
                    {
                        return c0.src->key().id() < c1.src->key().id();
                    }
                }
                else
                {
                    return c0.dest->key().id() < c1.dest->key().id();
                }
                return (c0.dest->key().id() == c1.dest->key().id())
                           ? (c0.src->key().id() < c1.src->key().id())
                           : (c0.dest->key().id() < c1.dest->key().id());
            };

            //send:
            {
                auto& tasks = inter_send_interface[rank_other];
                std::sort(tasks.begin(), tasks.end(), compare);
                if (tasks.empty())
                {
                    send_tasks_[rank_other] = nullptr;
                    continue;
                }
                boost::mpi::communicator c;
                int                      idx = tasks[0].src->idx();
                idx = c.rank();
                auto task_ptr = std::make_shared<TaskType>(idx);
                task_ptr->attach_data(&send_fields_[rank_other]);
                task_ptr->rank_other() = rank_other;
                task_ptr->requires_confirmation() = false;
                send_tasks_[rank_other] = task_ptr;
            }

            //recv:
            {
                auto& tasks = inter_recv_interface[rank_other];
                std::sort(tasks.begin(), tasks.end(), compare);

                if (tasks.empty())
                {
                    recv_tasks_[rank_other] = nullptr;
                    continue;
                }
                int idx = tasks[0].src->idx();
                idx = rank_other;
                auto task_ptr = std::make_shared<TaskType>(idx);
                task_ptr->attach_data(&recv_fields_[rank_other]);
                task_ptr->rank_other() = rank_other;
                task_ptr->requires_confirmation() = false;
                recv_tasks_[rank_other] = task_ptr;
            }
        }
    }

    /** @brief Fill the buffers and copy locally_owned fields into halo buffer*/
    void pack_messages()
    {
        //Copy locally owned fields to buffers:
        for (auto& sf : intra_send_interface)
        {
            auto& dfield =
                sf.dest->data_r(Field::tag(),sf.field_idx);
            auto dest_view = dfield.view(sf.view);
            dest_view.assign_toView(sf.view);
        }

        //Copy data into buffer:
        for (int rank_other = 0;
             rank_other < static_cast<int>(inter_send_interface.size());
             ++rank_other)
        {
            auto& tasks = inter_send_interface[rank_other];
            auto& rtasks = inter_recv_interface[rank_other];

            if (tasks.empty())
            {
                send_tasks_[rank_other] = nullptr;
                continue;
            }
            if (rtasks.empty())
            {
                recv_tasks_[rank_other] = nullptr;
                continue;
            }

            std::size_t size = 0;
            for (auto& interfc : inter_send_interface[rank_other])
            {
                for (auto it = interfc.view.begin(); it != interfc.view.end();
                     ++it)
                { ++size; }
            }
            if (size != send_fields_[rank_other].size())
                send_fields_[rank_other].resize(size);
            int count = 0;
            for (auto& interfc : inter_send_interface[rank_other])
            {
                interfc.view.iterate([&](const auto& val) {
                    send_fields_[rank_other][count++] = val;
                });
            }
            send_tasks_[rank_other]->attach_data(&send_fields_[rank_other]);
            recv_tasks_[rank_other]->attach_data(&recv_fields_[rank_other]);
        }
    }

    /** @brief Assign received buffers into field views*/
    void unpack_messages()
    {
        //Copy received messages from buffer into views
        for (int rank_other = 0;
             rank_other < static_cast<int>(inter_recv_interface.size());
             ++rank_other)
        {
            int count = 0;
            for (auto& interfc : inter_recv_interface[rank_other])
            {
                auto& recv_view = interfc.view;
                recv_view.iterate([&](auto& val) {
                    val = recv_fields_[rank_other][count++];
                });
            }
        }
    }





    /** @brief Fill the buffers and copy locally_owned fields of an speciic index into halo buffer*/
    void pack_messages(int _field_idx)
    {
        //Copy locally owned fields to buffers:
        for (auto& sf : intra_send_interface)
        {
            if ((sf.field_idx != _field_idx)) continue;
            auto& dfield =
                sf.dest->data_r(Field::tag(),sf.field_idx);
            auto dest_view = dfield.view(sf.view);
            dest_view.assign_toView(sf.view);
        }

        //Copy data into buffer:
        for (int rank_other = 0;
             rank_other < static_cast<int>(inter_send_interface.size());
             ++rank_other)
        {
            auto& tasks = inter_send_interface[rank_other];
            auto& rtasks = inter_recv_interface[rank_other];

            if (tasks.empty())
            {
                send_tasks_[rank_other] = nullptr;
                continue;
            }
            if (rtasks.empty())
            {
                recv_tasks_[rank_other] = nullptr;
                continue;
            }

            std::size_t size = 0;
            for (auto& interfc : inter_send_interface[rank_other])
            {
                if ((interfc.field_idx != _field_idx)) continue;
                for (auto it = interfc.view.begin(); it != interfc.view.end();
                     ++it)
                { ++size; }
            }
            if (size != send_fields_[rank_other].size())
                send_fields_[rank_other].resize(size);
            int count = 0;
            for (auto& interfc : inter_send_interface[rank_other])
            {
                if ((interfc.field_idx != _field_idx)) continue;
                interfc.view.iterate([&](const auto& val) {
                    send_fields_[rank_other][count++] = val;
                });
            }
            send_tasks_[rank_other]->attach_data(&send_fields_[rank_other]);
            recv_tasks_[rank_other]->attach_data(&recv_fields_[rank_other]);
        }
    }

    /** @brief Assign received buffers into field views*/
    void unpack_messages(int _field_idx)
    {
        //Copy received messages from buffer into views
        for (int rank_other = 0;
             rank_other < static_cast<int>(inter_recv_interface.size());
             ++rank_other)
        {
            int count = 0;
            for (auto& interfc : inter_recv_interface[rank_other])
            {
                if ((interfc.field_idx != _field_idx)) continue;
                auto& recv_view = interfc.view;
                recv_view.iterate([&](auto& val) {
                    val = recv_fields_[rank_other][count++];
                });
            }
        }
    }

    auto&       send_tasks() noexcept { return send_tasks_; }
    const auto& send_tasks() const noexcept { return send_tasks_; }

    auto&       recv_tasks() noexcept { return recv_tasks_; }
    const auto& recv_tasks() const noexcept { return recv_tasks_; }
    void        clear()
    {
        inter_send_interface.clear();
        inter_recv_interface.clear();
        send_fields_.clear();
        recv_fields_.clear();
        send_tasks_.clear();
        recv_tasks_.clear();

        boost::mpi::communicator world;
        inter_send_interface.resize(world.size());
        inter_recv_interface.resize(world.size());
        send_fields_.resize(world.size());
        recv_fields_.resize(world.size());
        send_tasks_.resize(world.size());
        recv_tasks_.resize(world.size());
    }

  private:
    //send/recv interfaces per processor
    std::vector<std::vector<interface>> inter_send_interface;
    std::vector<std::vector<interface>> inter_recv_interface;

    std::vector<std::shared_ptr<TaskType>> send_tasks_;
    std::vector<std::shared_ptr<TaskType>> recv_tasks_;

    std::vector<std::vector<float_type>> send_fields_;
    std::vector<std::vector<float_type>> recv_fields_;

    //intra processor send/recv interfaces
    std::vector<interface> intra_send_interface;
    std::vector<interface> intra_recv_interface;

    int field_idx;
};

} // namespace sr_mpi
} // namespace iblgf
#endif
