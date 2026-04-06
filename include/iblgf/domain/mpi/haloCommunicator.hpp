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
#ifdef IBLGF_COMPILE_CUDA
#include <cuda_runtime.h>
#include <iblgf/utilities/cuda_pack.hpp>
#endif

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
#ifdef IBLGF_COMPILE_CUDA
        send_fields_device_.resize(world.size(), nullptr);
        recv_fields_device_.resize(world.size(), nullptr);
        send_fields_device_size_.resize(world.size(), 0);
        recv_fields_device_size_.resize(world.size(), 0);
#endif
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
#ifdef IBLGF_COMPILE_CUDA
            if (!inter_send_interface[rank_other].empty())
            {
                auto& first_field =
                    inter_send_interface[rank_other][0].view.field();
                const bool use_device = first_field &&
                    first_field->device_valid();
                if (use_device)
                {
                    ensure_device_buffer_(send_fields_device_[rank_other],
                        send_fields_device_size_[rank_other], size);
                    std::size_t offset = 0;
                    for (auto& interfc : inter_send_interface[rank_other])
                    {
                        auto& field = *interfc.view.field();
                        auto desc = make_desc_(field, interfc.view);
                        const std::size_t n = interfc.view.size();
                        iblgf::gpu::pack_view_device_to_device(
                            field.device_ptr(), desc,
                            send_fields_device_[rank_other] + offset, n);
                        offset += n;
                    }
                    cudaMemcpy(send_fields_[rank_other].data(),
                        send_fields_device_[rank_other],
                        size * sizeof(float_type), cudaMemcpyDeviceToHost);
                }
                else
                {
                    int count = 0;
                    for (auto& interfc : inter_send_interface[rank_other])
                    {
                        interfc.view.iterate([&](const auto& val) {
                            send_fields_[rank_other][count++] = val;
                        });
                    }
                }
            }
#else
            int count = 0;
            for (auto& interfc : inter_send_interface[rank_other])
            {
                interfc.view.iterate([&](const auto& val) {
                    send_fields_[rank_other][count++] = val;
                });
            }
#endif
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
#ifdef IBLGF_COMPILE_CUDA
            if (!inter_recv_interface[rank_other].empty())
            {
                auto& first_field =
                    inter_recv_interface[rank_other][0].view.field();
                const bool use_device = first_field &&
                    first_field->device_valid();
                if (use_device)
                {
                    const std::size_t size = recv_fields_[rank_other].size();
                    ensure_device_buffer_(recv_fields_device_[rank_other],
                        recv_fields_device_size_[rank_other], size);
                    cudaMemcpy(recv_fields_device_[rank_other],
                        recv_fields_[rank_other].data(),
                        size * sizeof(float_type), cudaMemcpyHostToDevice);
                    std::size_t offset = 0;
                    for (auto& interfc : inter_recv_interface[rank_other])
                    {
                        auto& field = *interfc.view.field();
                        auto desc = make_desc_(field, interfc.view);
                        const std::size_t n = interfc.view.size();
                        iblgf::gpu::unpack_view_device_to_device(
                            recv_fields_device_[rank_other] + offset, desc,
                            field.device_ptr(), n);
                        field.mark_device_valid();
                        offset += n;
                    }
                }
                else
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
#else
            int count = 0;
            for (auto& interfc : inter_recv_interface[rank_other])
            {
                auto& recv_view = interfc.view;
                recv_view.iterate([&](auto& val) {
                    val = recv_fields_[rank_other][count++];
                });
            }
#endif
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
#ifdef IBLGF_COMPILE_CUDA
            if (!inter_send_interface[rank_other].empty())
            {
                auto& first_field =
                    inter_send_interface[rank_other][0].view.field();
                const bool use_device = first_field &&
                    first_field->device_valid();
                if (use_device)
                {
                    ensure_device_buffer_(send_fields_device_[rank_other],
                        send_fields_device_size_[rank_other], size);
                    std::size_t offset = 0;
                    for (auto& interfc : inter_send_interface[rank_other])
                    {
                        if ((interfc.field_idx != _field_idx)) continue;
                        auto& field = *interfc.view.field();
                        auto desc = make_desc_(field, interfc.view);
                        const std::size_t n = interfc.view.size();
                        iblgf::gpu::pack_view_device_to_device(
                            field.device_ptr(), desc,
                            send_fields_device_[rank_other] + offset, n);
                        offset += n;
                    }
                    cudaMemcpy(send_fields_[rank_other].data(),
                        send_fields_device_[rank_other],
                        size * sizeof(float_type), cudaMemcpyDeviceToHost);
                }
                else
                {
                    int count = 0;
                    for (auto& interfc : inter_send_interface[rank_other])
                    {
                        if ((interfc.field_idx != _field_idx)) continue;
                        interfc.view.iterate([&](const auto& val) {
                            send_fields_[rank_other][count++] = val;
                        });
                    }
                }
            }
#else
            int count = 0;
            for (auto& interfc : inter_send_interface[rank_other])
            {
                if ((interfc.field_idx != _field_idx)) continue;
                interfc.view.iterate([&](const auto& val) {
                    send_fields_[rank_other][count++] = val;
                });
            }
#endif
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
#ifdef IBLGF_COMPILE_CUDA
            if (!inter_recv_interface[rank_other].empty())
            {
                auto& first_field =
                    inter_recv_interface[rank_other][0].view.field();
                const bool use_device = first_field &&
                    first_field->device_valid();
                if (use_device)
                {
                    const std::size_t size = recv_fields_[rank_other].size();
                    ensure_device_buffer_(recv_fields_device_[rank_other],
                        recv_fields_device_size_[rank_other], size);
                    cudaMemcpy(recv_fields_device_[rank_other],
                        recv_fields_[rank_other].data(),
                        size * sizeof(float_type), cudaMemcpyHostToDevice);
                    std::size_t offset = 0;
                    for (auto& interfc : inter_recv_interface[rank_other])
                    {
                        if ((interfc.field_idx != _field_idx)) continue;
                        auto& field = *interfc.view.field();
                        auto desc = make_desc_(field, interfc.view);
                        const std::size_t n = interfc.view.size();
                        iblgf::gpu::unpack_view_device_to_device(
                            recv_fields_device_[rank_other] + offset, desc,
                            field.device_ptr(), n);
                        field.mark_device_valid();
                        offset += n;
                    }
                }
                else
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
#else
            int count = 0;
            for (auto& interfc : inter_recv_interface[rank_other])
            {
                if ((interfc.field_idx != _field_idx)) continue;
                auto& recv_view = interfc.view;
                recv_view.iterate([&](auto& val) {
                    val = recv_fields_[rank_other][count++];
                });
            }
#endif
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
#ifdef IBLGF_COMPILE_CUDA
        for (auto& ptr : send_fields_device_)
        {
            if (ptr) cudaFree(ptr);
            ptr = nullptr;
        }
        for (auto& ptr : recv_fields_device_)
        {
            if (ptr) cudaFree(ptr);
            ptr = nullptr;
        }
        send_fields_device_.clear();
        recv_fields_device_.clear();
        send_fields_device_size_.clear();
        recv_fields_device_size_.clear();
#endif

        boost::mpi::communicator world;
        inter_send_interface.resize(world.size());
        inter_recv_interface.resize(world.size());
        send_fields_.resize(world.size());
        recv_fields_.resize(world.size());
        send_tasks_.resize(world.size());
        recv_tasks_.resize(world.size());
#ifdef IBLGF_COMPILE_CUDA
        send_fields_device_.resize(world.size(), nullptr);
        recv_fields_device_.resize(world.size(), nullptr);
        send_fields_device_size_.resize(world.size(), 0);
        recv_fields_device_size_.resize(world.size(), 0);
#endif
    }

  private:
    //send/recv interfaces per processor
    std::vector<std::vector<interface>> inter_send_interface;
    std::vector<std::vector<interface>> inter_recv_interface;

    std::vector<std::shared_ptr<TaskType>> send_tasks_;
    std::vector<std::shared_ptr<TaskType>> recv_tasks_;

    std::vector<std::vector<float_type>> send_fields_;
    std::vector<std::vector<float_type>> recv_fields_;

#ifdef IBLGF_COMPILE_CUDA
    std::vector<float_type*> send_fields_device_;
    std::vector<float_type*> recv_fields_device_;
    std::vector<std::size_t> send_fields_device_size_;
    std::vector<std::size_t> recv_fields_device_size_;

    void ensure_device_buffer_(float_type*& ptr, std::size_t& current_size,
        std::size_t needed)
    {
        if (current_size == needed && ptr) return;
        if (ptr)
        {
            cudaFree(ptr);
            ptr = nullptr;
        }
        current_size = needed;
        if (needed > 0)
        {
            cudaMalloc(&ptr, needed * sizeof(float_type));
        }
    }

    template<class DataFieldType, class ViewType>
    iblgf::gpu::block_view_desc make_desc_(const DataFieldType& field,
        const ViewType& view) const
    {
        iblgf::gpu::block_view_desc desc{};
        const auto& f_base = field.real_block().base();
        const auto& f_ext = field.real_block().extent();
        const auto& v_base = view.base();
        const auto& v_ext = view.extent();
        const auto& v_stride = view.stride();

        constexpr std::size_t Dim = DataFieldType::dimension();
        desc.field_base[0] = f_base[0];
        desc.field_extent[0] = f_ext[0];
        desc.view_base[0] = v_base[0];
        desc.view_extent[0] = v_ext[0];
        desc.view_stride[0] = v_stride[0];

        desc.field_base[1] = f_base[1];
        desc.field_extent[1] = f_ext[1];
        desc.view_base[1] = v_base[1];
        desc.view_extent[1] = v_ext[1];
        desc.view_stride[1] = v_stride[1];

        if constexpr (Dim == 3)
        {
            desc.field_base[2] = f_base[2];
            desc.field_extent[2] = f_ext[2];
            desc.view_base[2] = v_base[2];
            desc.view_extent[2] = v_ext[2];
            desc.view_stride[2] = v_stride[2];
        }
        else
        {
            desc.field_base[2] = 0;
            desc.field_extent[2] = 1;
            desc.view_base[2] = 0;
            desc.view_extent[2] = 1;
            desc.view_stride[2] = 1;
        }
        return desc;
    }
#endif

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
#ifdef IBLGF_COMPILE_CUDA
        send_fields_device_.resize(world.size(), nullptr);
        recv_fields_device_.resize(world.size(), nullptr);
        send_fields_device_size_.resize(world.size(), 0);
        recv_fields_device_size_.resize(world.size(), 0);
#endif
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
#ifdef IBLGF_COMPILE_CUDA
            if (!inter_send_interface[rank_other].empty())
            {
                auto& first_field =
                    inter_send_interface[rank_other][0].view.field();
                const bool use_device = first_field &&
                    first_field->device_valid();
                if (use_device)
                {
                    ensure_device_buffer_(send_fields_device_[rank_other],
                        send_fields_device_size_[rank_other], size);
                    std::size_t offset = 0;
                    for (auto& interfc : inter_send_interface[rank_other])
                    {
                        auto& field = *interfc.view.field();
                        auto desc = make_desc_(field, interfc.view);
                        const std::size_t n = interfc.view.size();
                        iblgf::gpu::pack_view_device_to_device(
                            field.device_ptr(), desc,
                            send_fields_device_[rank_other] + offset, n);
                        offset += n;
                    }
                    cudaMemcpy(send_fields_[rank_other].data(),
                        send_fields_device_[rank_other],
                        size * sizeof(float_type), cudaMemcpyDeviceToHost);
                }
                else
                {
                    int count = 0;
                    for (auto& interfc : inter_send_interface[rank_other])
                    {
                        interfc.view.iterate([&](const auto& val) {
                            send_fields_[rank_other][count++] = val;
                        });
                    }
                }
            }
#else
            int count = 0;
            for (auto& interfc : inter_send_interface[rank_other])
            {
                interfc.view.iterate([&](const auto& val) {
                    send_fields_[rank_other][count++] = val;
                });
            }
#endif
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
#ifdef IBLGF_COMPILE_CUDA
            if (!inter_recv_interface[rank_other].empty())
            {
                auto& first_field =
                    inter_recv_interface[rank_other][0].view.field();
                const bool use_device = first_field &&
                    first_field->device_valid();
                if (use_device)
                {
                    const std::size_t size = recv_fields_[rank_other].size();
                    ensure_device_buffer_(recv_fields_device_[rank_other],
                        recv_fields_device_size_[rank_other], size);
                    cudaMemcpy(recv_fields_device_[rank_other],
                        recv_fields_[rank_other].data(),
                        size * sizeof(float_type), cudaMemcpyHostToDevice);
                    std::size_t offset = 0;
                    for (auto& interfc : inter_recv_interface[rank_other])
                    {
                        auto& field = *interfc.view.field();
                        auto desc = make_desc_(field, interfc.view);
                        const std::size_t n = interfc.view.size();
                        iblgf::gpu::unpack_view_device_to_device(
                            recv_fields_device_[rank_other] + offset, desc,
                            field.device_ptr(), n);
                        field.mark_device_valid();
                        offset += n;
                    }
                }
                else
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
#else
            int count = 0;
            for (auto& interfc : inter_recv_interface[rank_other])
            {
                auto& recv_view = interfc.view;
                recv_view.iterate([&](auto& val) {
                    val = recv_fields_[rank_other][count++];
                });
            }
#endif
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
#ifdef IBLGF_COMPILE_CUDA
            if (!inter_send_interface[rank_other].empty())
            {
                auto& first_field =
                    inter_send_interface[rank_other][0].view.field();
                const bool use_device = first_field &&
                    first_field->device_valid();
                if (use_device)
                {
                    ensure_device_buffer_(send_fields_device_[rank_other],
                        send_fields_device_size_[rank_other], size);
                    std::size_t offset = 0;
                    for (auto& interfc : inter_send_interface[rank_other])
                    {
                        if ((interfc.field_idx != _field_idx)) continue;
                        auto& field = *interfc.view.field();
                        auto desc = make_desc_(field, interfc.view);
                        const std::size_t n = interfc.view.size();
                        iblgf::gpu::pack_view_device_to_device(
                            field.device_ptr(), desc,
                            send_fields_device_[rank_other] + offset, n);
                        offset += n;
                    }
                    cudaMemcpy(send_fields_[rank_other].data(),
                        send_fields_device_[rank_other],
                        size * sizeof(float_type), cudaMemcpyDeviceToHost);
                }
                else
                {
                    int count = 0;
                    for (auto& interfc : inter_send_interface[rank_other])
                    {
                        if ((interfc.field_idx != _field_idx)) continue;
                        interfc.view.iterate([&](const auto& val) {
                            send_fields_[rank_other][count++] = val;
                        });
                    }
                }
            }
#else
            int count = 0;
            for (auto& interfc : inter_send_interface[rank_other])
            {
                if ((interfc.field_idx != _field_idx)) continue;
                interfc.view.iterate([&](const auto& val) {
                    send_fields_[rank_other][count++] = val;
                });
            }
#endif
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
#ifdef IBLGF_COMPILE_CUDA
            if (!inter_recv_interface[rank_other].empty())
            {
                auto& first_field =
                    inter_recv_interface[rank_other][0].view.field();
                const bool use_device = first_field &&
                    first_field->device_valid();
                if (use_device)
                {
                    const std::size_t size = recv_fields_[rank_other].size();
                    ensure_device_buffer_(recv_fields_device_[rank_other],
                        recv_fields_device_size_[rank_other], size);
                    cudaMemcpy(recv_fields_device_[rank_other],
                        recv_fields_[rank_other].data(),
                        size * sizeof(float_type), cudaMemcpyHostToDevice);
                    std::size_t offset = 0;
                    for (auto& interfc : inter_recv_interface[rank_other])
                    {
                        if ((interfc.field_idx != _field_idx)) continue;
                        auto& field = *interfc.view.field();
                        auto desc = make_desc_(field, interfc.view);
                        const std::size_t n = interfc.view.size();
                        iblgf::gpu::unpack_view_device_to_device(
                            recv_fields_device_[rank_other] + offset, desc,
                            field.device_ptr(), n);
                        field.mark_device_valid();
                        offset += n;
                    }
                }
                else
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
#else
            int count = 0;
            for (auto& interfc : inter_recv_interface[rank_other])
            {
                if ((interfc.field_idx != _field_idx)) continue;
                auto& recv_view = interfc.view;
                recv_view.iterate([&](auto& val) {
                    val = recv_fields_[rank_other][count++];
                });
            }
#endif
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
#ifdef IBLGF_COMPILE_CUDA
        for (auto& ptr : send_fields_device_)
        {
            if (ptr) cudaFree(ptr);
            ptr = nullptr;
        }
        for (auto& ptr : recv_fields_device_)
        {
            if (ptr) cudaFree(ptr);
            ptr = nullptr;
        }
        send_fields_device_.clear();
        recv_fields_device_.clear();
        send_fields_device_size_.clear();
        recv_fields_device_size_.clear();
#endif

        boost::mpi::communicator world;
        inter_send_interface.resize(world.size());
        inter_recv_interface.resize(world.size());
        send_fields_.resize(world.size());
        recv_fields_.resize(world.size());
        send_tasks_.resize(world.size());
        recv_tasks_.resize(world.size());
#ifdef IBLGF_COMPILE_CUDA
        send_fields_device_.resize(world.size(), nullptr);
        recv_fields_device_.resize(world.size(), nullptr);
        send_fields_device_size_.resize(world.size(), 0);
        recv_fields_device_size_.resize(world.size(), 0);
#endif
    }

  private:
    //send/recv interfaces per processor
    std::vector<std::vector<interface>> inter_send_interface;
    std::vector<std::vector<interface>> inter_recv_interface;

    std::vector<std::shared_ptr<TaskType>> send_tasks_;
    std::vector<std::shared_ptr<TaskType>> recv_tasks_;

    std::vector<std::vector<float_type>> send_fields_;
    std::vector<std::vector<float_type>> recv_fields_;

#ifdef IBLGF_COMPILE_CUDA
    std::vector<float_type*> send_fields_device_;
    std::vector<float_type*> recv_fields_device_;
    std::vector<std::size_t> send_fields_device_size_;
    std::vector<std::size_t> recv_fields_device_size_;

    void ensure_device_buffer_(float_type*& ptr, std::size_t& current_size,
        std::size_t needed)
    {
        if (current_size == needed && ptr) return;
        if (ptr)
        {
            cudaFree(ptr);
            ptr = nullptr;
        }
        current_size = needed;
        if (needed > 0)
        {
            cudaMalloc(&ptr, needed * sizeof(float_type));
        }
    }

    template<class DataFieldType, class ViewType>
    iblgf::gpu::block_view_desc make_desc_(const DataFieldType& field,
        const ViewType& view) const
    {
        iblgf::gpu::block_view_desc desc{};
        const auto& f_base = field.real_block().base();
        const auto& f_ext = field.real_block().extent();
        const auto& v_base = view.base();
        const auto& v_ext = view.extent();
        const auto& v_stride = view.stride();

        constexpr std::size_t Dim = DataFieldType::dimension();
        desc.field_base[0] = f_base[0];
        desc.field_extent[0] = f_ext[0];
        desc.view_base[0] = v_base[0];
        desc.view_extent[0] = v_ext[0];
        desc.view_stride[0] = v_stride[0];

        desc.field_base[1] = f_base[1];
        desc.field_extent[1] = f_ext[1];
        desc.view_base[1] = v_base[1];
        desc.view_extent[1] = v_ext[1];
        desc.view_stride[1] = v_stride[1];

        if constexpr (Dim == 3)
        {
            desc.field_base[2] = f_base[2];
            desc.field_extent[2] = f_ext[2];
            desc.view_base[2] = v_base[2];
            desc.view_extent[2] = v_ext[2];
            desc.view_stride[2] = v_stride[2];
        }
        else
        {
            desc.field_base[2] = 0;
            desc.field_extent[2] = 1;
            desc.view_base[2] = 0;
            desc.view_extent[2] = 1;
            desc.view_stride[2] = 1;
        }
        return desc;
    }
#endif

    //intra processor send/recv interfaces
    std::vector<interface> intra_send_interface;
    std::vector<interface> intra_recv_interface;

    int field_idx;
};

} // namespace sr_mpi
} // namespace iblgf
#endif
