#pragma once

#include "../ns_amr_lgf/ns_amr_lgf.hpp"
#include <algorithm>
#include <limits>

#ifdef IBLGF_COMPILE_CUDA
#include <cuda_runtime.h>
#include <iblgf/operators/operators_GPU.hpp>
#endif

namespace iblgf
{
namespace debug
{

struct NS_AMR_LGF_Debug : public NS_AMR_LGF
{
    using NS_AMR_LGF::NS_AMR_LGF;
    using super_type::domain_;
    using super_type::simulation_;
};

#ifdef IBLGF_COMPILE_CUDA

template<class Block, class DataField>
static iblgf::gpu::ops::block_desc make_desc_(const Block& block,
    const DataField& field, bool use_bounding_box) noexcept
{
    iblgf::gpu::ops::block_desc desc{};
    const auto& b = use_bounding_box ? block.bounding_box() : block.descriptor();
    const auto& b_base = b.base();
    const auto& b_ext = b.extent();
    desc.block_base[0] = b_base[0];
    desc.block_extent[0] = b_ext[0];
    desc.block_base[1] = b_base[1];
    desc.block_extent[1] = b_ext[1];
    if constexpr (DataField::dimension() == 3)
    {
        desc.block_base[2] = b_base[2];
        desc.block_extent[2] = b_ext[2];
        desc.dim = 3;
    }
    else
    {
        desc.block_base[2] = 0;
        desc.block_extent[2] = 1;
        desc.dim = 2;
    }

    const auto& f_base = field.real_block().base();
    const auto& f_ext = field.real_block().extent();
    desc.field_base[0] = f_base[0];
    desc.field_extent[0] = f_ext[0];
    desc.field_base[1] = f_base[1];
    desc.field_extent[1] = f_ext[1];
    if constexpr (DataField::dimension() == 3)
    {
        desc.field_base[2] = f_base[2];
        desc.field_extent[2] = f_ext[2];
    }
    else
    {
        desc.field_base[2] = 0;
        desc.field_extent[2] = 1;
    }
    return desc;
}

#endif

template<class DataField>
static float_type debug_maxabs_datafield_(DataField& df, bool prefer_device)
{
    using data_t = typename DataField::data_type;
    float_type max_abs = 0.0;
#ifdef IBLGF_COMPILE_CUDA
    if (prefer_device && df.device_valid())
    {
        const std::size_t n = df.real_block().size();
        std::vector<data_t> host(n);
        cudaMemcpy(host.data(), df.device_ptr(), n * sizeof(data_t),
            cudaMemcpyDeviceToHost);
        for (const auto& v : host)
        {
            const float_type av = std::abs(static_cast<float_type>(v));
            if (av > max_abs) max_abs = av;
        }
        return max_abs;
    }
#else
    (void)prefer_device;
#endif
    for (const auto& v : df.data())
    {
        const float_type av = std::abs(static_cast<float_type>(v));
        if (av > max_abs) max_abs = av;
    }
    return max_abs;
}

template<class Field>
static float_type debug_maxabs_field_(NS_AMR_LGF_Debug& setup,
    bool prefer_device)
{
    float_type max_abs = 0.0;
    for (auto it = setup.domain_->begin(); it != setup.domain_->end(); ++it)
    {
        if (!it->locally_owned()) continue;
        if (!it->has_data() || !it->data().is_allocated()) continue;
        if (!it->is_leaf() && !it->is_correction()) continue;
        if (it->is_leaf() && it->is_correction()) continue;
        for (std::size_t field_idx = 0; field_idx < Field::nFields();
             ++field_idx)
        {
            auto& df = it->data_r(Field::tag(), field_idx);
            const float_type local = debug_maxabs_datafield_(df, prefer_device);
            if (local > max_abs) max_abs = local;
        }
    }
    return max_abs;
}

static void debug_block_census(NS_AMR_LGF_Debug& setup)
{
    int local_blocks = 0;
    int local_leaves = 0;
    int local_has_data = 0;
    int local_allocated = 0;

    for (auto it = setup.domain_->begin(); it != setup.domain_->end(); ++it)
    {
        if (!it->locally_owned()) continue;
        ++local_blocks;
        if (it->is_leaf()) ++local_leaves;
        if (it->has_data()) ++local_has_data;
        if (it->has_data() && it->data().is_allocated()) ++local_allocated;
    }

    boost::mpi::communicator world;
    std::cout << "Rank " << world.rank()
              << " census | server=" << setup.domain_->is_server()
              << " client=" << setup.domain_->is_client()
              << " blocks=" << local_blocks
              << " leaves=" << local_leaves
              << " has_data=" << local_has_data
              << " allocated=" << local_allocated
              << std::endl;
}

static void debug_init_stats(NS_AMR_LGF_Debug& setup,
    bool prefer_device = true)
{
    debug_block_census(setup);
    if (setup.domain_->is_server()) return;
    const auto u_max =
        debug_maxabs_field_<parameters::u_type>(setup, prefer_device);
    const auto p_max =
        debug_maxabs_field_<parameters::p_type>(setup, prefer_device);
    const auto t_max =
        debug_maxabs_field_<parameters::test_type>(setup, prefer_device);
    const auto edge_max =
        debug_maxabs_field_<NS_AMR_LGF_Debug::edge_aux_type>(setup, prefer_device);
    boost::mpi::communicator world;
    std::cout << "Rank " << world.rank()
              << " init stats | maxabs(u)=" << u_max
              << " maxabs(p)=" << p_max
              << " maxabs(test)=" << t_max
              << " maxabs(edge_aux)=" << edge_max << std::endl;
}

static void debug_kernel_microtests(NS_AMR_LGF_Debug& setup,
    bool prefer_device = true)
{
#ifdef IBLGF_COMPILE_CUDA
    debug_block_census(setup);
    if (setup.domain_->is_server()) return;
    auto it = setup.domain_->begin();
    for (; it != setup.domain_->end(); ++it)
    {
        if (!it->locally_owned()) continue;
        if (!it->has_data() || !it->data().is_allocated()) continue;
        if (!it->is_leaf() && !it->is_correction()) continue;
        if (it->is_leaf() && it->is_correction()) continue;
        break;
    }
    if (it == setup.domain_->end())
    {
        boost::mpi::communicator world;
        std::cout << "Rank " << world.rank()
                  << " kernel microtests: no valid local blocks" << std::endl;
        return;
    }

    auto& block = it->data();
    auto& df = it->data_r(parameters::test_type::tag());

    std::fill(df.data().begin(), df.data().end(),
        static_cast<float_type>(1.0));
    df.update_device();

    auto desc = make_desc_(block, df, false);
    iblgf::gpu::ops::set_constant_field_device(
        df.device_ptr(), desc, static_cast<float_type>(2.0));
    cudaDeviceSynchronize();

    const auto after_set = debug_maxabs_datafield_(df, prefer_device);

    iblgf::gpu::ops::zero_boundary_device(df.device_ptr(), desc, 1);
    cudaDeviceSynchronize();

    const auto after_zero = debug_maxabs_datafield_(df, prefer_device);

    boost::mpi::communicator world;
    std::cout << "Rank " << world.rank()
              << " kernel microtests | maxabs(test) after set_constant= "
              << after_set << " after zero_boundary= " << after_zero
              << std::endl;
#else
    (void)setup;
    (void)prefer_device;
#endif
}

static void debug_run_lgf(NS_AMR_LGF_Debug& setup,
    bool prefer_device = true)
{
    debug_block_census(setup);
    if (setup.domain_->is_server()) return;

    using edge_aux_t = typename NS_AMR_LGF_Debug::edge_aux_type;
    using stream_f_t = typename NS_AMR_LGF_Debug::stream_f_type;

    typename NS_AMR_LGF_Debug::poisson_solver_t psolver(&setup.simulation_);
    psolver.template apply_lgf<edge_aux_t, stream_f_t>();

    const auto edge_max = debug_maxabs_field_<edge_aux_t>(setup, prefer_device);
    const auto stream_max =
        debug_maxabs_field_<stream_f_t>(setup, prefer_device);

    boost::mpi::communicator world;
    std::cout << "Rank " << world.rank()
              << " lgf stats | maxabs(edge_aux)=" << edge_max
              << " maxabs(stream_f)=" << stream_max << std::endl;
}

} // namespace debug
} // namespace iblgf
