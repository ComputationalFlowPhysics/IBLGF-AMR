#ifndef IBLGF_INCLUDED_FMM_AMREX_HPP
#define IBLGF_INCLUDED_FMM_AMREX_HPP

#ifdef IBLGF_USE_AMREX

#include <iblgf/global.hpp>
#include <iblgf/amrex/amrex_domain.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>
#include <iblgf/types.hpp>
#include <iblgf/utilities/convolution.hpp>

#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace iblgf {
namespace fmm {

template<int Dim>
class AmrexFmmFrontEnd
{
  public:
    using Domain = amrex_ext::AmrexDomain;
    using FieldId = amrex_ext::FieldId;
    using float_type = iblgf::types::float_type;

    AmrexFmmFrontEnd(Domain* domain, int block_extent,
        int convolution_batch_size = 10)
    : domain_(domain)
    , block_extent_(block_extent)
    , convolution_batch_size_(convolution_batch_size)
    , conv_(make_conv_dims(), make_conv_dims())
    {}

    template<class Kernel>
    void apply(Kernel* kernel, int level, FieldId source, FieldId target,
        int field_idx, bool non_leaf_as_source, float_type add_with_scale,
        int fmm_type)
    {
        if (domain_ == nullptr || kernel == nullptr)
        {
            throw std::runtime_error("AMReX FMM front-end: null input");
        }

        kernel->change_level(level);

        build_level_cache(level);

        auto& tgt = domain_->field(target, level);
        auto& src = domain_->field(source, level);

        amrex::MultiFab* src_use = &src;
        amrex::MultiFab* tgt_use = &tgt;
        std::unique_ptr<amrex::MultiFab> src_root;
        std::unique_ptr<amrex::MultiFab> tgt_root;

        if (amrex::ParallelDescriptor::NProcs() > 1)
        {
            const auto& ba = domain_->boxArray(level);
            amrex::Vector<int> pmap(ba.size(), 0);
            amrex::DistributionMapping dm_root(pmap);

            src_root = std::make_unique<amrex::MultiFab>(
                ba, dm_root, src.nComp(), src.nGrow());
            tgt_root = std::make_unique<amrex::MultiFab>(
                ba, dm_root, tgt.nComp(), tgt.nGrow());

            src_root->setVal(0.0);
            tgt_root->setVal(0.0);

            src_root->ParallelCopy(src, field_idx, field_idx, 1, src.nGrow(),
                src_root->nGrow());

            src_use = src_root.get();
            tgt_use = tgt_root.get();
        }

        tgt_use->setVal(0.0, field_idx, 1, domain_->nGrow());

        const float_type dx_level =
            static_cast<float_type>(domain_->geom(level).CellSize()[0]);
        const float_type dx_scale =
            kernel->neighbor_only() ? 1.0 : dx_level * dx_level;

        (void)non_leaf_as_source;
        (void)fmm_type;

        auto& cache = level_cache_.at(level);

        for (amrex::MFIter mfi(*tgt_use); mfi.isValid(); ++mfi)
        {
            const int t_idx = mfi.index();
            if (t_idx < 0 ||
                t_idx >= static_cast<int>(cache.blocks.size()))
                continue;

            auto& t_fab = (*tgt_use)[mfi];
            const amrex::Box t_box = t_fab.box();
            FabView<amrex::Array4<amrex::Real>> t_view(
                t_fab.array(), t_box.smallEnd(), field_idx);

            conv_.fft_backward_field_clean();

            if (!printed_t_ && amrex::ParallelDescriptor::MyProc() == 0)
            {
                printed_t_ = true;
                const auto conv_dims = make_conv_dims();
                std::fprintf(stderr,
                    "[AMReX] FMM dims: block_extent=%d ngrow=%d conv=%d,%d,%d tbox=%d,%d,%d\n",
                    block_extent_, domain_->nGrow(),
                    conv_dims[0], (Dim > 1 ? conv_dims[1] : 1),
                    (Dim > 2 ? conv_dims[2] : 1),
                    t_box.length(0), (Dim > 1 ? t_box.length(1) : 1),
                    (Dim > 2 ? t_box.length(2) : 1));
                std::fflush(stderr);
            }

            auto add_source = [&](int s_idx) {
                if (s_idx < 0 ||
                    s_idx >= static_cast<int>(cache.blocks.size()))
                    return;
                if (src_use->DistributionMap()[s_idx] !=
                    amrex::ParallelDescriptor::MyProc())
                    return;

                const auto& s_fab = (*src_use)[s_idx];
                const amrex::Box s_box = s_fab.box();
                FabView<amrex::Array4<const amrex::Real>> s_view(
                    s_fab.const_array(), s_box.smallEnd(), field_idx);
                if (!printed_s_ && amrex::ParallelDescriptor::MyProc() == 0)
                {
                    printed_s_ = true;
                    std::fprintf(stderr,
                        "[AMReX] FMM sbox=%d,%d,%d\n",
                        s_box.length(0), (Dim > 1 ? s_box.length(1) : 1),
                        (Dim > 2 ? s_box.length(2) : 1));
                    std::fflush(stderr);
                }
                fmm_tt(s_box, s_view, t_box, kernel, 0);
            };

            for (int s_idx : cache.neighbors[t_idx])
            {
                add_source(s_idx);
            }

            if (!kernel->neighbor_only())
            {
                for (int s_idx : cache.influences[t_idx])
                {
                    add_source(s_idx);
                }
            }

            block_dsrp_t extractor(dims_t(0), box_extent(t_box));
            conv_.apply_backward(
                extractor, t_view, dx_scale * add_with_scale);
        }

        if (amrex::ParallelDescriptor::NProcs() > 1 && tgt_root)
        {
            tgt.ParallelCopy(*tgt_root, field_idx, field_idx, 1,
                tgt_root->nGrow(), tgt.nGrow());
        }
    }

  private:
    using dims_t = iblgf::types::vector_type<int, Dim>;
    using block_dsrp_t = iblgf::domain::BlockDescriptor<int, Dim>;
    using convolution_t = iblgf::fft::Convolution<Dim>;

    template<class Array4T>
    struct FabView
    {
        FabView(Array4T _arr, const amrex::IntVect& _lo, int _comp)
        : arr(_arr), lo(_lo), comp(_comp) {}

        inline decltype(auto) get_real_local(int i, int j) noexcept
        {
            return arr(i + lo[0], j + lo[1], 0, comp);
        }

        inline decltype(auto) get_real_local(int i, int j) const noexcept
        {
            return arr(i + lo[0], j + lo[1], 0, comp);
        }

        inline decltype(auto) get_real_local(int i, int j, int k) noexcept
        {
            return arr(i + lo[0], j + lo[1], k + lo[2], comp);
        }

        inline decltype(auto) get_real_local(int i, int j, int k) const noexcept
        {
            return arr(i + lo[0], j + lo[1], k + lo[2], comp);
        }

        Array4T        arr;
        amrex::IntVect lo;
        int            comp = 0;
    };

    static dims_t box_extent(const amrex::Box& box)
    {
        dims_t ext{};
        ext[0] = box.length(0);
        if constexpr (Dim > 1) ext[1] = box.length(1);
        if constexpr (Dim > 2) ext[2] = box.length(2);
        return ext;
    }

    static dims_t box_base(const amrex::Box& box)
    {
        dims_t base{};
        base[0] = box.smallEnd(0);
        if constexpr (Dim > 1) base[1] = box.smallEnd(1);
        if constexpr (Dim > 2) base[2] = box.smallEnd(2);
        return base;
    }

    dims_t make_conv_dims() const
    {
        const int ng = domain_ ? domain_->nGrow() : 0;
        const int extent = block_extent_ + 2 * ng;
        dims_t dims{};
        dims[0] = extent;
        if constexpr (Dim > 1) dims[1] = extent;
        if constexpr (Dim > 2) dims[2] = extent;
        return dims;
    }

    template<class Kernel, class SourceView>
    void fmm_tt(const amrex::Box& s_box, const SourceView& s_view,
        const amrex::Box& t_box, Kernel* kernel, int level_diff)
    {
        const auto t_base = box_base(t_box);
        const auto s_base = box_base(s_box);
        const auto s_extent = box_extent(s_box);
        const auto shift = t_base - s_base;
        const auto base_lgf = shift - (s_extent - 1);
        const auto extent_lgf = 2 * s_extent - 1;
        block_dsrp_t lgf_block(base_lgf, extent_lgf);
        conv_.apply_forward_add(lgf_block, kernel, level_diff, s_view);
    }

    struct VecKey
    {
        std::array<int, Dim> v;
        bool operator==(const VecKey& other) const noexcept
        {
            return v == other.v;
        }
    };

    struct VecKeyHash
    {
        std::size_t operator()(const VecKey& k) const noexcept
        {
            std::size_t h = 0;
            for (int i = 0; i < Dim; ++i)
            {
                h ^= std::hash<int>{}(k.v[i] + 0x9e3779b9 + (h << 6) + (h >> 2));
            }
            return h;
        }
    };

    struct BlockInfo
    {
        amrex::Box box;
        std::array<int, Dim> coord;
    };

    struct M2LPair
    {
        int target = -1;
        int source = -1;
        std::array<int, Dim> offset;
    };

    struct LevelCache
    {
        bool built = false;
        std::vector<BlockInfo> blocks;
        std::vector<std::vector<int>> neighbors;
        std::vector<std::vector<int>> influences;
        std::vector<M2LPair> m2l_pairs;
        std::unordered_map<VecKey, int, VecKeyHash> coord_to_id;
    };

    void build_level_cache(int level)
    {
        if (level < 0 || level > domain_->finestLevel())
        {
            throw std::runtime_error("AMReX FMM front-end: invalid level");
        }

        if (static_cast<int>(level_cache_.size()) <= level)
        {
            level_cache_.resize(level + 1);
        }

        auto& cache = level_cache_[level];
        if (cache.built) return;

        const auto& ba = domain_->boxArray(level);
        const auto domain_box = domain_->geom(level).Domain();
        const auto base = domain_box.smallEnd();

        cache.blocks.reserve(ba.size());
        for (int i = 0; i < ba.size(); ++i)
        {
            const auto& box = ba[i];
            const auto  small = box.smallEnd();
            std::array<int, Dim> coord{};
            coord[0] = (small[0] - base[0]) / block_extent_;
            if constexpr (Dim > 1)
                coord[1] = (small[1] - base[1]) / block_extent_;
            if constexpr (Dim > 2)
                coord[2] = (small[2] - base[2]) / block_extent_;

            cache.blocks.push_back(BlockInfo{box, coord});
            cache.coord_to_id.emplace(VecKey{coord}, i);
        }

        cache.neighbors.resize(cache.blocks.size());
        cache.influences.resize(cache.blocks.size());

        for (std::size_t i = 0; i < cache.blocks.size(); ++i)
        {
            const auto& c = cache.blocks[i].coord;

            // Neighbor list: offsets in [-1, 1] in each dim (including self)
            for (int dz = (Dim == 3 ? -1 : 0); dz <= (Dim == 3 ? 1 : 0); ++dz)
            {
                for (int dy = (Dim >= 2 ? -1 : 0);
                     dy <= (Dim >= 2 ? 1 : 0); ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        std::array<int, Dim> nc{c[0] + dx,
                            (Dim > 1 ? c[1] + dy : 0),
                            (Dim > 2 ? c[2] + dz : 0)};

                        auto it = cache.coord_to_id.find(VecKey{nc});
                        if (it != cache.coord_to_id.end())
                        {
                            cache.neighbors[i].push_back(it->second);
                        }
                    }
                }
            }

            // Influence list: parent neighbors' children excluding near field
            if (level == 0) continue;

            std::array<int, Dim> parent{c[0] / 2,
                (Dim > 1 ? c[1] / 2 : 0),
                (Dim > 2 ? c[2] / 2 : 0)};

            std::unordered_set<int> infl_set;
            for (int dz = (Dim == 3 ? -1 : 0); dz <= (Dim == 3 ? 1 : 0); ++dz)
            {
                for (int dy = (Dim >= 2 ? -1 : 0);
                     dy <= (Dim >= 2 ? 1 : 0); ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        std::array<int, Dim> pn{parent[0] + dx,
                            (Dim > 1 ? parent[1] + dy : 0),
                            (Dim > 2 ? parent[2] + dz : 0)};

                        for (int cz = 0; cz <= (Dim == 3 ? 1 : 0); ++cz)
                        {
                            for (int cy = 0; cy <= (Dim >= 2 ? 1 : 0); ++cy)
                            {
                                for (int cx = 0; cx <= 1; ++cx)
                                {
                                    std::array<int, Dim> child{
                                        pn[0] * 2 + cx,
                                        (Dim > 1 ? pn[1] * 2 + cy : 0),
                                        (Dim > 2 ? pn[2] * 2 + cz : 0)};

                                    const int dx_c = std::abs(child[0] - c[0]);
                                    const int dy_c = (Dim > 1)
                                                         ? std::abs(child[1] - c[1])
                                                         : 0;
                                    const int dz_c = (Dim > 2)
                                                         ? std::abs(child[2] - c[2])
                                                         : 0;

                                    if (dx_c <= 1 && dy_c <= 1 && dz_c <= 1)
                                        continue;

                                    auto it =
                                        cache.coord_to_id.find(VecKey{child});
                                    if (it != cache.coord_to_id.end())
                                    {
                                        infl_set.emplace(it->second);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            cache.influences[i].assign(infl_set.begin(), infl_set.end());
        }

        for (std::size_t t = 0; t < cache.influences.size(); ++t)
        {
            const auto& tc = cache.blocks[t].coord;
            for (int s : cache.influences[t])
            {
                const auto& sc = cache.blocks[s].coord;
                std::array<int, Dim> off{
                    sc[0] - tc[0],
                    (Dim > 1 ? sc[1] - tc[1] : 0),
                    (Dim > 2 ? sc[2] - tc[2] : 0)};
                cache.m2l_pairs.push_back(M2LPair{
                    static_cast<int>(t), s, off});
            }
        }

        cache.built = true;
    }

    Domain* domain_ = nullptr;
    int     block_extent_ = 0;
    int     convolution_batch_size_ = 0;
    convolution_t conv_;
    std::vector<LevelCache> level_cache_;
    bool printed_t_ = false;
    bool printed_s_ = false;
};

} // namespace fmm
} // namespace iblgf

#else

#error "AMReX support not enabled. Configure with -DIBLGF_USE_AMREX=ON"

#endif

#endif // IBLGF_INCLUDED_FMM_AMREX_HPP
