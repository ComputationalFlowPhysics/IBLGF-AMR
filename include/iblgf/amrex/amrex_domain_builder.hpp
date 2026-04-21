#ifndef IBLGF_INCLUDED_AMREX_DOMAIN_BUILDER_HPP
#define IBLGF_INCLUDED_AMREX_DOMAIN_BUILDER_HPP

#ifdef IBLGF_USE_AMREX

#include <iblgf/amrex/amrex_domain.hpp>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_RealBox.H>
#include <stdexcept>
#include <iostream>
#include <cstdio>

namespace iblgf {
namespace amrex_ext {

namespace detail {

template<class Coord>
amrex::IntVect to_intvect(const Coord& c)
{
    return amrex::IntVect(AMREX_D_DECL(c[0], c[1], c[2]));
}

template<class Coord>
amrex::RealBox to_realbox(const Coord& lo, const Coord& hi)
{
    return amrex::RealBox({AMREX_D_DECL(static_cast<amrex::Real>(lo[0]),
                               static_cast<amrex::Real>(lo[1]),
                               static_cast<amrex::Real>(lo[2]))},
        {AMREX_D_DECL(static_cast<amrex::Real>(hi[0]),
            static_cast<amrex::Real>(hi[1]),
            static_cast<amrex::Real>(hi[2]))});
}

} // namespace detail

template<class DomainT>
AmrexDomain build_amrex_domain_from_iblgf(DomainT* domain, int ncomp, int ngrow)
{
    if (domain == nullptr)
    {
        throw std::runtime_error("AMReX domain build failed: null domain");
    }

    const int block_extent = domain->block_extent()[0];

    const int myproc = amrex::ParallelDescriptor::MyProc();
    std::fprintf(stderr,
        "[AMReX] build_amrex_domain_from_iblgf: rank=%d domain=%p tree=%p\n",
        myproc, static_cast<void*>(domain),
        static_cast<void*>(domain->tree().get()));
    std::fflush(stderr);

    const auto bb = domain->bounding_box();
    const auto dx_base = domain->dx_base();

    const int l_min = domain->tree()->base_level();
    const int l_max = domain->tree()->depth() - 1;
    const int nlevels = l_max - l_min + 1;

    amrex::Vector<amrex::Geometry> geom(nlevels);
    amrex::Vector<amrex::BoxArray> ba(nlevels);
    amrex::Vector<amrex::DistributionMapping> dm(nlevels);
    amrex::Vector<amrex::IntVect> ref_ratio;
    if (nlevels > 1) ref_ratio.resize(nlevels - 1, amrex::IntVect(2));

    const auto phys_lo = bb.base() * dx_base;
    const auto phys_hi = (bb.base() + bb.extent()) * dx_base;
    amrex::RealBox real_box = detail::to_realbox(phys_lo, phys_hi);
    int is_per[3] = {0, 0, 0};

    const int nprocs = amrex::ParallelDescriptor::NProcs();

    for (int lev = 0; lev < nlevels; ++lev)
    {
        if (myproc == 0)
        {
            std::cout << "[AMReX] level " << lev
                      << " of " << nlevels
                      << std::endl;
        }
        const int level = l_min + lev;
        const int scale = 1 << lev;

        const auto base = bb.base() * scale;
        const auto extent = bb.extent() * scale;

        amrex::Box domain_box(detail::to_intvect(base),
            detail::to_intvect(base + extent - 1));

        geom[lev] = amrex::Geometry(domain_box, &real_box,
            amrex::CoordSys::cartesian, is_per);

        std::vector<int> local_boxes;
        for (auto it = domain->begin(level); it != domain->end(level); ++it)
        {
            if (!it->has_data() || !it->is_leaf() || !it->locally_owned())
                continue;
            const auto& desc = it->data().descriptor();
            const auto  bbase = desc.base();
            const auto  bext = desc.extent();
            for (int d = 0; d < static_cast<int>(DomainT::dimension()); ++d)
                local_boxes.push_back(bbase[d]);
            for (int d = 0; d < static_cast<int>(DomainT::dimension()); ++d)
                local_boxes.push_back(bext[d]);
        }

        std::vector<int> all_boxes;
        if (nprocs > 1)
        {
            const int local_count = static_cast<int>(local_boxes.size());
            if (local_count == 0)
            {
                // Provide a non-null send buffer for MPI_Gatherv on empty ranks
                local_boxes.resize(1, 0);
            }
            if (myproc == 0)
            {
                std::cout << "[AMReX] local_count=" << local_count << std::endl;
            }
            std::vector<int> recv_counts(nprocs, 0);

            amrex::ParallelDescriptor::Gather(
                &local_count, 1, recv_counts.data(), 1, 0);

            std::vector<int> displs(nprocs, 0);
            int total_count = 0;
            if (myproc == 0)
            {
                for (int p = 0; p < nprocs; ++p)
                {
                    displs[p] = total_count;
                    total_count += recv_counts[p];
                }
                all_boxes.resize(total_count, 0);
            }
            else
            {
                // Provide a non-null recv buffer for MPI_Gatherv on non-root
                all_boxes.resize(1, 0);
            }

            amrex::ParallelDescriptor::Gatherv(
                local_boxes.data(), local_count,
                all_boxes.data(), recv_counts, displs, 0);

            int total_count_bcast =
                (myproc == 0) ? static_cast<int>(all_boxes.size()) : 0;
            amrex::ParallelDescriptor::Bcast(&total_count_bcast, 1, 0);
            if (myproc != 0) all_boxes.resize(total_count_bcast, 0);
            if (total_count_bcast > 0)
            {
                amrex::ParallelDescriptor::Bcast(
                    all_boxes.data(), total_count_bcast, 0);
            }
        }
        else
        {
            all_boxes = std::move(local_boxes);
        }

        amrex::BoxList bl;
        const int stride = 2 * static_cast<int>(DomainT::dimension());
        for (int i = 0; i + stride - 1 < static_cast<int>(all_boxes.size());
             i += stride)
        {
            typename DomainT::coordinate_type bbase;
            typename DomainT::coordinate_type bext;
            for (int d = 0; d < static_cast<int>(DomainT::dimension()); ++d)
            {
                bbase[d] = all_boxes[i + d];
                bext[d] = all_boxes[i + static_cast<int>(DomainT::dimension()) + d];
            }
            amrex::Box box(detail::to_intvect(bbase),
                detail::to_intvect(bbase + bext - 1));
            bl.push_back(box);
        }

        ba[lev] = amrex::BoxArray(bl);
        ba[lev].removeOverlap();
        if (block_extent > 0)
        {
            ba[lev].maxSize(block_extent);
        }
        dm[lev] = amrex::DistributionMapping(ba[lev]);
    }

    return AmrexDomain(geom, ba, dm, ref_ratio, l_min, ncomp, ngrow);
}

} // namespace amrex_ext
} // namespace iblgf

#else

#error "AMReX support not enabled. Configure with -DIBLGF_USE_AMREX=ON"

#endif

#endif // IBLGF_INCLUDED_AMREX_DOMAIN_BUILDER_HPP
