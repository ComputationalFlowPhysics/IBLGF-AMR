#ifndef IBLGF_INCLUDED_SOLVER_POISSON_AMREX_HPP
#define IBLGF_INCLUDED_SOLVER_POISSON_AMREX_HPP

#ifdef IBLGF_USE_AMREX

#include <iblgf/amrex/amrex_domain.hpp>
#include <iblgf/fmm/fmm_amrex.hpp>

#include <AMReX_MultiFabUtil.H>

namespace iblgf {
namespace solver {

template<class Fmm, class Kernel>
class PoissonSolverAMReX
{
  public:
    using Domain = amrex_ext::AmrexDomain;
    using FieldId = amrex_ext::FieldId;

    PoissonSolverAMReX(Domain* domain, Fmm* fmm)
    : domain_(domain)
    , fmm_(fmm)
    {}

    void apply_lgf(Kernel* kernel, int field_idx, int fmm_type)
    {
        if (domain_ == nullptr || fmm_ == nullptr || kernel == nullptr) return;

        clean_field(FieldId::SourceTmp);
        clean_field(FieldId::TargetTmp);
        clean_field(FieldId::CorrectionTmp);

        // Copy source into temporary storage (leaf-level equivalent)
        copy_leaf(FieldId::Source, FieldId::SourceTmp, field_idx, true);

        // Coarsen source to all levels
        coarsen_source(field_idx);

        const int finest = domain_->finestLevel();
        const int base_level = domain_->baseLevel();
        for (int lev = 0; lev <= finest; ++lev)
        {
            // Kernel may depend on resolution (LGF level)
            kernel->change_level(base_level + lev);

            fmm_->apply(kernel, lev, FieldId::SourceTmp, FieldId::TargetTmp,
                field_idx, false, 1.0, fmm_type);

            // TODO: Interpolate coarse-to-fine targets if needed for AMR2AMR
            // interpolate_target(lev);

            copy_level(FieldId::TargetTmp, FieldId::Target, lev, field_idx, true);
        }
    }

    void apply_lgf_all(Kernel* kernel, int fmm_type)
    {
        for (int comp = 0; comp < domain_->nComp(); ++comp)
        {
            apply_lgf(kernel, comp, fmm_type);
        }
    }

  private:
    void clean_field(FieldId id)
    {
        domain_->setVal(id, 0.0);
    }

    void copy_level(FieldId from, FieldId to, int level, int field_idx,
        bool with_ghost)
    {
        const int ng = with_ghost ? domain_->nGrow() : 0;
        amrex::MultiFab::Copy(domain_->field(to, level),
            domain_->field(from, level), field_idx, field_idx, 1, ng);
    }

    void copy_leaf(FieldId from, FieldId to, int field_idx, bool with_ghost)
    {
        const int ng = with_ghost ? domain_->nGrow() : 0;
        const int finest = domain_->finestLevel();
        for (int lev = 0; lev <= finest; ++lev)
        {
            amrex::MultiFab::Copy(domain_->field(to, lev),
                domain_->field(from, lev), field_idx, field_idx, 1, ng);
        }
    }

    void coarsen_source(int field_idx)
    {
        const int finest = domain_->finestLevel();
        for (int lev = finest - 1; lev >= 0; --lev)
        {
            const amrex::IntVect& rr = domain_->refRatio().at(lev);
            amrex::average_down(domain_->field(FieldId::SourceTmp, lev + 1),
                domain_->field(FieldId::SourceTmp, lev), field_idx, 1, rr);
        }
    }

    Domain* domain_ = nullptr;
    Fmm*    fmm_ = nullptr;
};

} // namespace solver
} // namespace iblgf

#else

#error "AMReX support not enabled. Configure with -DIBLGF_USE_AMREX=ON"

#endif

#endif // IBLGF_INCLUDED_SOLVER_POISSON_AMREX_HPP
