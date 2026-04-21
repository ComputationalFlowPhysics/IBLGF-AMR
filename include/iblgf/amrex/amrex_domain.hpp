#ifndef IBLGF_INCLUDED_AMREX_DOMAIN_HPP
#define IBLGF_INCLUDED_AMREX_DOMAIN_HPP

#ifdef IBLGF_USE_AMREX

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

namespace iblgf {
namespace amrex_ext {

enum class FieldId
{
    Source,
    Target,
    SourceTmp,
    TargetTmp,
    CorrectionTmp
};

class AmrexDomain
{
  public:
    AmrexDomain(const amrex::Vector<amrex::Geometry>& geom,
        const amrex::Vector<amrex::BoxArray>&        ba,
        const amrex::Vector<amrex::DistributionMapping>& dm,
        const amrex::Vector<amrex::IntVect>&         ref_ratio,
        int                                          base_level,
        int                                          ncomp,
        int                                          ngrow)
    : geom_(geom)
    , ba_(ba)
    , dm_(dm)
    , ref_ratio_(ref_ratio)
    , base_level_(base_level)
    , ncomp_(ncomp)
    , ngrow_(ngrow)
    , finest_level_(static_cast<int>(geom.size()) - 1)
    {
        source_.reserve(geom.size());
        target_.reserve(geom.size());
        source_tmp_.reserve(geom.size());
        target_tmp_.reserve(geom.size());
        correction_tmp_.reserve(geom.size());

        for (int lev = 0; lev <= finest_level_; ++lev)
        {
            source_.emplace_back(ba_[lev], dm_[lev], ncomp_, ngrow_);
            target_.emplace_back(ba_[lev], dm_[lev], ncomp_, ngrow_);
            source_tmp_.emplace_back(ba_[lev], dm_[lev], ncomp_, ngrow_);
            target_tmp_.emplace_back(ba_[lev], dm_[lev], ncomp_, ngrow_);
            correction_tmp_.emplace_back(ba_[lev], dm_[lev], ncomp_, ngrow_);
        }
    }

    int finestLevel() const noexcept { return finest_level_; }
    int baseLevel() const noexcept { return base_level_; }
    int nLevels() const noexcept { return static_cast<int>(geom_.size()); }
    int nComp() const noexcept { return ncomp_; }
    int nGrow() const noexcept { return ngrow_; }

    const amrex::Vector<amrex::IntVect>& refRatio() const noexcept
    {
        return ref_ratio_;
    }

    const amrex::Geometry& geom(int level) const { return geom_.at(level); }
    amrex::Geometry&       geom(int level) { return geom_.at(level); }

    const amrex::BoxArray& boxArray(int level) const { return ba_.at(level); }
    const amrex::DistributionMapping& distMap(int level) const
    {
        return dm_.at(level);
    }

    amrex::MultiFab& field(FieldId id, int level)
    {
        switch (id)
        {
            case FieldId::Source: return source_.at(level);
            case FieldId::Target: return target_.at(level);
            case FieldId::SourceTmp: return source_tmp_.at(level);
            case FieldId::TargetTmp: return target_tmp_.at(level);
            case FieldId::CorrectionTmp: return correction_tmp_.at(level);
            default: return source_.at(level);
        }
    }

    const amrex::MultiFab& field(FieldId id, int level) const
    {
        switch (id)
        {
            case FieldId::Source: return source_.at(level);
            case FieldId::Target: return target_.at(level);
            case FieldId::SourceTmp: return source_tmp_.at(level);
            case FieldId::TargetTmp: return target_tmp_.at(level);
            case FieldId::CorrectionTmp: return correction_tmp_.at(level);
            default: return source_.at(level);
        }
    }

    void setVal(FieldId id, amrex::Real value)
    {
        for (int lev = 0; lev <= finest_level_; ++lev)
        {
            field(id, lev).setVal(value);
        }
    }

  private:
    amrex::Vector<amrex::Geometry>             geom_;
    amrex::Vector<amrex::BoxArray>             ba_;
    amrex::Vector<amrex::DistributionMapping>  dm_;
    amrex::Vector<amrex::IntVect>              ref_ratio_;
    int                                        ncomp_ = 0;
    int                                        ngrow_ = 0;
    int                                        finest_level_ = 0;
    int                                        base_level_ = 0;

    amrex::Vector<amrex::MultiFab> source_;
    amrex::Vector<amrex::MultiFab> target_;
    amrex::Vector<amrex::MultiFab> source_tmp_;
    amrex::Vector<amrex::MultiFab> target_tmp_;
    amrex::Vector<amrex::MultiFab> correction_tmp_;
};

} // namespace amrex_ext
} // namespace iblgf

#else

#error "AMReX support not enabled. Configure with -DIBLGF_USE_AMREX=ON"

#endif

#endif // IBLGF_INCLUDED_AMREX_DOMAIN_HPP
