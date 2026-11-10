#ifndef IBLGF_INCLUDED_SOLVER_FIELD_VEC_HPP
#define IBLGF_INCLUDED_SOLVER_FIELD_VEC_HPP

// Grid field <-> PETSc Vec transfers shared by the modal-analysis and
// stability tools (POD, StabilityTS, StabilityEVP*asm, JacCheck).
//
// A DOF is a node on a locally owned leaf block whose index field Idx holds a
// positive 1-based slot (0 = not a DOF); the Vec's local array is indexed by
// slot - 1. Which nodes are DOFs (e.g. with or without correction blocks, a
// sub-box only) is decided entirely by how Idx was filled, so the transfers
// themselves just loop over leaf blocks and skip slot <= 0.
//
// part: real  -- grid2vec writes the value as the real part (overwrite);
//                vec2grid reads the real part
//       imag  -- grid2vec ADDS value * i (so a real pass followed by an
//                imaginary pass assembles a complex vector);
//                vec2grid reads the imaginary part
// weight(level): per-level scale applied as value * w (grid2vec) and
//                value / w (vec2grid), e.g. sqrt(cell volume) for an
//                energy inner product on AMR meshes; default 1.

#include <cmath>
#include <functional>

#include <petscvec.h>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

#include <iblgf/global.hpp>

namespace iblgf
{
namespace solver
{
namespace field_vec
{
enum class Part
{
    real,
    imag
};

using weight_fn = std::function<float_type(int level)>;
inline float_type unit_weight(int) { return 1.0; }

/** @brief Number the DOFs of Idx (all its components) on locally owned leaf
 *  blocks (optionally excluding correction blocks) with contiguous 1-based
 *  local slots, after zeroing Idx everywhere. Returns the local count. */
template<class Idx, class Domain>
int assign_leaf_idx(Domain* domain, bool include_correction)
{
    for (auto it = domain->begin(); it != domain->end(); ++it)
    {
        if (!it->has_data() || !it->data().is_allocated()) continue;
        for (std::size_t f = 0; f < Idx::nFields(); ++f)
        {
            auto& lin = it->data_r(Idx::tag(), f).linalg_data();
            std::fill(lin.begin(), lin.end(), 0.0);
        }
    }
    if (domain->is_server()) return 0;
    int n = 0;
    for (auto it = domain->begin(); it != domain->end(); ++it)
    {
        if (!it->locally_owned() || !it->has_data() || !it->is_leaf()) continue;
        if (!include_correction && it->is_correction()) continue;
        for (std::size_t f = 0; f < Idx::nFields(); ++f)
            for (auto& node : it->data()) node(Idx::tag(), f) = ++n;
    }
    return n;
}

/** @brief Field -> Vec (local array). zero_first clears the whole local
 *  array before writing (use it on a real pass that should leave non-DOF
 *  entries at 0). */
template<class Idx, class Field, class Domain>
PetscErrorCode grid2vec(Domain* domain, Vec x, Part part = Part::real,
    bool zero_first = true, const weight_fn& weight = unit_weight)
{
    PetscScalar* a;
    PetscInt     n_loc;
    PetscCall(VecGetLocalSize(x, &n_loc));
    PetscCall(VecGetArray(x, &a));
    if (zero_first)
        for (PetscInt i = 0; i < n_loc; ++i) a[i] = 0.0;
    if (domain->is_client())
    {
        const int base_level = domain->tree()->base_level();
        for (int l = base_level; l < domain->tree()->depth(); ++l)
        {
            const float_type w = weight(l);
            for (auto it = domain->begin(l); it != domain->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data() || !it->is_leaf())
                    continue;
                for (std::size_t f = 0; f < Field::nFields(); ++f)
                    for (auto& node : it->data())
                    {
                        int i = int(node(Idx::tag(), f)) - 1;
                        if (i < 0) continue;
                        float_type v = node(Field::tag(), f) * w;
                        if (part == Part::real) a[i] = v;
                        else
                            a[i] += v * PETSC_i;
                    }
            }
        }
    }
    PetscCall(VecRestoreArray(x, &a));
    return 0;
}

/** @brief Vec (local array, or an all-gathered sequential Vec indexed by a
 *  global Idx) -> Field, real or imaginary part, divided by weight(level).
 *  Only DOF nodes are written. */
template<class Idx, class Field, class Domain>
PetscErrorCode vec2grid(Domain* domain, Vec x, Part part = Part::real,
    const weight_fn& weight = unit_weight)
{
    if (!domain->is_client()) return 0;
    const PetscScalar* a;
    PetscCall(VecGetArrayRead(x, &a));
    const int base_level = domain->tree()->base_level();
    for (int l = base_level; l < domain->tree()->depth(); ++l)
    {
        const float_type w = weight(l);
        for (auto it = domain->begin(l); it != domain->end(l); ++it)
        {
            if (!it->locally_owned() || !it->has_data() || !it->is_leaf())
                continue;
            for (std::size_t f = 0; f < Field::nFields(); ++f)
                for (auto& node : it->data())
                {
                    int i = int(node(Idx::tag(), f)) - 1;
                    if (i < 0) continue;
                    node(Field::tag(), f) = (part == Part::real
                                                 ? PetscRealPart(a[i])
                                                 : PetscImaginaryPart(a[i])) /
                                            w;
                }
        }
    }
    PetscCall(VecRestoreArrayRead(x, &a));
    return 0;
}

/** @brief sqrt(sum over DOFs of (value * weight(level))^2), all ranks. */
template<class Idx, class Field, class Domain>
float_type weighted_l2_norm(Domain* domain, const weight_fn& weight = unit_weight)
{
    boost::mpi::communicator world;
    float_type               s_loc = 0.0;
    if (domain->is_client())
    {
        const int base_level = domain->tree()->base_level();
        for (int l = base_level; l < domain->tree()->depth(); ++l)
        {
            const float_type w = weight(l);
            for (auto it = domain->begin(l); it != domain->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data() || !it->is_leaf())
                    continue;
                for (auto& node : it->data())
                    for (std::size_t f = 0; f < Field::nFields(); ++f)
                    {
                        if (node(Idx::tag(), int(f)) <= 0) continue;
                        const float_type vw = node(Field::tag(), int(f)) * w;
                        s_loc += vw * vw;
                    }
            }
        }
    }
    float_type s = 0.0;
    boost::mpi::all_reduce(world, s_loc, s, std::plus<float_type>());
    return std::sqrt(std::max<float_type>(s, 1e-30));
}

} // namespace field_vec
} // namespace solver
} // namespace iblgf

#endif
