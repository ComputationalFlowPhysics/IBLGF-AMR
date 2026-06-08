#ifndef IBLGF_SOLVER_MODAL_ANALYSIS_SYMMETRY_UTILS_HPP
#define IBLGF_SOLVER_MODAL_ANALYSIS_SYMMETRY_UTILS_HPP

namespace iblgf
{
namespace solver
{
namespace modal_analysis
{

template<class KeyT, class CoordT>
inline KeyT mirrored_y_key(
    const CoordT& coord, const KeyT& key, int mirror_span, int base_level)
{
    auto opposite_coord = coord;
    const int ref_level = static_cast<int>(key.level()) - base_level;
    opposite_coord[1] = mirror_span * (1 << ref_level) - (coord[1] + 1);
    return KeyT(opposite_coord, key.level());
}

} // namespace modal_analysis
} // namespace solver
} // namespace iblgf

#endif // IBLGF_SOLVER_MODAL_ANALYSIS_SYMMETRY_UTILS_HPP
