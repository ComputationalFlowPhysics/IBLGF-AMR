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

#ifndef IBLGF_INCLUDED_IFHERK_GPU_HPP
#define IBLGF_INCLUDED_IFHERK_GPU_HPP

#include <iblgf/solver/time_integration/ifherk.hpp>

namespace iblgf
{
namespace solver
{
/**
 * @brief GPU-specialized IFHERK entry point.
 *
 * This header provides a dedicated type for progressive GPU refactors while
 * preserving the existing IFHERK implementation as the numerical reference.
 * In CUDA builds SetupBase can route to this type without affecting CPU builds.
 */
template<class Setup>
class Ifherk_gpu : public Ifherk<Setup>
{
  public:
    using super_type = Ifherk<Setup>;
    using simulation_type = typename super_type::simulation_type;

    explicit Ifherk_gpu(simulation_type* simulation)
    : super_type(configure_gpu_defaults(simulation))
    {
    }

  private:
    static simulation_type* configure_gpu_defaults(
        simulation_type* simulation)
    {
        if (!simulation) return simulation;
        auto dict = simulation->dictionary();
        if (!dict) return simulation;

        // Keep user-provided values authoritative; only fill defaults for the
        // dedicated GPU IFHERK path.
        if (!dict->has_key("gpu_operators"))
            dict->template set_or_insert<bool>("gpu_operators", true);
        if (!dict->has_key("ifherk_extra_client_barriers"))
            dict->template set_or_insert<bool>("ifherk_extra_client_barriers", false);
        if (!dict->has_key("ifherk_gpu_resident_mode"))
            dict->template set_or_insert<bool>("ifherk_gpu_resident_mode", false);

        return simulation;
    }
};

} // namespace solver
} // namespace iblgf

#endif // IBLGF_INCLUDED_IFHERK_GPU_HPP
