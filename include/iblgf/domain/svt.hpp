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

#ifndef IBLGF_DOMAIN_STARTING_VORTEX_THEORY_HPP
#define IBLGF_DOMAIN_STARTING_VORTEX_THEORY_HPP
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/types.hpp>
#include <cmath>
#include <stdexcept>

namespace iblgf
{
namespace ib
{
using namespace dictionary;
using namespace types;

template<int Dim>
class SVT
{
  public: // member types
    using real_coordinate_type = types::vector_type<float_type, Dim>;
    using coordinate_type = types::vector_type<int, Dim>;
    using point_force_type = types::vector_type<float_type, Dim>;
    using force_type = std::vector<point_force_type>;

  public: // Ctors
    SVT() = default;
    SVT(const SVT&) = default;
    SVT(SVT&&) = default;
    SVT& operator=(const SVT&) & = default;
    SVT& operator=(SVT&&) & = default;
    ~SVT() = default;

  public: // init functions
    template<class DictionaryPtr>
    void init(DictionaryPtr dict)
    {
        auto d2 = dict->get_dictionary("svt");
        p = d2->template get_or<float_type>("p", 2);
        m = d2->template get_or<float_type>("m", 1);
        omega0 = d2->template get_or<float_type>("Omega0", 0.0);
        U0 = d2->template get_or<float_type>("U0", 1.0);
        alpha0 = d2->template get_or<float_type>("alpha0", 0.0);
        x0 = d2->template get_or<float_type>("x0", 0.0);
        // beta_hat = d2->template get_or<float_type>("beta_hat", 0.5);
        // beta = d2->template get_or<float_type>("beta", 1.0);
        // x0 = d2->template get_or<float_type>("x0", 0.0);
    }

    float_type operator()(std::size_t idx, float_type t, real_coordinate_type coord = real_coordinate_type{})
    {
        float_type alpha = alpha0;
        if (p >= 0) alpha = alpha0 + 1 / (p + 1) * omega0 * std::pow(t, p + 1); // for pure translation in y set m for power and alpha=pi/2,p=-1
        if constexpr (Dim == 2)
        {
            // float_type f_alpha;
            // float_type f_u;
            // if (idx == 0) f_alpha=-coord[1]*omega0; //tangent direction
            // else f_alpha = (coord[0]-x0)*omega0; //normal direction
            if (idx == 0) // tangent direction
            {
                float_type u_tan;
                if (m < 0) u_tan = 0.0;
                else
                    u_tan = U0 * std::cos(alpha) * std::pow(t, m);

                float_type u_rot;
                if (p < 0) u_rot = 0.0;
                else
                    u_rot = -coord[1] * omega0 * std::pow(t, p);

                return -(u_tan + u_rot);
            }
            else if (idx == 1) //normal direction
            {
                float_type u_tan;
                if (m < 0) u_tan = 0.0;
                else
                    u_tan = -U0 * std::sin(alpha) * std::pow(t, m);

                float_type u_rot;
                if (p < 0) u_rot = 0.0;
                else
                    u_rot = (coord[0] - x0) * omega0 * std::pow(t, p);

                return -(u_tan + u_rot);
            }
            else
            {
                throw std::runtime_error("idx should be 0 or 1 for 2D problems");
            }
        }
        else if constexpr (Dim == 3)
        {
            if (idx == 1) return 0.0; //no velocity in y direction
            if (idx == 2)             //tangent direction
            {
                float_type u_tan;
                if (m < 0) u_tan = 0.0;
                else
                    u_tan = U0 * std::cos(alpha) * std::pow(t, m);

                float_type u_rot;
                if (p < 0) u_rot = 0.0;
                else
                    u_rot = -coord[0] * omega0 * std::pow(t, p);

                return -(u_tan + u_rot);
            }
            else if (idx == 0) //normal direction
            {
                float_type u_tan;
                if (m < 0) u_tan = 0.0;
                else
                    u_tan = -U0 * std::sin(alpha) * std::pow(t, m);

                float_type u_rot;
                if (p < 0) u_rot = 0.0;
                else
                    u_rot = (coord[2] - x0) * omega0 * std::pow(t, p);

                return -(u_tan + u_rot);
            }
            else
            {
                throw std::runtime_error("idx should be 0,1, 2 for 3D problems");
            }
        }
    }
    float_type p;
    float_type m;
    bool       rotV;
    float_type x0;
    float_type omega0;
    float_type U0;
    float_type alpha0;
};
} // namespace ib

} // namespace iblgf

#endif // IBLGF_DOMAIN_STARTING_VORTEX_THEORY_HPP