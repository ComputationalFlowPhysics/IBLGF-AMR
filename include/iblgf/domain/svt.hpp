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

namespace iblgf
{
namespace ib
{
using namespace dictionary;
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
        p = d2->template get_or<int>("p", 2);
        m = d2->template get_or<int>("m", 1);
        beta_hat = d2->template get_or<float_type>("beta_hat", 0.5);
        rotV = d2->template get_or<bool>("rotV", false);
        x0 = d2->template get_or<float_type>("x0", 0.0);
    }

    float_type operator()(std::size_t idx, float_type t, real_coordinate_type coord = real_coordinate_type{})
    {
        if (!rotV) // seperate U and V power laws
        {
            if (Dim == 3 && idx == 1) return 0.0;
            if ((Dim == 2 && idx == 0) || (Dim == 3 && idx == 2)) //tangent direction, for just tangent set beta_hat=0
            {
                if (m < 0) return 0.0;
                auto h1 = 1.0 / std::sqrt(4 * std::pow(beta_hat, 2) + 1);
                return -h1 * std::pow(t, m);
            }
            else if ((Dim == 2 && idx == 1) ||
                     (Dim == 3 && idx == 0)) //normal direction, for just normal set beta_hat<0
            {
                if (p < 0) return 0.0;
                float_type h1;
                if (beta_hat < 0) { h1 = 1.0; }
                else { h1 = 2.0 * beta_hat / std::sqrt(4 * std::pow(beta_hat, 2) + 1); }
                return -h1 * std::pow(t, p);
            }
        }
        else // combined rotation
        {
            float_type f_alpha = 0.0;
            float_type f_u = 0.0;
            if (Dim != 2) throw std::runtime_error("rotV option is only implemented for 2-D problems");
            if (idx == 0)
            {
                f_alpha = -(coord[1] * beta_hat * 4)/std::sqrt(4 * std::pow(beta_hat, 2) + 1);
                f_u = 1.0 / std::sqrt(4 * std::pow(beta_hat, 2) + 1);
            }
            else if (idx == 1)
            {
                f_alpha = ((coord[0] - x0) * beta_hat * 4)/ std::sqrt(4 * std::pow(beta_hat, 2) + 1);
                f_u = 0.0;
            }
			float_type val_u=std::pow(t,m);
			float_type val_a=std::pow(t,p);
			return -(f_u * val_u + f_alpha * val_a);
        }
        return 0.0;
    }

  private:
    int        p;
    int        m;
    float_type beta_hat;
    float_type d;
    bool       rotV;
    float_type x0;
};
} // namespace ib

} // namespace iblgf

#endif // IBLGF_DOMAIN_STARTING_VORTEX_THEORY_HPP