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

#ifndef IBLGF_INCLUDED_OPERATORS_HPP
#define IBLGF_INCLUDED_OPERATORS_HPP

#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/domain/dataFields/datafield.hpp>
#include <iblgf/types.hpp>

namespace iblgf
{
namespace domain
{
struct Operator
{
  public:
    Operator(const Operator& other) = default;
    Operator(Operator&& other) = default;
    Operator& operator=(const Operator& other) & = default;
    Operator& operator=(Operator&& other) & = default;
    ~Operator() = default;
    Operator() = default;

  public:
    template<class F_in, class F_tmp, class Block>
    static void cell_center_average(Block& block) noexcept
    {

        constexpr auto f_in = F_in::tag();
        constexpr auto tmp = F_tmp::tag();

        for (std::size_t field_idx = 0; field_idx < F_in::nFields(); ++field_idx)
        {

            std::array<int, 3> off{{0,0,0}};

            if (F_in::mesh_type() == MeshObject::face){
                off[field_idx]=1;
            }
            else if (F_in::mesh_type() == MeshObject::cell){
                return;
            }
            else if (F_in::mesh_type() == MeshObject::edge){
                off[0] = 1;
                off[1] = 1;
                off[2] = 1;
                off[field_idx] = 0;
            }

            for (auto& n : block){
                n(tmp, 0) =
                0.5 * (
                    n(f_in,field_idx)
                  + n.at_offset(f_in, off[0], off[1], off[2], field_idx));
            }

            for (auto& n : block){
                n(f_in,field_idx) = n(tmp,0);
            }
        }

    }

    template<class Field, class Block>
    static float_type blockRootMeanSquare(Block& block) noexcept
    {
        float_type m = 0.0;
        float_type c = 0.0;

        for (auto& n : block)
        {
            float_type tmp=0.0;
            for (std::size_t field_idx = 0; field_idx < Field::nFields(); ++field_idx)
            {
                tmp += n(Field::tag(), field_idx)*n(Field::tag(), field_idx);
            }
            m+=tmp;
            c+=1.0;
        }
        return sqrt(m/c);
    }

    template<class Field, class Block>
    static float_type maxnorm(Block& block) noexcept
    {
        float_type m = 0.0;

        for (auto& n : block)
        {
            float_type tmp=0.0;
            for (std::size_t field_idx = 0; field_idx < Field::nFields(); ++field_idx)
            {
                tmp += n(Field::tag(), field_idx)*n(Field::tag(), field_idx);
            }
            tmp = sqrt(tmp);
            if (tmp > m) m = tmp;
        }
        return m;
    }

    template<class Source, class Dest, class Block>
    static void laplace(Block& block, float_type dx_level) noexcept
    {
        const auto     fac = 1.0 / (dx_level * dx_level);
        constexpr auto source = Source::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {
            n(dest) =
                -6.0 * n(source) + n.at_offset(source, 0, 0, -1) +
                n.at_offset(source, 0, 0, +1) + n.at_offset(source, 0, -1, 0) +
                n.at_offset(source, 0, +1, 0) + n.at_offset(source, -1, 0, 0) +
                n.at_offset(source, +1, 0, 0);
            n(dest) *= fac;
        }
    }

    template<class Source, class Dest, class Block,
        typename std::enable_if<(Source::mesh_type() == MeshObject::cell) &&
                                    (Dest::mesh_type() == MeshObject::face),
            void>::type* = nullptr>
    static void gradient(Block& block, float_type dx_level) noexcept
    {
        const auto     fac = 1.0 / dx_level;
        constexpr auto source = Source::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {
            n(dest, 0) = fac * (n(source) - n.at_offset(source, -1, 0, 0));
            n(dest, 1) = fac * (n(source) - n.at_offset(source, 0, -1, 0));
            n(dest, 2) = fac * (n(source) - n.at_offset(source, 0, 0, -1));
        }
    }

    template<class SourceTuple, class Dest, class Block,
        typename std::enable_if<(Dest::mesh_type() == MeshObject::cell) &&
                                    (SourceTuple::mesh_type() ==
                                        MeshObject::face),
            void>::type* = nullptr>
    static void divergence(Block& block, float_type dx_level) noexcept
    {
        const auto     fac = 1.0 / dx_level;
        constexpr auto source = SourceTuple::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {
            n(dest) = -n(source, 0) - n(source, 1) - n(source, 2) +
                      n.at_offset(source, 1, 0, 0, 0) +
                      n.at_offset(source, 0, 1, 0, 1) +
                      n.at_offset(source, 0, 0, 1, 2);
            n(dest) *= fac;
        }
    }

    template<class Source, class Dest, class Block,
        typename std::enable_if<(Source::mesh_type() == MeshObject::face) &&
                                    (Dest::mesh_type() == MeshObject::edge),
            void>::type* = nullptr>
    static void curl(Block& block, float_type dx_level) noexcept
    {
        const auto     fac = 1.0 / dx_level;
        constexpr auto source = Source::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {
            n(dest, 0) = n(source, 2) - n.at_offset(source, 0, -1, 0, 2) -
                         n(source, 1) + n.at_offset(source, 0, 0, -1, 1);
            n(dest, 0) *= fac;

            n(dest, 1) = n(source, 0) - n.at_offset(source, 0, 0, -1, 0) -
                         n(source, 2) + n.at_offset(source, -1, 0, 0, 2);
            n(dest, 1) *= fac;

            n(dest, 2) = n(source, 1) - n.at_offset(source, -1, 0, 0, 1) -
                         n(source, 0) + n.at_offset(source, 0, -1, 0, 0);
            n(dest, 2) *= fac;
        }
    }

    template<class Source, class Dest, class Block,
        typename std::enable_if<(Source::mesh_type() == MeshObject::edge) &&
                                    (Dest::mesh_type() == MeshObject::face),
            void>::type* = nullptr>
    static void curl_transpose(
        Block& block, float_type dx_level, float_type scale = 1.0) noexcept
    {
        const auto     fac = 1.0 / dx_level * scale;
        constexpr auto source = Source::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {
            n(dest, 0) = +n(source, 1) - n.at_offset(source, 0, 0, 1, 1) +
                         n.at_offset(source, 0, 1, 0, 2) - n(source, 2);
            n(dest, 0) *= fac;

            n(dest, 1) = +n(source, 2) - n.at_offset(source, 1, 0, 0, 2) +
                         n.at_offset(source, 0, 0, 1, 0) - n(source, 0);
            n(dest, 1) *= fac;

            n(dest, 2) = +n(source, 0) - n.at_offset(source, 0, 1, 0, 0) +
                         n.at_offset(source, 1, 0, 0, 1) - n(source, 1);
            n(dest, 2) *= fac;
        }
    }

    template<class Face, class Edge, class Dest, class Block,
        typename std::enable_if<(Face::mesh_type() == MeshObject::face) &&
                                    (Edge::mesh_type() == MeshObject::edge) &&
                                    (Dest::mesh_type() == MeshObject::face),
            void>::type* = nullptr>
    static void nonlinear(Block& block) noexcept
    {
        constexpr auto face = Face::tag();
        constexpr auto edge = Edge::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {
            //TODO: Can be done much better by getting the appropriate nodes
            //      directly
            n(dest, 0) = 0.25 * (+n.at_offset(edge, 0, 0, 0, 1) *
                                        (+n.at_offset(face, 0, 0, 0, 2) +
                                            n.at_offset(face, -1, 0, 0, 2)) +
                                    n.at_offset(edge, 0, 0, 1, 1) *
                                        (+n.at_offset(face, 0, 0, 1, 2) +
                                            n.at_offset(face, -1, 0, 1, 2)) -
                                    n.at_offset(edge, 0, 0, 0, 2) *
                                        (+n.at_offset(face, 0, 0, 0, 1) +
                                            n.at_offset(face, -1, 0, 0, 1)) -
                                    n.at_offset(edge, 0, 1, 0, 2) *
                                        (+n.at_offset(face, 0, 1, 0, 1) +
                                            n.at_offset(face, -1, 1, 0, 1)));

            n(dest, 1) = 0.25 * (+n.at_offset(edge, 0, 0, 0, 2) *
                                        (+n.at_offset(face, 0, 0, 0, 0) +
                                            n.at_offset(face, 0, -1, 0, 0)) +
                                    n.at_offset(edge, 1, 0, 0, 2) *
                                        (+n.at_offset(face, 1, 0, 0, 0) +
                                            n.at_offset(face, 1, -1, 0, 0)) -
                                    n.at_offset(edge, 0, 0, 0, 0) *
                                        (+n.at_offset(face, 0, 0, 0, 2) +
                                            n.at_offset(face, 0, -1, 0, 2)) -
                                    n.at_offset(edge, 0, 0, 1, 0) *
                                        (+n.at_offset(face, 0, 0, 1, 2) +
                                            n.at_offset(face, 0, -1, 1, 2)));
            n(dest, 2) = 0.25 * (+n.at_offset(edge, 0, 0, 0, 0) *
                                        (+n.at_offset(face, 0, 0, 0, 1) +
                                            n.at_offset(face, 0, 0, -1, 1)) +
                                    n.at_offset(edge, 0, 1, 0, 0) *
                                        (+n.at_offset(face, 0, 1, 0, 1) +
                                            n.at_offset(face, 0, 1, -1, 1)) -
                                    n.at_offset(edge, 0, 0, 0, 1) *
                                        (+n.at_offset(face, 0, 0, 0, 0) +
                                            n.at_offset(face, 0, 0, -1, 0)) -
                                    n.at_offset(edge, 1, 0, 0, 1) *
                                        (+n.at_offset(face, 1, 0, 0, 0) +
                                            n.at_offset(face, 1, 0, -1, 0)));
        }
    }
};
} // namespace domain
} // namespace iblgf

#endif
