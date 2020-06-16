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
    template<class Field, class Block>
    static float_type maxabs(Block& block) noexcept
    {
        float_type m = 0.0;

        auto& nodes_domain = block.nodes_domain();
        for (std::size_t field_idx = 0; field_idx < Field::nFields; ++field_idx)
        {
            for (auto it2 = nodes_domain.begin(); it2 != nodes_domain.end();
                 ++it2)
            {
                auto tmp = std::fabs(it2->template get<Field>(field_idx));
                if (tmp > m) m = tmp;
            }
        }
        return m;
    }

    template<class Source, class Dest, class Block>
    static void laplace(Block& block, float_type dx_level) noexcept
    {
        auto& nodes_domain = block.nodes_domain();

        const auto fac = 1.0 / (dx_level * dx_level);
        for (auto it2 = nodes_domain.begin(); it2 != nodes_domain.end(); ++it2)
        {
            it2->template get<Dest>() =
                -6.0 * it2->template get<Source>() +
                it2->template at_offset<Source>(0, 0, -1) +
                it2->template at_offset<Source>(0, 0, +1) +
                it2->template at_offset<Source>(0, -1, 0) +
                it2->template at_offset<Source>(0, +1, 0) +
                it2->template at_offset<Source>(-1, 0, 0) +
                it2->template at_offset<Source>(+1, 0, 0);
            it2->template get<Dest>() *= fac;
        }
    }

    template<class Source, class Dest, class Block,
        typename std::enable_if<(Source::mesh_type == MeshObject::cell) &&
                                    (Dest::mesh_type == MeshObject::face),
            void>::type* = nullptr>
    static void gradient(Block& block, float_type dx_level) noexcept
    {
        auto&      nodes_domain = block.nodes_domain();
        const auto fac = 1.0 / dx_level;
        for (auto it2 = nodes_domain.begin(); it2 != nodes_domain.end(); ++it2)
        {
            it2->template get<Dest>(0) =
                fac * (it2->template get<Source>() -
                          it2->template at_offset<Source>(-1, 0, 0));
            it2->template get<Dest>(1) =
                fac * (it2->template get<Source>() -
                          it2->template at_offset<Source>(0, -1, 0));
            it2->template get<Dest>(2) =
                fac * (it2->template get<Source>() -
                          it2->template at_offset<Source>(0, 0, -1));
        }
    }

    template<class SourceTuple, class Dest, class Block,
        typename std::enable_if<(Dest::mesh_type == MeshObject::cell) &&
                                    (SourceTuple::mesh_type ==
                                        MeshObject::face),
            void>::type* = nullptr>
    static void divergence(Block& block, float_type dx_level) noexcept
    {
        auto&      nodes_domain = block.nodes_domain();
        const auto fac = 1.0 / dx_level;
        for (auto it2 = nodes_domain.begin(); it2 != nodes_domain.end(); ++it2)
        {
            it2->template get<Dest>() =
                -it2->template get<SourceTuple>(0) -
                it2->template get<SourceTuple>(1) -
                it2->template get<SourceTuple>(2) +
                it2->template at_offset<SourceTuple>(1, 0, 0, 0) +
                it2->template at_offset<SourceTuple>(0, 1, 0, 1) +
                it2->template at_offset<SourceTuple>(0, 0, 1, 2);
            it2->template get<Dest>() *= fac;
        }
    }

    template<class Source, class Dest, class Block,
        typename std::enable_if<(Source::mesh_type == MeshObject::face) &&
                                    (Dest::mesh_type == MeshObject::edge),
            void>::type* = nullptr>
    static void curl(Block& block, float_type dx_level) noexcept
    {
        auto&      nodes_domain = block.nodes_domain();
        const auto fac = 1.0 / dx_level;
        for (auto it2 = nodes_domain.begin(); it2 != nodes_domain.end(); ++it2)
        {
            it2->template get<Dest>(0) =
                +it2->template get<Source>(2) -
                it2->template at_offset<Source>(0, -1, 0, 2) -
                it2->template get<Source>(1) +
                it2->template at_offset<Source>(0, 0, -1, 1);
            it2->template get<Dest>(0) *= fac;

            it2->template get<Dest>(1) =
                +it2->template get<Source>(0) -
                it2->template at_offset<Source>(0, 0, -1, 0) -
                it2->template get<Source>(2) +
                it2->template at_offset<Source>(-1, 0, 0, 2);
            it2->template get<Dest>(1) *= fac;

            it2->template get<Dest>(2) =
                +it2->template get<Source>(1) -
                it2->template at_offset<Source>(-1, 0, 0, 1) -
                it2->template get<Source>(0) +
                it2->template at_offset<Source>(0, -1, 0, 0);
            it2->template get<Dest>(2) *= fac;
        }
    }

    template<class Source, class Dest, class Block,
        typename std::enable_if<(Source::mesh_type == MeshObject::edge) &&
                                    (Dest::mesh_type == MeshObject::face),
            void>::type* = nullptr>
    static void curl_transpose(
        Block& block, float_type dx_level, float_type scale = 1.0) noexcept
    {
        auto&      nodes_domain = block.nodes_domain();
        const auto fac = 1.0 / dx_level * scale;
        for (auto it2 = nodes_domain.begin(); it2 != nodes_domain.end(); ++it2)
        {
            it2->template get<Dest>(0) =
                +it2->template get<Source>(1) -
                it2->template at_offset<Source>(0, 0, 1, 1) +
                it2->template at_offset<Source>(0, 1, 0, 2) -
                it2->template get<Source>(2);
            it2->template get<Dest>(0) *= fac;

            it2->template get<Dest>(1) =
                +it2->template get<Source>(2) -
                it2->template at_offset<Source>(1, 0, 0, 2) +
                it2->template at_offset<Source>(0, 0, 1, 0) -
                it2->template get<Source>(0);
            it2->template get<Dest>(1) *= fac;

            it2->template get<Dest>(2) =
                +it2->template get<Source>(0) -
                it2->template at_offset<Source>(0, 1, 0, 0) +
                it2->template at_offset<Source>(1, 0, 0, 1) -
                it2->template get<Source>(1);
            it2->template get<Dest>(2) *= fac;

            //it2->template get<Dest>(0)=
            //    +it2->template get<Source>(1)
            //    -it2->template at_offset<Source>(0,0, 1,1)
            //    +it2->template at_offset<Source>(0, 1,0,2)
            //    -it2->template get<Source>(2);
            //it2->template get<Dest>(0)*=fac;

            //it2->template get<Dest>(1)=
            //    +it2->template get<Source>(2)
            //    -it2->template at_offset<Source>(1,0,0,2)
            //    +it2->template at_offset<Source>(0,0,1,0)
            //    -it2->template get<Source>(0);
            //it2->template get<Dest>(1)*=fac;

            //it2->template get<Dest>(2)=
            //    +it2->template get<Source>(0)
            //    -it2->template at_offset<Source>(0,1,0,0)
            //    +it2->template at_offset<Source>(1,0,0,1)
            //    -it2->template get<Source>(1);
            //it2->template get<Dest>(2)*=fac;
        }
    }

    template<class Face, class Edge, class Dest, class Block,
        typename std::enable_if<(Face::mesh_type == MeshObject::face) &&
                                    (Edge::mesh_type == MeshObject::edge) &&
                                    (Dest::mesh_type == MeshObject::face),
            void>::type* = nullptr>
    static void nonlinear(Block& block) noexcept
    {
        auto& nodes_domain = block.nodes_domain();
        for (auto it2 = nodes_domain.begin(); it2 != nodes_domain.end(); ++it2)
        {
            //TODO: Can be done much better by getting the appropriate nodes
            //      directly
            it2->template get<Dest>(0) =
                0.25 * (+it2->template at_offset<Edge>(0, 0, 0, 1) *
                               (+it2->template at_offset<Face>(0, 0, 0, 2) +
                                   it2->template at_offset<Face>(-1, 0, 0, 2)) +
                           it2->template at_offset<Edge>(0, 0, 1, 1) *
                               (+it2->template at_offset<Face>(0, 0, 1, 2) +
                                   it2->template at_offset<Face>(-1, 0, 1, 2)) -
                           it2->template at_offset<Edge>(0, 0, 0, 2) *
                               (+it2->template at_offset<Face>(0, 0, 0, 1) +
                                   it2->template at_offset<Face>(-1, 0, 0, 1)) -
                           it2->template at_offset<Edge>(0, 1, 0, 2) *
                               (+it2->template at_offset<Face>(0, 1, 0, 1) +
                                   it2->template at_offset<Face>(-1, 1, 0, 1)));

            it2->template get<Dest>(1) =
                0.25 * (+it2->template at_offset<Edge>(0, 0, 0, 2) *
                               (+it2->template at_offset<Face>(0, 0, 0, 0) +
                                   it2->template at_offset<Face>(0, -1, 0, 0)) +
                           it2->template at_offset<Edge>(1, 0, 0, 2) *
                               (+it2->template at_offset<Face>(1, 0, 0, 0) +
                                   it2->template at_offset<Face>(1, -1, 0, 0)) -
                           it2->template at_offset<Edge>(0, 0, 0, 0) *
                               (+it2->template at_offset<Face>(0, 0, 0, 2) +
                                   it2->template at_offset<Face>(0, -1, 0, 2)) -
                           it2->template at_offset<Edge>(0, 0, 1, 0) *
                               (+it2->template at_offset<Face>(0, 0, 1, 2) +
                                   it2->template at_offset<Face>(0, -1, 1, 2)));
            it2->template get<Dest>(2) =
                0.25 * (+it2->template at_offset<Edge>(0, 0, 0, 0) *
                               (+it2->template at_offset<Face>(0, 0, 0, 1) +
                                   it2->template at_offset<Face>(0, 0, -1, 1)) +
                           it2->template at_offset<Edge>(0, 1, 0, 0) *
                               (+it2->template at_offset<Face>(0, 1, 0, 1) +
                                   it2->template at_offset<Face>(0, 1, -1, 1)) -
                           it2->template at_offset<Edge>(0, 0, 0, 1) *
                               (+it2->template at_offset<Face>(0, 0, 0, 0) +
                                   it2->template at_offset<Face>(0, 0, -1, 0)) -
                           it2->template at_offset<Edge>(1, 0, 0, 1) *
                               (+it2->template at_offset<Face>(1, 0, 0, 0) +
                                   it2->template at_offset<Face>(1, 0, -1, 0)));
        }
    }

  public:
};
} // namespace domain
} // namespace iblgf

#endif
