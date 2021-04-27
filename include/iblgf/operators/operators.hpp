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
#include <cmath>

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

  public: // DomainOprs

    template <typename F, class Domain>
    static void domainClean(Domain* domain)
    {
        for (auto it = domain->begin(); it != domain->end(); ++it)
        {
            if (!it->has_data() || !it->data().is_allocated()) continue;
            for (std::size_t field_idx = 0; field_idx < F::nFields();
                    ++field_idx)
            {
                auto& lin_data = it->data_r(F::tag(), field_idx).linalg_data();
                std::fill(lin_data.begin(), lin_data.end(), 0.0);
            }
        }
    }

    // TODO: move up_and_down
    template <typename F, class Domain>
    static void clean_ib_region_boundary(Domain* domain, int l, int clean_width=2) noexcept
    {
        for (auto it  = domain->begin(l);
                it != domain->end(l); ++it)
        {
            if(!it->locally_owned()) continue;
            if(!it->has_data() || !it->data().is_allocated()) continue;

            for(std::size_t i=0;i< it->num_neighbors();++i)
            {
                auto it2=it->neighbor(i);
                if ((!it2 || !it2->has_data()) || (!it2->is_ib()))
                {
                    for (std::size_t field_idx=0; field_idx<F::nFields(); ++field_idx)
                    {
                        auto& lin_data =
                            it->data_r(F::tag(), field_idx).linalg_data();

                        int N=it->data().descriptor().extent()[0];

                        // somehow we delete the outer 2 planes
                        if (i==4)
                            view(lin_data,xt::all(),xt::all(),xt::range(0,clean_width))  *= 0.0;
                        else if (i==10)
                            view(lin_data,xt::all(),xt::range(0,clean_width),xt::all())  *= 0.0;
                        else if (i==12)
                            view(lin_data,xt::range(0,clean_width),xt::all(),xt::all())  *= 0.0;
                        else if (i==14)
                            view(lin_data,xt::range(N+2-clean_width,N+3),xt::all(),xt::all())  *= 0.0;
                        else if (i==16)
                            view(lin_data,xt::all(),xt::range(N+2-clean_width,N+3),xt::all())  *= 0.0;
                        else if (i==22)
                            view(lin_data,xt::all(),xt::all(),xt::range(N+2-clean_width,N+3))  *= 0.0;
                    }
                }
            }
        }}

    template <typename F, class Domain>
    static void clean_leaf_correction_boundary(Domain* domain, int l, bool leaf_only_boundary=false, int clean_width=1) noexcept
    {
        for (auto it = domain->begin(l); it != domain->end(l); ++it)
        {
            if (!it->locally_owned())
            {
                if (!it->has_data() || !it->data().is_allocated()) continue;
                for (std::size_t field_idx = 0; field_idx < F::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(F::tag(), field_idx).linalg_data();
                    std::fill(lin_data.begin(), lin_data.end(), 0.0);
                }
            }
        }

        for (auto it = domain->begin(l); it != domain->end(l); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->has_data() || !it->data().is_allocated()) continue;

            if (leaf_only_boundary && (it->is_correction() || it->is_old_correction() ))
            {
                for (std::size_t field_idx = 0; field_idx < F::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(F::tag(), field_idx).linalg_data();
                    std::fill(lin_data.begin(), lin_data.end(), 0.0);
                }
            }
        }

        //---------------
        if (l==domain->tree()->base_level())

        for (auto it  = domain->begin(l);
                it != domain->end(l); ++it)
        {
            if(!it->locally_owned()) continue;
            if(!it->has_data() || !it->data().is_allocated()) continue;

            for(std::size_t i=0;i< it->num_neighbors();++i)
            {
                auto it2=it->neighbor(i);
                if ((!it2 || !it2->has_data()) || (leaf_only_boundary && (it2->is_correction() || it2->is_old_correction() )))
                {
                    for (std::size_t field_idx=0; field_idx<F::nFields(); ++field_idx)
                    {
                        auto& lin_data =
                            it->data_r(F::tag(), field_idx).linalg_data();

                        int N=it->data().descriptor().extent()[0];

                        // somehow we delete the outer 2 planes
                        if (i==4)
                            view(lin_data,xt::all(),xt::all(),xt::range(0,clean_width))  *= 0.0;
                        else if (i==10)
                            view(lin_data,xt::all(),xt::range(0,clean_width),xt::all())  *= 0.0;
                        else if (i==12)
                            view(lin_data,xt::range(0,clean_width),xt::all(),xt::all())  *= 0.0;
                        else if (i==14)
                            view(lin_data,xt::range(N+2-clean_width,N+3),xt::all(),xt::all())  *= 0.0;
                        else if (i==16)
                            view(lin_data,xt::all(),xt::range(N+2-clean_width,N+3),xt::all())  *= 0.0;
                        else if (i==22)
                            view(lin_data,xt::all(),xt::all(),xt::range(N+2-clean_width,N+3))  *= 0.0;
                    }
                }
            }
        }
    }

    template<class Source, class Target, class Domain>
    static void levelDivergence(Domain* domain, int l) noexcept
    {
        auto client = domain->decomposition().client();
        client->template buffer_exchange<Source>(l);
        const auto dx_base = domain->dx_base();

        for (auto it = domain->begin(l); it != domain->end(l); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->has_data() || !it->data().is_allocated()) continue;

            const auto dx_level = dx_base / std::pow(2,it->refinement_level());
            divergence<Source, Target>( it->data(), dx_level);
        }

        clean_leaf_correction_boundary<Target>(domain, l, true, 2);
    }

    template<class Source, class Target, class Domain>
    static void domainDivergence(Domain* domain) noexcept
    {
        auto client = domain->decomposition().client();

        //up_and_down<Source>();

        for (int l = domain->tree()->base_level();
             l < domain->tree()->depth(); ++l)
            levelDivergence<Source, Target>(domain, l);
    }

    template<class Source, class Target, class Domain>
    static void levelGradient(Domain* domain, int l) noexcept
    {
        auto client = domain->decomposition().client();
        //client->template buffer_exchange<Source>(l);
        const auto dx_base = domain->dx_base();

        for (auto it = domain->begin(l); it != domain->end(l); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->has_data() || !it->data().is_allocated()) continue;

            const auto dx_level = dx_base / std::pow(2,it->refinement_level());
            gradient<Source, Target>( it->data(), dx_level);
        }

        client->template buffer_exchange<Target>(l);
    }


    template<class Source, class Target, class Domain>
    static void domainGradient(Domain* domain, float_type _scale = 1.0) noexcept
    {
        //up_and_down<Source>();

        for (int l = domain->tree()->base_level();
                l < domain->tree()->depth(); ++l)

            levelGradient<Source, Target>(domain, l);
    }


    template<class Field, class Block>
    static void smooth2zero(Block& block, std::size_t ngb_idx) noexcept
    {

        //auto f =
        //    [](float_type x)
        //    {
        //        if (x>=1) return 1.0;
        //        if (x<=0) return 0.0;
        //        x = x - 0.3;

        //        float_type h1 = exp(-1/x);
        //        float_type h2 = exp(-1/(1 - x));

        //        return h1/(h1+h2);
        //    };


        auto f =
        [](float_type x)
        {
            const float_type fac=20.0;
            const float_type shift = 0.2;
            const float_type c = 1-(0.5 + 0.5 * tanh(fac*(1-shift)));

            return ( (0.5 + 0.5 * tanh(fac*(x-shift))) +c);
        };

        const std::size_t dim = 3;
        std::size_t x = ngb_idx % dim;
        std::size_t y = (ngb_idx/dim) % dim;
        std::size_t z = (ngb_idx/dim/dim) % dim;

        for (std::size_t field_idx = 0; field_idx < Field::nFields();
                ++field_idx)
        {
            for (auto& n: block.node_field())
            {
                auto pct  = n.local_pct();

                float_type square = 0.0;
                float_type c = 0;

                if (z==0){
                    square = std::max(square, f(pct[2]));
                    c+=1;
                }
                else if (z==(dim-1))
                {
                    square = std::max(square, f(1-pct[2]));
                    c+=1;
                }

                if (y==0){
                    square = std::max(square, f(pct[1]));
                    c+=1;
                }
                else if (y==(dim-1))
                {
                    square = std::max(square, f(1-pct[1]));
                    c+=1;
                }

                if (x==0){
                    square = std::max(square, f(pct[0]));
                    c+=1;
                }
                else if (x==(dim-1))
                {
                    square = std::max(square, f(1-pct[0]));
                    c+=1;
                }


                if (c>0)
                    n(Field::tag(), field_idx) = n(Field::tag(), field_idx) * square;


                //if      (z == 0)
                //{
                //    if (x != 0 && y!=0)
                //        n(Field::tag(), field_idx) = n(Field::tag(), field_idx) * f(pct[2]);
                //    else if (x==0 && y!=0)
                //        n(Field::tag(), field_idx) = n(Field::tag(), field_idx) * sqrt(f(pct[2])*f(pct[2])+ f(pct[0])* f(pct[0]));
                //    else if (x!=0 && y==0)
                //        n(Field::tag(), field_idx) = n(Field::tag(), field_idx) * sqrt(f(pct[2])*f(pct[2])+ f(pct[0])* f(pct[0]));
                //}
                //else if (y == 0)
                //    n(Field::tag(), field_idx) = n(Field::tag(), field_idx) * f(pct[1]);
                //else if (x == 0)
                //    n(Field::tag(), field_idx) = n(Field::tag(), field_idx) * f(pct[0]);
                //else if (x == (dim-1))
                //    n(Field::tag(), field_idx) = n(Field::tag(), field_idx) * f(1-pct[0]);
                //else if (y == (dim-1))
                //    n(Field::tag(), field_idx) = n(Field::tag(), field_idx) * f(1-pct[1]);
                //else if (z == (dim-1))
                //    n(Field::tag(), field_idx) = n(Field::tag(), field_idx) * f(1-pct[2]);
            }
        }
    }


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

    template<class U, class Block, class Coord, class Force, class DeltaFunc,
        typename std::enable_if<(U::mesh_type() == MeshObject::face), void>::type* = nullptr>
    static void ib_projection(Coord ib_coord, Force& f, Block& block, DeltaFunc& ddf)
    {
        constexpr auto u = U::tag();
        for (auto& node : block)
        {
            auto n_coord = node.level_coordinate();
            auto dist = n_coord - ib_coord;

            for (std::size_t field_idx=0; field_idx<U::nFields(); field_idx++)
            {
                decltype(ib_coord) off(0.5); off[field_idx] = 0.0; // face data location
                f[field_idx] += node(u, field_idx)  * ddf(dist+off);
            }
        }
    }


    template<class U, class Block, class Coord, class Force, class DeltaFunc,
        typename std::enable_if<(U::mesh_type() == MeshObject::face), void>::type* = nullptr>
    static void ib_smearing(Coord ib_coord, Force& f, Block& block, DeltaFunc& ddf, float_type factor=1.0)
    {
        constexpr auto u = U::tag();
        for (auto& node : block)
        {
            auto n_coord = node.level_coordinate();
            auto dist = n_coord - ib_coord;

            for (std::size_t field_idx=0; field_idx<U::nFields(); field_idx++)
            {
                decltype(ib_coord) off(0.5); off[field_idx] = 0.0; // face data location
                node(u, field_idx) += f[field_idx] * ddf(dist+off) * factor;
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

    template<typename Field, typename Domain, typename Func>
    static void add_field_expression(Domain* domain, Func& f, float_type t, float_type scale=1.0) noexcept
    {
        const auto dx_base = domain->dx_base();
        for (auto it = domain->begin(); it != domain->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;

            for (std::size_t field_idx = 0; field_idx < Field::nFields();
                    ++field_idx)
                for (auto& n:it->data().node_field())
                {
                    auto coord = n.global_coordinate()*dx_base;
                    n(Field::tag(), field_idx) += f(field_idx, t, coord)*scale;
                }
        }
    }

    template<typename From, typename To, typename Domain>
    static void add(Domain* domain, float_type scale = 1.0) noexcept
    {
        static_assert(From::nFields() == To::nFields(),
            "number of fields doesn't match when add");

        for (auto it = domain->begin(); it != domain->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;

            for (std::size_t field_idx = 0; field_idx < From::nFields();
                 ++field_idx)
            {
                for (auto& n:it->data().node_field())
                    n(To::tag(), field_idx) += n(From::tag(), field_idx) * scale;
            }
        }
    }


};
} // namespace domain
} // namespace iblgf

#endif
