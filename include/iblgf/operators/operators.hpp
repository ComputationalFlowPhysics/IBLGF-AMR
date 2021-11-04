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
#include <iblgf/solver/time_integration/HelmholtzFFT.hpp>
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
    template<typename F, class Domain>
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
    template<typename F, class Domain>
    static void clean_ib_region_boundary(Domain* domain, int l,
        int clean_width = 2) noexcept
    {
        for (auto it = domain->begin(l); it != domain->end(l); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->has_data() || !it->data().is_allocated()) continue;
            int Dim = domain->dimension();

            if (Dim == 2)
            {
                int idx2D[it->num_neighbors()];
                for (int i = 0; i < it->num_neighbors(); i++) { idx2D[i] = 1; }
                auto coord_it = it->tree_coordinate();
                for (std::size_t i = 0; i < it->num_neighbors(); ++i)
                {
                    auto it2 = it->neighbor(i);
                    if ((!it2 || !it2->has_data()) || (!it2->is_ib()))
                    {
                        continue;
                    }
                    auto cood = it2->tree_coordinate();
                    int  tmp = 0;

                    tmp += 3 * (((cood.y() - coord_it.y()) > 0) + 1) +
                           ((cood.x() - coord_it.x()) > 0) + 1;
                    idx2D[tmp] = -1;
                }

                for (std::size_t i = 0; i < it->num_neighbors(); ++i)
                {
                    if (idx2D[i] > 0)
                    {
                        for (std::size_t field_idx = 0;
                             field_idx < F::nFields(); ++field_idx)
                        {
                            auto& lin_data =
                                it->data_r(F::tag(), field_idx).linalg_data();

                            int N = it->data().descriptor().extent()[0];
                            if (i == 1)
                                view(lin_data, xt::all(),
                                    xt::range(0, clean_width)) *= 0.0;
                            else if (i == 3)
                                view(lin_data, xt::range(0, clean_width),
                                    xt::all()) *= 0.0;
                            else if (i == 5)
                                view(lin_data,
                                    xt::range(N + 2 - clean_width, N + 3),
                                    xt::all()) *= 0.0;
                            else if (i == 7)
                                view(lin_data, xt::all(),
                                    xt::range(N + 2 - clean_width, N + 3)) *=
                                    0.0;
                        }
                    }
                }
            }

            for (std::size_t i = 0; i < it->num_neighbors(); ++i)
            {
                auto it2 = it->neighbor(i);
                if ((!it2 || !it2->has_data()) || (!it2->is_ib()))
                {
                    for (std::size_t field_idx = 0; field_idx < F::nFields();
                         ++field_idx)
                    {
                        auto& lin_data =
                            it->data_r(F::tag(), field_idx).linalg_data();

                        int N = it->data().descriptor().extent()[0];

                        // somehow we delete the outer 2 planes
                        int Dim = domain->dimension();
                        if (Dim == 3)
                        {
                            if (i == 4)
                                view(lin_data, xt::all(), xt::all(),
                                    xt::range(0, clean_width)) *= 0.0;
                            else if (i == 10)
                                view(lin_data, xt::all(),
                                    xt::range(0, clean_width), xt::all()) *=
                                    0.0;
                            else if (i == 12)
                                view(lin_data, xt::range(0, clean_width),
                                    xt::all(), xt::all()) *= 0.0;
                            else if (i == 14)
                                view(lin_data,
                                    xt::range(N + 2 - clean_width, N + 3),
                                    xt::all(), xt::all()) *= 0.0;
                            else if (i == 16)
                                view(lin_data, xt::all(),
                                    xt::range(N + 2 - clean_width, N + 3),
                                    xt::all()) *= 0.0;
                            else if (i == 22)
                                view(lin_data, xt::all(), xt::all(),
                                    xt::range(N + 2 - clean_width, N + 3)) *=
                                    0.0;
                        }
                        /*if (Dim == 2) {
			if (i==1)
			    view(lin_data,xt::all(),xt::range(0,clean_width))  *= 0.0;
			else if (i==3)
			    view(lin_data,xt::range(0,clean_width),xt::all())  *= 0.0;
			else if (i==5)
			    view(lin_data,xt::range(N+2-clean_width,N+3),xt::all())  *= 0.0;
			else if (i==7)
			    view(lin_data,xt::all(),xt::range(N+2-clean_width,N+3))  *= 0.0;
			}*/
                    }
                }
            }
        }
    }

    template<typename F, class Domain>
    static void clean_leaf_correction_boundary(Domain* domain, int l,
        bool leaf_only_boundary = false, int clean_width = 1) noexcept
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

            if (leaf_only_boundary &&
                (it->is_correction() || it->is_old_correction()))
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
        if (l == domain->tree()->base_level())

            for (auto it = domain->begin(l); it != domain->end(l); ++it)
            {
                if (!it->locally_owned()) continue;
                if (!it->has_data() || !it->data().is_allocated()) continue;

                int Dim = domain->dimension();

                if (Dim == 2)
                {
                    int idx2D[it->num_neighbors()];
                    for (int i = 0; i < it->num_neighbors(); i++)
                    {
                        idx2D[i] = 1;
                    }
                    auto coord_it = it->tree_coordinate();
                    for (std::size_t i = 0; i < it->num_neighbors(); ++i)
                    {
                        auto it2 = it->neighbor(i);
                        if ((!it2 || !it2->has_data()) ||
                            (leaf_only_boundary &&
                                (it2->is_correction() ||
                                    it2->is_old_correction())))
                        {
                            continue;
                        }
                        auto cood = it2->tree_coordinate();
                        int  tmp = 0;
                        tmp += 3 * (cood.y() - coord_it.y() + 1) + cood.x() -
                               coord_it.x() + 1;

                        idx2D[tmp] = -1;
                    }

                    for (std::size_t i = 0; i < it->num_neighbors(); ++i)
                    {
                        if (idx2D[i] > 0)
                        {
                            for (std::size_t field_idx = 0;
                                 field_idx < F::nFields(); ++field_idx)
                            {
                                auto& lin_data = it->data_r(F::tag(), field_idx)
                                                     .linalg_data();

                                int N = it->data().descriptor().extent()[0];
                                if (i == 1)
                                    view(lin_data, xt::all(),
                                        xt::range(0, clean_width)) *= 0.0;
                                else if (i == 3)
                                    view(lin_data, xt::range(0, clean_width),
                                        xt::all()) *= 0.0;
                                else if (i == 5)
                                    view(lin_data,
                                        xt::range(N + 2 - clean_width, N + 3),
                                        xt::all()) *= 0.0;
                                else if (i == 7)
                                    view(lin_data, xt::all(),
                                        xt::range(N + 2 - clean_width,
                                            N + 3)) *= 0.0;
                            }
                        }
                    }
                }

                for (std::size_t i = 0; i < it->num_neighbors(); ++i)
                {
                    auto it2 = it->neighbor(i);
                    if ((!it2 || !it2->has_data()) ||
                        (leaf_only_boundary &&
                            (it2->is_correction() || it2->is_old_correction())))
                    {
                        for (std::size_t field_idx = 0;
                             field_idx < F::nFields(); ++field_idx)
                        {
                            auto& lin_data =
                                it->data_r(F::tag(), field_idx).linalg_data();

                            int N = it->data().descriptor().extent()[0];

                            // somehow we delete the outer 2 planes
                            int Dim = domain->dimension();
                            if (Dim == 3)
                            {
                                if (i == 4)
                                    view(lin_data, xt::all(), xt::all(),
                                        xt::range(0, clean_width)) *= 0.0;
                                else if (i == 10)
                                    view(lin_data, xt::all(),
                                        xt::range(0, clean_width), xt::all()) *=
                                        0.0;
                                else if (i == 12)
                                    view(lin_data, xt::range(0, clean_width),
                                        xt::all(), xt::all()) *= 0.0;
                                else if (i == 14)
                                    view(lin_data,
                                        xt::range(N + 2 - clean_width, N + 3),
                                        xt::all(), xt::all()) *= 0.0;
                                else if (i == 16)
                                    view(lin_data, xt::all(),
                                        xt::range(N + 2 - clean_width, N + 3),
                                        xt::all()) *= 0.0;
                                else if (i == 22)
                                    view(lin_data, xt::all(), xt::all(),
                                        xt::range(N + 2 - clean_width,
                                            N + 3)) *= 0.0;
                            }
                            /*if (Dim == 2) {
			if (i==1)
			    view(lin_data,xt::all(),xt::range(0,clean_width))  *= 0.0;
			else if (i==3)
			    view(lin_data,xt::range(0,clean_width),xt::all())  *= 0.0;
			else if (i==5)
			    view(lin_data,xt::range(N+2-clean_width,N+3),xt::all())  *= 0.0;
			else if (i==7)
			    view(lin_data,xt::all(),xt::range(N+2-clean_width,N+3))  *= 0.0;
			}*/
                            /*if (i==4)
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
                            view(lin_data,xt::all(),xt::all(),xt::range(N+2-clean_width,N+3))  *= 0.0;*/
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

            const auto dx_level = dx_base / std::pow(2, it->refinement_level());
            divergence<Source, Target>(it->data(), dx_level);
        }

        clean_leaf_correction_boundary<Target>(domain, l, true, 2);
    }


    template<class Source, class Target, class Domain>
    static void levelDivergence_helmholtz_complex(Domain* domain, int l, int N_modes, float_type c) noexcept
    {
        auto client = domain->decomposition().client();
        client->template buffer_exchange<Source>(l);
        const auto dx_base = domain->dx_base();

        for (auto it = domain->begin(l); it != domain->end(l); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->has_data() || !it->data().is_allocated()) continue;

            const auto dx_level = dx_base / std::pow(2, it->refinement_level());
            divergence_helmholtz_complex<Source, Target>(it->data(), dx_level, N_modes, c);
        }

        clean_leaf_correction_boundary<Target>(domain, l, true, 2);
    }

    template<class Source, class Target, class Domain>
    static void domainDivergence(Domain* domain) noexcept
    {
        auto client = domain->decomposition().client();

        //up_and_down<Source>();

        for (int l = domain->tree()->base_level(); l < domain->tree()->depth();
             ++l)
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

            const auto dx_level = dx_base / std::pow(2, it->refinement_level());
            gradient<Source, Target>(it->data(), dx_level);
        }

        client->template buffer_exchange<Target>(l);
    }

    template<class Source, class Target, class Domain>
    static void levelGradient_helmholtz_complex(Domain* domain, int l, int N_modes, float_type c) noexcept
    {
        auto client = domain->decomposition().client();
        //client->template buffer_exchange<Source>(l);
        const auto dx_base = domain->dx_base();

        for (auto it = domain->begin(l); it != domain->end(l); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->has_data() || !it->data().is_allocated()) continue;

            const auto dx_level = dx_base / std::pow(2, it->refinement_level());
            gradient_helmholtz_complex<Source, Target>(it->data(), dx_level, N_modes, c);
        }

        client->template buffer_exchange<Target>(l);
    }

    template<class Source, class Target, class Domain>
    static void domainGradient(Domain* domain, float_type _scale = 1.0) noexcept
    {
        //up_and_down<Source>();

        for (int l = domain->tree()->base_level(); l < domain->tree()->depth();
             ++l)

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

        auto f = [](float_type x)
        {
            const float_type fac = 10.0;
            const float_type shift = 0.2;
            const float_type c = 1 - (0.5 + 0.5 * tanh(fac * (1 - shift)));

            return ((0.5 + 0.5 * tanh(fac * (x - shift))) + c);
        };

        const std::size_t dim = 3;
        std::size_t       x = ngb_idx % dim;
        std::size_t       y = (ngb_idx / dim) % dim;
        std::size_t       z = (ngb_idx / dim / dim) % dim;

        for (std::size_t field_idx = 0; field_idx < Field::nFields();
             ++field_idx)
        {
            for (auto& n : block.node_field())
            {
                auto pct = n.local_pct();
                int  dimension = pct.size();

                float_type square = 0.0;
                float_type c = 0;

                if ((z == 0) && (dimension == 3))
                {
                    square = std::max(square, f(pct[2]));
                    c += 1;
                }
                else if ((z == (dim - 1)) && (dimension == 3))
                {
                    square = std::max(square, f(1 - pct[2]));
                    c += 1;
                }

                if (y == 0)
                {
                    square = std::max(square, f(pct[1]));
                    c += 1;
                }
                else if (y == (dim - 1))
                {
                    square = std::max(square, f(1 - pct[1]));
                    c += 1;
                }

                if (x == 0)
                {
                    square = std::max(square, f(pct[0]));
                    c += 1;
                }
                else if (x == (dim - 1))
                {
                    square = std::max(square, f(1 - pct[0]));
                    c += 1;
                }

                if (c > 0)
                    n(Field::tag(), field_idx) =
                        n(Field::tag(), field_idx) * square;

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

        for (std::size_t field_idx = 0; field_idx < F_in::nFields();
             ++field_idx)
        {
            std::array<int, 3> off{{0, 0, 0}};

            if (F_in::mesh_type() == MeshObject::face) { off[field_idx] = 1; }
            else if (F_in::mesh_type() == MeshObject::cell)
            {
                return;
            }
            else if (F_in::mesh_type() == MeshObject::edge)
            {
                off[0] = 1;
                off[1] = 1;
                off[2] = 1;
                off[field_idx] = 0;
            }

            for (auto& n : block)
            {
                auto pct = n.local_pct();
                int  dimension = pct.size();
                if (dimension == 3)
                {
                    n(tmp, 0) = 0.5 * (n(f_in, field_idx) +
                                          n.at_offset(f_in, off[0], off[1],
                                              off[2], field_idx));
                }
                if (dimension == 2)
                {
                    if (F_in::mesh_type() == MeshObject::edge)
                    {
                        n(tmp, 0) =
                            0.5 * (n(f_in, field_idx) +
                                      n.at_offset(f_in, 1, 1, field_idx));
                    }
                    else
                    {
                        n(tmp, 0) = 0.5 * (n(f_in, field_idx) +
                                              n.at_offset(f_in, off[0], off[1],
                                                  field_idx));
                    }
                }
            }

            for (auto& n : block) { n(f_in, field_idx) = n(tmp, 0); }
        }
    }

    template<class U, class Block, class Coord, class Force, class DeltaFunc,
        typename std::enable_if<(U::mesh_type() == MeshObject::face),
            void>::type* = nullptr>
    static void ib_projection(Coord ib_coord, Force& f, Block& block,
        DeltaFunc& ddf)
    {
        constexpr auto u = U::tag();
        for (auto& node : block)
        {
            auto n_coord = node.level_coordinate();
            auto dist = n_coord - ib_coord;

            for (std::size_t field_idx = 0; field_idx < U::nFields();
                 field_idx++)
            {
                decltype(ib_coord) off(0.5);
                off[field_idx] = 0.0; // face data location
                f[field_idx] += node(u, field_idx) * ddf(dist + off);
            }
        }
    }

    template<class U, class Block, class Coord, class Force, class DeltaFunc,
        typename std::enable_if<(U::mesh_type() == MeshObject::face),
            void>::type* = nullptr>
    static void ib_projection_helmholtz(Coord ib_coord, Force& f, Block& block,
        DeltaFunc& ddf, int sep)
    {
        constexpr auto u = U::tag();
        for (auto& node : block)
        {
            auto n_coord = node.level_coordinate();
            auto dist = n_coord - ib_coord;

            for (std::size_t field_idx = 0; field_idx < U::nFields();
                 field_idx++)
            {
                decltype(ib_coord) off(0.5);
                off[field_idx/sep] = 0.0; // face data location
                f[field_idx] += node(u, field_idx) * ddf(dist + off);
            }
        }
    }

    template<class U, class Block, class Coord, class Force, class DeltaFunc,
        typename std::enable_if<(U::mesh_type() == MeshObject::face),
            void>::type* = nullptr>
    static void ib_smearing(Coord ib_coord, Force& f, Block& block,
        DeltaFunc& ddf, float_type factor = 1.0)
    {
        constexpr auto u = U::tag();
        for (auto& node : block)
        {
            auto n_coord = node.level_coordinate();
            auto dist = n_coord - ib_coord;

            for (std::size_t field_idx = 0; field_idx < U::nFields();
                 field_idx++)
            {
                decltype(ib_coord) off(0.5);
                off[field_idx] = 0.0; // face data location
                node(u, field_idx) += f[field_idx] * ddf(dist + off) * factor;
            }
        }
    }

    template<class U, class Block, class Coord, class Force, class DeltaFunc,
        typename std::enable_if<(U::mesh_type() == MeshObject::face),
            void>::type* = nullptr>
    static void ib_smearing_helmholtz(Coord ib_coord, Force& f, Block& block,
        DeltaFunc& ddf, int sep, float_type factor = 1.0)
    {
        constexpr auto u = U::tag();
        for (auto& node : block)
        {
            auto n_coord = node.level_coordinate();
            auto dist = n_coord - ib_coord;

            for (std::size_t field_idx = 0; field_idx < U::nFields();
                 field_idx++)
            {
                decltype(ib_coord) off(0.5);
                off[field_idx/sep] = 0.0; // face data location
                node(u, field_idx) += f[field_idx] * ddf(dist + off) * factor;
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
            float_type tmp = 0.0;
            for (std::size_t field_idx = 0; field_idx < Field::nFields();
                 ++field_idx)
            {
                tmp += n(Field::tag(), field_idx) * n(Field::tag(), field_idx);
            }
            m += tmp;
            c += 1.0;
        }
        return sqrt(m / c);
    }

    template<class Field, class Block>
    static float_type maxnorm(Block& block) noexcept
    {
        float_type m = 0.0;

        for (auto& n : block)
        {
            float_type tmp = 0.0;
            for (std::size_t field_idx = 0; field_idx < Field::nFields();
                 ++field_idx)
            {
                tmp += n(Field::tag(), field_idx) * n(Field::tag(), field_idx);
            }
            tmp = sqrt(tmp);
            if (tmp > m) m = tmp;
        }
        return m;
    }

    template<class Field, class Block>
    static float_type maxabs(Block& block) noexcept
    {
        float_type m = 0.0;
        for (std::size_t field_idx = 0; field_idx < Field::nFields();
             ++field_idx)
        {
            for (auto& n : block)
            {
                auto tmp = std::fabs(n(Field::tag(), field_idx));
                if (tmp > m) m = tmp;
            }
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
            auto pct = n.local_pct();
            int  dimension = pct.size();
            if (dimension == 3)
            {
                n(dest) = -6.0 * n(source) + n.at_offset(source, 0, 0, -1) +
                          n.at_offset(source, 0, 0, +1) +
                          n.at_offset(source, 0, -1, 0) +
                          n.at_offset(source, 0, +1, 0) +
                          n.at_offset(source, -1, 0, 0) +
                          n.at_offset(source, +1, 0, 0);
            }
            else
            {
                n(dest) = -4.0 * n(source) + n.at_offset(source, 0, -1) +
                          n.at_offset(source, 0, +1) +
                          n.at_offset(source, -1, 0) +
                          n.at_offset(source, +1, 0);
            }

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
            auto pct = n.local_pct();
            int  dimension = pct.size();
            if (dimension == 3)
            {
                n(dest, 0) = fac * (n(source) - n.at_offset(source, -1, 0, 0));
                n(dest, 1) = fac * (n(source) - n.at_offset(source, 0, -1, 0));
                n(dest, 2) = fac * (n(source) - n.at_offset(source, 0, 0, -1));
            }
            else
            {
                n(dest, 0) = fac * (n(source) - n.at_offset(source, -1, 0));
                n(dest, 1) = fac * (n(source) - n.at_offset(source, 0, -1));
            }
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
            auto pct = n.local_pct();
            int  dimension = pct.size();
            if (dimension == 3)
            {
                n(dest) = -n(source, 0) - n(source, 1) - n(source, 2) +
                          n.at_offset(source, 1, 0, 0, 0) +
                          n.at_offset(source, 0, 1, 0, 1) +
                          n.at_offset(source, 0, 0, 1, 2);
            }
            else
            {
                n(dest) = -n(source, 0) - n(source, 1) +
                          n.at_offset(source, 1, 0, 0) +
                          n.at_offset(source, 0, 1, 1);
            }
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
            auto pct = n.local_pct();
            int  dimension = pct.size();
            if (dimension == 3)
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
            else
            {
                n(dest, 0) = n(source, 1) - n.at_offset(source, -1, 0, 1) -
                             n(source, 0) + n.at_offset(source, 0, -1, 0);
                n(dest, 0) *= fac;
            }
        }
    }


    template<class Source, class Dest, class Block>
    static void laplace_helmholtz_complex(Block& block, float_type dx_level,
        int                                   N_modes,
        float_type                            c,
        int                                   PREFAC = 2) noexcept
    {
        const auto     fac = 1.0 / (dx_level * dx_level);
        constexpr auto source = Source::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {

            int sep = N_modes*PREFAC;
            for (int i = 0; i < sep ; i++) {
                n(dest, i) = -4.0 * n(source, i) + n.at_offset(source, 0, -1, i) +
                          n.at_offset(source, 0, +1, i) +
                          n.at_offset(source, -1, 0, i) +
                          n.at_offset(source, +1, 0, i);

                n(dest, i) *= fac;
            }
            
            for (int i = 0; i < N_modes; i++) { 
                float_type omega = 2.0*M_PI*static_cast<float_type>(i)/c;

                n(dest, i * 2)     -= n(source, i * 2    ) * omega * omega;
                n(dest, i * 2 + 1) -= n(source, i * 2 + 1) * omega * omega;

            }            
        }
    }


    template<class Source, class Dest, class Block,
        typename std::enable_if<(Source::mesh_type() == MeshObject::cell) &&
                                    (Dest::mesh_type() == MeshObject::face),
            void>::type* = nullptr>
    static void gradient_helmholtz_complex(Block& block, float_type dx_level,
        int                                   N_modes,
        float_type                            c,
        int                                   PREFAC = 2) noexcept
    {
        const auto     fac = 1.0 / dx_level;
        constexpr auto source = Source::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {


            int sep = N_modes*PREFAC;
            for (int i = 0; i < sep ; i++) {
                n(dest, 0 * sep + i) =  fac * (n(source, i) - n.at_offset(source, -1, 0, i));

                n(dest, 1 * sep + i) =  fac * (n(source, i) - n.at_offset(source, 0, -1, i));
            }

            for (int i = 0; i < N_modes; i++) { 
                float_type omega = 2.0*M_PI*static_cast<float_type>(i)/c;

                n(dest, 2 * sep + i * 2)     = -n(source, i * 2 + 1) * omega;
                n(dest, 2 * sep + i * 2 + 1) =  n(source, i * 2    ) * omega;

            }

        }
    }



    template<class SourceTuple, class Dest, class Block,
        typename std::enable_if<(Dest::mesh_type() == MeshObject::cell) &&
                                    (SourceTuple::mesh_type() ==
                                        MeshObject::face),
            void>::type* = nullptr>
    static void divergence_helmholtz_complex(Block& block, float_type dx_level,
        int                                   N_modes,
        float_type                            c,
        int                                   PREFAC = 2) noexcept
    {
        const auto     fac = 1.0 / dx_level;
        constexpr auto source = SourceTuple::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {


            int sep = N_modes*PREFAC;
            for (int i = 0; i < sep ; i++) {
                n(dest, i) =  -n(source, 0 * sep + i) - n(source, 1 * sep + i) +
                          n.at_offset(source, 1, 0, 0 * sep + i) +
                          n.at_offset(source, 0, 1, 1 * sep + i);

                n(dest, i) *= fac;
            }

            for (int i = 0; i < N_modes; i++) { 
                float_type omega = 2.0*M_PI*static_cast<float_type>(i)/c;

                n(dest, i * 2)     -= n(source, 2 * sep + i * 2 + 1) * omega;
                n(dest, i * 2 + 1) += n(source, 2 * sep + i * 2    ) * omega;

            }

        }
    }


    template<class Source, class Dest, class Block,
        typename std::enable_if<(Source::mesh_type() == MeshObject::face) &&
                                    (Dest::mesh_type() == MeshObject::edge),
            void>::type* = nullptr>
    static void curl_helmholtz_complex(Block& block,
        float_type                            dx_level,
        int                                   N_modes,
        float_type                            c,
        int                                   PREFAC = 2) noexcept
    {
        const auto     fac = 1.0 / dx_level;
        constexpr auto source = Source::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {


            int sep = N_modes*PREFAC;
            for (int i = 0; i < (N_modes * PREFAC) ; i++) {
                n(dest, 0 * sep + i) =   n(source, 2 * sep + i) - n.at_offset(source, 0, -1, 2 * sep + i);
                n(dest, 0 * sep + i) *= fac;

                n(dest, 1 * sep + i) =  -n(source, 2 * sep + i) + n.at_offset(source, -1, 0, 2 * sep + i);
                n(dest, 1 * sep + i) *= fac;

                n(dest, 2 * sep + i) = 
                n(source, 1 * sep + i) - n.at_offset(source, -1, 0, 1 * sep + i) -
                n(source, 0 * sep + i) + n.at_offset(source, 0, -1, 0 * sep + i);
                n(dest, 2 * sep + i) *= fac;
            }

            for (int i = 0; i < (N_modes) ; i++) { 
                float_type omega = 2.0*M_PI*static_cast<float_type>(i)/c;
                n(dest, 0 * sep + i * 2)     += n(source, 1 * sep + i * 2 + 1) * omega;
                n(dest, 0 * sep + i * 2 + 1) -= n(source, 1 * sep + i * 2    ) * omega;

                n(dest, 1 * sep + i * 2)     -= n(source, 0 * sep + i * 2 + 1) * omega;
                n(dest, 1 * sep + i * 2 + 1) += n(source, 0 * sep + i * 2    ) * omega;

            }

        }
    }


    template<class Source, class Dest, class Block,
        typename std::enable_if<(Source::mesh_type() == MeshObject::edge) &&
                                    (Dest::mesh_type() == MeshObject::face),
            void>::type* = nullptr>
    static void curl_transpose_helmholtz_complex(Block& block,
        float_type                            dx_level,
        int                                   N_modes,
        float_type                            c,
        float_type                            scale = 1.0,
        int                                   PREFAC = 2) noexcept
    {
        const auto     fac = 1.0 / dx_level *scale;
        constexpr auto source = Source::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {


            int sep = N_modes*PREFAC;
            for (int i = 0; i < (N_modes * PREFAC) ; i++) {
                n(dest, 0 * sep + i) =  - n(source, 2 * sep + i) + n.at_offset(source, 0, 1, 2 * sep + i);
                n(dest, 0 * sep + i) *= fac;

                n(dest, 1 * sep + i) =  n(source, 2 * sep + i) - n.at_offset(source, 1, 0, 2 * sep + i);
                n(dest, 1 * sep + i) *= fac;

                n(dest, 2 * sep + i) = 
                - n(source, 1 * sep + i) + n.at_offset(source, 1, 0, 1 * sep + i) +
                n(source, 0 * sep + i) - n.at_offset(source, 0, 1, 0 * sep + i);
                n(dest, 2 * sep + i) *= fac;
            }

            for (int i = 0; i < (N_modes) ; i++) { 
                float_type omega = 2.0*M_PI*static_cast<float_type>(i)/c;
                n(dest, 0 * sep + i * 2)     += n(source, 1 * sep + i * 2 + 1) * omega * scale;
                n(dest, 0 * sep + i * 2 + 1) -= n(source, 1 * sep + i * 2    ) * omega * scale;

                n(dest, 1 * sep + i * 2)     -= n(source, 0 * sep + i * 2 + 1) * omega * scale;
                n(dest, 1 * sep + i * 2 + 1) += n(source, 0 * sep + i * 2    ) * omega * scale;

            }

        }
    }

    template<class Source, class Dest, class Block,
        typename std::enable_if<(Source::mesh_type() == MeshObject::face) &&
                                    (Dest::mesh_type() == MeshObject::edge),
            void>::type* = nullptr>
    static void curl_helmholtz(Block& block, float_type dx_level, int N_modes, float_type dx_fine, int PREFAC = 3) noexcept
    {
        const auto     fac = 1.0 / dx_level;
        constexpr auto source = Source::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {
            int sep = N_modes*PREFAC;
            for (int i = 1; i < (N_modes * PREFAC) ; i++) {
                n(dest, 0 * sep + i) = 
                (n(source, 2 * sep + i) - n.at_offset(source, 0, -1, 2 * sep + i)) / dx_level -
                //(n(source, 1 * sep + i) - n.at_offset(source, 0, 0,  1 * sep + i - 1)) / dx_fine;
                (n(source, 1 * sep + i) - n(source, 1 * sep + i - 1)) / dx_fine;
                //n(dest, 0 * sep + i) *= fac;

                n(dest, 1 * sep + i) = 
                (n(source, 0 * sep + i) - n.at_offset(source, 0, 0,  0 * sep + i - 1)) / dx_fine -
                (n(source, 2 * sep + i) - n.at_offset(source, -1, 0, 2 * sep + i)) / dx_level;
                //n(dest, 1 * sep + i) *= fac;

                n(dest, 2 * sep + i) = 
                n(source, 1 * sep + i) - n.at_offset(source, -1, 0, 1 * sep + i) -
                n(source, 0 * sep + i) + n.at_offset(source, 0, -1, 0 * sep + i);
                n(dest, 2 * sep + i) *= fac;
            }

            //i = 0 case
            n(dest, 0 * sep) = (n(source, 2 * sep) -
                               n.at_offset(source, 0, -1, 2 * sep)) / dx_level -
                               (n(source, 1 * sep) -
                               n.at_offset(source, 0, 0, 1 * sep + sep - 1)) / dx_fine;
            //n(dest, 0 * sep) *= fac;

            n(dest, 1 * sep) = (n(source, 0 * sep) -
                               n.at_offset(source, 0, 0, 0 * sep + sep - 1)) / dx_fine -
                               (n(source, 2 * sep) -
                               n.at_offset(source, -1, 0, 2 * sep)) / dx_level;
            //n(dest, 1 * sep) *= fac;

            n(dest, 2 * sep) = (n(source, 1 * sep) -
                               n.at_offset(source, -1, 0, 1 * sep)) / dx_level -
                               (n(source, 0 * sep) -
                               n.at_offset(source, 0, -1, 0 * sep)) / dx_level;
            //n(dest, 2 * sep) *= fac;

            /*//i = sep - 1 case
            n(dest, 0 * sep + sep - 1) = 
                n(source, 2 * sep + sep - 1) - n.at_offset(source, 0, -1, 2 * sep + sep - 1) -
                n(source, 1 * sep + sep - 1) + n.at_offset(source, 0, 0,  1 * sep + sep - 1 - 1);
                n(dest, 0 * sep + sep - 1) *= fac;

                n(dest, 1 * sep + i) = 
                n(source, 0 * sep + i) - n.at_offset(source, 0, 0,   0 * sep + sep - 1 - 1) -
                n(source, 2 * sep + i) + n.at_offset(source, -1, 0,  2 * sep + sep - 1);
                n(dest, 1 * sep + i) *= fac;

                n(dest, 2 * sep + i) = 
                n(source, 1 * sep + i) - n.at_offset(source, -1, 0, 1 * sep + sep - 1) -
                n(source, 0 * sep + i) + n.at_offset(source, 0, -1, 0 * sep + sep - 1);
                n(dest, 2) *= fac;*/
        }
    }

    template<class Source, class Dest, class Block,
        typename std::enable_if<(Source::mesh_type() == MeshObject::edge) &&
                                    (Dest::mesh_type() == MeshObject::face),
            void>::type* = nullptr>
    static void curl_transpose(Block& block, float_type dx_level,
        float_type scale = 1.0) noexcept
    {
        const auto     fac = 1.0 / dx_level * scale;
        constexpr auto source = Source::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {
            auto pct = n.local_pct();
            int  dimension = pct.size();
            if (dimension == 3)
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
            else
            {
                n(dest, 0) = -n(source) + n.at_offset(source, 0, 1);
                n(dest, 0) *= fac;
                n(dest, 1) = -n.at_offset(source, 1, 0) + n(source);
                n(dest, 1) *= fac;
            }
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
            auto pct = n.local_pct();
            int  dimension = pct.size();
            if (dimension == 3)
            {
                n(dest, 0) =
                    0.25 * (+n.at_offset(edge, 0, 0, 0, 1) *
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

                n(dest, 1) =
                    0.25 * (+n.at_offset(edge, 0, 0, 0, 2) *
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
                n(dest, 2) =
                    0.25 * (+n.at_offset(edge, 0, 0, 0, 0) *
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
            else
            {
                n(dest, 0) = 0.25 * (-n.at_offset(edge, 0, 0, 0) *
                                            (+n.at_offset(face, 0, 0, 1) +
                                                n.at_offset(face, -1, 0, 1)) -
                                        n.at_offset(edge, 0, 1, 0) *
                                            (+n.at_offset(face, 0, 1, 1) +
                                                n.at_offset(face, -1, 1, 1)));

                n(dest, 1) = 0.25 * (+n.at_offset(edge, 0, 0, 0) *
                                            (+n.at_offset(face, 0, 0, 0) +
                                                n.at_offset(face, 0, -1, 0)) +
                                        n.at_offset(edge, 1, 0, 0) *
                                            (+n.at_offset(face, 1, 0, 0) +
                                                n.at_offset(face, 1, -1, 0)));
            }
        }
    }

    template<class Face, class Edge, class Dest, class Block,
        typename std::enable_if<(Face::mesh_type() == MeshObject::face) &&
                                    (Edge::mesh_type() == MeshObject::edge) &&
                                    (Dest::mesh_type() == MeshObject::face),
            void>::type* = nullptr>
    static void nonlinear_helmholtz(Block& block, int N_modes, int PREFAC = 3) noexcept
    {
        constexpr auto face = Face::tag();
        constexpr auto edge = Edge::tag();
        constexpr auto dest = Dest::tag();
        for (auto& n : block)
        {
            //TODO: Can be done much better by getting the appropriate nodes
            //      directly

            int sep = N_modes * PREFAC;
            int x_s = 0;
            int y_s = sep;
            int z_s = sep*2;
            for (int i = 1; i < (N_modes * PREFAC - 1); i++)
            {
                int x_p = x_s + i;
                int y_p = y_s + i;
                int z_p = z_s + i;
                /*n(dest, 0 * sep + i) =
                    0.25 *
                    (+n.at_offset(edge, 0, 0, 1 * sep + i) *
                            (+n.at_offset(face, 0, 0, 2 * sep + i) +
                                n.at_offset(face, -1, 0, 2 * sep + i)) +
                        n.at_offset(edge, 0, 0, 1 * sep + i + 1) *
                            (+n.at_offset(face, 0, 0, 2 * sep + i + 1) +
                                n.at_offset(face, -1, 0, 2 * sep + i + 1)) -
                        n.at_offset(edge, 0, 0, 2 * sep + i) *
                            (+n.at_offset(face, 0, 0, 1 * sep + i) +
                                n.at_offset(face, -1, 0, 1 * sep + i)) -
                        n.at_offset(edge, 0, 1, 2 * sep + i) *
                            (+n.at_offset(face, 0, 1, 1 * sep + i) +
                                n.at_offset(face, -1, 1, 1 * sep + i)));

                n(dest, 1 * sep + i) =
                    0.25 *
                    (+n.at_offset(edge, 0, 0, 2 * sep + i) *
                            (+n.at_offset(face, 0, 0, 0 * sep + i) +
                                n.at_offset(face, 0, -1, 0 * sep + i)) +
                        n.at_offset(edge, 1, 0, 2 * sep + i) *
                            (+n.at_offset(face, 1, 0, 0 * sep + i) +
                                n.at_offset(face, 1, -1, 0 * sep + i)) -
                        n.at_offset(edge, 0, 0, 0 * sep + i) *
                            (+n.at_offset(face, 0, 0, 2 * sep + i) +
                                n.at_offset(face, 0, -1, 2 * sep + i)) -
                        n.at_offset(edge, 0, 0, 0 * sep + i + 1) *
                            (+n.at_offset(face, 0, 0, 2 * sep + i + 1) +
                                n.at_offset(face, 0, -1, 2 * sep + i + 1)));
                n(dest, 2 * sep + i) =
                    0.25 *
                    (+n.at_offset(edge, 0, 0, 0 * sep + i) *
                            (+n.at_offset(face, 0, 0, 1 * sep + i) +
                                n.at_offset(face, 0, 0, 1 * sep + i - 1)) +
                        n.at_offset(edge, 0, 1, 0 * sep + i) *
                            (+n.at_offset(face, 0, 1, 1 * sep + i) +
                                n.at_offset(face, 0, 1, 1 * sep + i - 1)) -
                        n.at_offset(edge, 0, 0, 1 * sep + i) *
                            (+n.at_offset(face, 0, 0, 0 * sep + i) +
                                n.at_offset(face, 0, 0, 0 * sep + i - 1)) -
                        n.at_offset(edge, 1, 0, 1 * sep + i) *
                            (+n.at_offset(face, 1, 0, 0 * sep + i) +
                                n.at_offset(face, 1, 0, 0 * sep + i - 1)));*/

                n(dest, x_p) =
                    0.25 *
                    (+n.at_offset(edge, 0, 0, y_p) *
                            (+n.at_offset(face, 0, 0, z_p) +
                                n.at_offset(face, -1, 0, z_p)) +
                        n.at_offset(edge, 0, 0, y_p + 1) *
                            (+n.at_offset(face, 0, 0, z_p + 1) +
                                n.at_offset(face, -1, 0, z_p + 1)) -
                        n.at_offset(edge, 0, 0, z_p) *
                            (+n.at_offset(face, 0, 0, y_p) +
                                n.at_offset(face, -1, 0, y_p)) -
                        n.at_offset(edge, 0, 1, z_p) *
                            (+n.at_offset(face, 0, 1, y_p) +
                                n.at_offset(face, -1, 1, y_p)));

                n(dest, y_p) =
                    0.25 *
                    (+n.at_offset(edge, 0, 0, z_p) *
                            (+n.at_offset(face, 0, 0, x_p) +
                                n.at_offset(face, 0, -1, x_p)) +
                        n.at_offset(edge, 1, 0, z_p) *
                            (+n.at_offset(face, 1, 0, x_p) +
                                n.at_offset(face, 1, -1, x_p)) -
                        n.at_offset(edge, 0, 0, x_p) *
                            (+n.at_offset(face, 0, 0, z_p) +
                                n.at_offset(face, 0, -1, z_p)) -
                        n.at_offset(edge, 0, 0, x_p + 1) *
                            (+n.at_offset(face, 0, 0, z_p + 1) +
                                n.at_offset(face, 0, -1, z_p + 1)));
                n(dest, z_p) =
                    0.25 *
                    (+n.at_offset(edge, 0, 0, x_p) *
                            (+n.at_offset(face, 0, 0, y_p) +
                                n.at_offset(face, 0, 0, y_p - 1)) +
                        n.at_offset(edge, 0, 1, x_p) *
                            (+n.at_offset(face, 0, 1, y_p) +
                                n.at_offset(face, 0, 1, y_p - 1)) -
                        n.at_offset(edge, 0, 0, y_p) *
                            (+n.at_offset(face, 0, 0, x_p) +
                                n.at_offset(face, 0, 0, x_p - 1)) -
                        n.at_offset(edge, 1, 0, y_p) *
                            (+n.at_offset(face, 1, 0, x_p) +
                                n.at_offset(face, 1, 0, x_p - 1)));
            }
            //i = 0
            n(dest, 0 * sep) =
                0.25 * (+n.at_offset(edge, 0, 0, 1 * sep) *
                               (+n.at_offset(face, 0, 0, 2 * sep) +
                                   n.at_offset(face, -1, 0, 2 * sep)) +
                           n.at_offset(edge, 0, 0, 1 * sep + 1) *
                               (+n.at_offset(face, 0, 0, 2 * sep + 1) +
                                   n.at_offset(face, -1, 0, 2 * sep + 1)) -
                           n.at_offset(edge, 0, 0, 2 * sep) *
                               (+n.at_offset(face, 0, 0, 1 * sep) +
                                   n.at_offset(face, -1, 0, 1 * sep)) -
                           n.at_offset(edge, 0, 1, 2 * sep) *
                               (+n.at_offset(face, 0, 1, 1 * sep) +
                                   n.at_offset(face, -1, 1, 1 * sep)));

            n(dest, 1 * sep) =
                0.25 * (+n.at_offset(edge, 0, 0, 2 * sep) *
                               (+n.at_offset(face, 0, 0, 0 * sep) +
                                   n.at_offset(face, 0, -1, 0 * sep)) +
                           n.at_offset(edge, 1, 0, 2 * sep) *
                               (+n.at_offset(face, 1, 0, 0 * sep) +
                                   n.at_offset(face, 1, -1, 0 * sep)) -
                           n.at_offset(edge, 0, 0, 0 * sep) *
                               (+n.at_offset(face, 0, 0, 2 * sep) +
                                   n.at_offset(face, 0, -1, 2 * sep)) -
                           n.at_offset(edge, 0, 0, 0 * sep + 1) *
                               (+n.at_offset(face, 0, 0, 2 * sep + 1) +
                                   n.at_offset(face, 0, -1, 2 * sep + 1)));
            n(dest, 2 * sep) =
                0.25 * (+n.at_offset(edge, 0, 0, 0 * sep) *
                               (+n.at_offset(face, 0, 0, 1 * sep) +
                                   n.at_offset(face, 0, 0, 1 * sep + sep - 1)) +
                           n.at_offset(edge, 0, 1, 0 * sep) *
                               (+n.at_offset(face, 0, 1, 1 * sep) +
                                   n.at_offset(face, 0, 1, 1 * sep + sep - 1)) -
                           n.at_offset(edge, 0, 0, 1 * sep) *
                               (+n.at_offset(face, 0, 0, 0 * sep) +
                                   n.at_offset(face, 0, 0, 0 * sep + sep - 1)) -
                           n.at_offset(edge, 1, 0, 1 * sep) *
                               (+n.at_offset(face, 1, 0, 0 * sep) +
                                   n.at_offset(face, 1, 0, 0 * sep + sep - 1)));
            //i = sep - 1
            n(dest, 0 * sep + sep - 1) =
                0.25 *
                (+n.at_offset(edge, 0, 0, 1 * sep + sep - 1) *
                        (+n.at_offset(face, 0, 0, 2 * sep + sep - 1) +
                            n.at_offset(face, -1, 0, 2 * sep + sep - 1)) +
                    n.at_offset(edge, 0, 0, 1 * sep) *
                        (+n.at_offset(face, 0, 0, 2 * sep) +
                            n.at_offset(face, -1, 0, 2 * sep)) -
                    n.at_offset(edge, 0, 0, 2 * sep + sep - 1) *
                        (+n.at_offset(face, 0, 0, 1 * sep + sep - 1) +
                            n.at_offset(face, -1, 0, 1 * sep + sep - 1)) -
                    n.at_offset(edge, 0, 1, 2 * sep + sep - 1) *
                        (+n.at_offset(face, 0, 1, 1 * sep + sep - 1) +
                            n.at_offset(face, -1, 1, 1 * sep + sep - 1)));

            n(dest, 1 * sep + sep - 1) =
                0.25 *
                (+n.at_offset(edge, 0, 0, 2 * sep + sep - 1) *
                        (+n.at_offset(face, 0, 0, 0 * sep + sep - 1) +
                            n.at_offset(face, 0, -1, 0 * sep + sep - 1)) +
                    n.at_offset(edge, 1, 0, 2 * sep + sep - 1) *
                        (+n.at_offset(face, 1, 0, 0 * sep + sep - 1) +
                            n.at_offset(face, 1, -1, 0 * sep + sep - 1)) -
                    n.at_offset(edge, 0, 0, 0 * sep + sep - 1) *
                        (+n.at_offset(face, 0, 0, 2 * sep + sep - 1) +
                            n.at_offset(face, 0, -1, 2 * sep + sep - 1)) -
                    n.at_offset(edge, 0, 0, 0 * sep) *
                        (+n.at_offset(face, 0, 0, 2 * sep) +
                            n.at_offset(face, 0, -1, 2 * sep)));
            n(dest, 2 * sep + sep - 1) =
                0.25 *
                (+n.at_offset(edge, 0, 0, 0 * sep + sep - 1) *
                        (+n.at_offset(face, 0, 0, 1 * sep + sep - 1) +
                            n.at_offset(face, 0, 0, 1 * sep + sep - 1 - 1)) +
                    n.at_offset(edge, 0, 1, 0 * sep + sep - 1) *
                        (+n.at_offset(face, 0, 1, 1 * sep + sep - 1) +
                            n.at_offset(face, 0, 1, 1 * sep + sep - 1 - 1)) -
                    n.at_offset(edge, 0, 0, 1 * sep + sep - 1) *
                        (+n.at_offset(face, 0, 0, 0 * sep + sep - 1) +
                            n.at_offset(face, 0, 0, 0 * sep + sep - 1 - 1)) -
                    n.at_offset(edge, 1, 0, 1 * sep + sep - 1) *
                        (+n.at_offset(face, 1, 0, 0 * sep + sep - 1) +
                            n.at_offset(face, 1, 0, 0 * sep + sep - 1 - 1)));
        }
    }

    template<typename Field, typename Domain, typename Func>
    static void add_field_expression(Domain* domain, Func& f, float_type t,
        float_type scale = 1.0) noexcept
    {
        const auto dx_base = domain->dx_base();
        for (auto it = domain->begin(); it != domain->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;

            for (std::size_t field_idx = 0; field_idx < Field::nFields();
                 ++field_idx)
                for (auto& n : it->data().node_field())
                {
                    auto coord = n.global_coordinate() * dx_base;
                    n(Field::tag(), field_idx) +=
                        f(field_idx, t, coord) * scale;
                }
        }
    }

    template<typename Field, typename Domain, typename Func>
    static void add_field_expression_nonlinear_helmholtz(Domain* domain, int N_modes,
        Func& f, float_type t, float_type scale = 1.0) noexcept
    {
        const auto dx_base = domain->dx_base();
        int sep = 3*N_modes; 
        // because of 3/2 rule from fft and conjugate relation from real variable transform. 
        // With 3/2 * 2 * N_modes, the total number of value in the container are 3 components * 3*N_modes
        for (auto it = domain->begin(); it != domain->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;

            for (std::size_t field_idx = 0; field_idx < Field::nFields();
                 ++field_idx)
            {
                int comp_idx = field_idx/sep;
                for (auto& n : it->data().node_field())
                {
                    auto coord = n.global_coordinate() * dx_base;
                    n(Field::tag(), field_idx) +=
                        f(comp_idx, t, coord) * scale;
                }
            }
        }
    }

    template<typename Field, typename Domain, typename Func>
    static void add_field_expression_complex_helmholtz(Domain* domain, int N_modes,
        Func& f, float_type t, float_type scale = 1.0) noexcept
    {
        const auto dx_base = domain->dx_base();
        int sep = 2*N_modes; 
        
        for (auto it = domain->begin(); it != domain->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
	    
            for (auto& n : it->data().node_field()) {
	    
            auto coord = n.global_coordinate() * dx_base;

            n(Field::tag(), 0)     += f(0, t, coord) * scale;
            n(Field::tag(), sep)   += f(1, t, coord) * scale;
            n(Field::tag(), 2*sep) += f(2, t, coord) * scale;
	    }

            /*auto coord = n.global_coordinate() * dx_base;

            n(Field::tag(), 0)     += f(0, t, coord) * scale;
            n(Field::tag(), sep)   += f(1, t, coord) * scale;
            n(Field::tag(), 2*sep) += f(2, t, coord) * scale;*/
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
                for (auto& n : it->data().node_field())
                    n(To::tag(), field_idx) +=
                        n(From::tag(), field_idx) * scale;
            }
        }
    }

    
    template<typename From, typename To, typename Block>
    static void FourierTransformC2R(Block it, int N_modes,
        int padded_dim, int vec_size, int nonzero_dim, int dim_0, int dim_1, 
        fft::helm_dfft_c2r& c2rFunc,
        int NComp = 3, int PREFAC_FROM = 2, int PREFAC_TO = 3, int lBuffer = 1,
        int rBuffer = 1)
    {
        int N_from = From::nFields();
        int N_to = To::nFields();

        std::vector<std::complex<float_type>> tmp_vec(vec_size,
            std::complex<float_type>(0.0));
        for (int i = 0; i < N_modes * NComp; i++)
        {
            auto& lin_data_real = it->data_r(From::tag(), i * 2);
            auto& lin_data_imag = it->data_r(From::tag(), i * 2 + 1);
            for (int j = 0; j < dim_0 * dim_1; j++)
            {
                if (i % N_modes == 0) lin_data_imag[j] = 0;
                std::complex<float_type> tmp_val(lin_data_real[j],
                    lin_data_imag[j]);
                //stacking data from the same x y location contiguous
                //[x_0(0,0), x_1(0,0) ...][y_0(0,0), y_1(0,0) ...][z_0(0,0), z_1(0,0) ...]
                //[x_0(0,1), x_1(0,1) ...][y_0(0,1), y_1(0,1) ...][z_0(0,1), z_1(0,1) ...]
                //...
                int idx = j * N_modes * NComp + i;
                tmp_vec[idx] = tmp_val;
            }
        }

        c2rFunc.copy_field(tmp_vec);
        c2rFunc.execute();
        std::vector<float_type> output_vel;
        c2rFunc.output_field_padded(output_vel);

        for (int i = 0; i < padded_dim * NComp; i++)
        {
            auto& lin_data_ = it->data_r(To::tag(), i);
            for (int j = 0; j < dim_0 * dim_1; j++)
            {
                int idx = j * padded_dim * NComp + i;
                lin_data_[j] = output_vel[idx]; 
            }
        }
    }

    template<typename From, typename To, typename Block>
    static void FourierTransformR2C(Block it, int N_modes, int padded_dim,
        int vec_size, int nonzero_dim, int dim_0, int dim_1,
        fft::helm_dfft_r2c& r2cFunc, int N_outputModes, int NComp = 3, int PREFAC_FROM = 3,
        int PREFAC_TO = 2, int lBuffer = 1, int rBuffer = 1)
    {
        int N_from = From::nFields();
        int N_to = To::nFields();

        std::vector<float_type> tmp_vec(vec_size, 0.0);
        for (int i = 0; i < N_modes * PREFAC_FROM * NComp; i++)
        {
            auto& lin_data_ = it->data_r(From::tag(), i);
            for (int j = 0; j < dim_0 * dim_1; j++)
            {
                int idx = j * N_modes * PREFAC_FROM * NComp + i;
                tmp_vec[idx] = lin_data_[j];
            }
        }

        r2cFunc.copy_field(tmp_vec);
        r2cFunc.execute();
        std::vector<std::complex<float_type>> output_vel;
        r2cFunc.output_field(output_vel);

        for (int i = 0; i < N_outputModes * NComp; i++)
        {
            auto& lin_data_real = it->data_r(To::tag(), i * 2);
            auto& lin_data_imag = it->data_r(To::tag(), i * 2 + 1);
            for (int j = 0; j < dim_0 * dim_1; j++)
            {
                //stacking data from the same x y location contiguous
                //[x_0(0,0), x_1(0,0) ...][y_0(0,0), y_1(0,0) ...][z_0(0,0), z_1(0,0) ...]
                //[x_0(0,1), x_1(0,1) ...][y_0(0,1), y_1(0,1) ...][z_0(0,1), z_1(0,1) ...]
                //...
                int idx = j * PREFAC_FROM * (nonzero_dim / 2 + 1) + i;
                //use (nonzero_dim/2 + 1) when use output_field
                //use (nonzero_dim/2)     when use output_field_neglect_last
                lin_data_real[j] = output_vel[idx].real() /
                                   static_cast<float_type>(padded_dim);
                lin_data_imag[j] = output_vel[idx].imag() /
                                   static_cast<float_type>(padded_dim);
            }
        }
    }
};
} // namespace domain
} // namespace iblgf

#endif
