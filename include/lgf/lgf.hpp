#ifndef INCLUDED_LGFS_HPP
#define INCLUDED_LGFS_HPP

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <boost/math/special_functions/bessel.hpp>

// IBLGF-specific
#include <lgf/lgf_lookup.hpp>

#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <global.hpp>

namespace lgf
{

using namespace domain;

template<class Policy, std::size_t Dim=3>

    class LGF : public Policy
    {

        make_field_type(Dim,lookup_field_, float_type)

    public: //Ctor:

        using datablock_t = DataBlock<Dim, node, lookup_field_>;
        using block_descriptor_t = typename datablock_t::block_descriptor_type;
        using coordinate_t = typename block_descriptor_t::base_t;

    public: //Ctor:
        LGF()=default;
        static_assert(Dim==3, "LGF only implemented for D=3");

        void get_subblock(const block_descriptor_t& _b,
                          std::vector<float_type>&  _lgf, int level_diff = 0) noexcept
        {
            const auto base = _b.base();
            const auto max  = _b.max();
            int step = pow(2, level_diff);

            _lgf.resize(_b.size());
            for (auto k = base[2]; k <= max[2] ; ++k)
            {
                for (auto j = base[1]; j <= max[1]; ++j)
                {
                    for (auto i = base[0]; i <= max[0]; ++i)
                    {
                        //get view
                        _lgf[_b.index(i,j,k)] =
                            Lookup::get(coordinate_t({i*step, j*step, k*step}));
                    }
                }
            }
        }


    public:

        template<class Coordinate>
        static auto get(const Coordinate& _coord) noexcept
        {
            return Policy::get(_coord);
        }

    private:
        std::vector< std::vector<float_type> > level_lgf_maps_;   ///< Octants per level

    };

}

#endif
