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

#ifndef IBLGF_INCLUDED_IFHERK_SOLVER_HPP
#define IBLGF_INCLUDED_IFHERK_SOLVER_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <array>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/IO/parallel_ostream.hpp>
#include <iblgf/solver/poisson/poisson.hpp>
#include <iblgf/solver/linsys/linsys.hpp>
#include <iblgf/operators/operators.hpp>
#include <iblgf/utilities/misc_math_functions.hpp>

namespace iblgf
{
namespace solver
{
using namespace domain;

/** @brief Integrating factor 3-stage Runge-Kutta time integration
 * */
template<class Setup>
class Ifherk
{
  public: //member types
    using simulation_type = typename Setup::simulation_t;
    using domain_type = typename simulation_type::domain_type;
    using datablock_type = typename domain_type::datablock_t;
    using tree_t = typename domain_type::tree_t;
    using octant_t = typename tree_t::octant_type;
    using MASK_TYPE = typename octant_t::MASK_TYPE;
    using block_type = typename datablock_type::block_descriptor_type;
    using real_coordinate_type = typename domain_type::real_coordinate_type;
    using coordinate_type = typename domain_type::coordinate_type;
    using poisson_solver_t = typename Setup::poisson_solver_t;
    using linsys_solver_t = typename Setup::linsys_solver_t;

    using ib_t = typename domain_type::ib_t;
    using force_type = typename ib_t::force_type;

    //FMM
    using Fmm_t = typename Setup::Fmm_t;

    using test_type = typename Setup::test_type;

    using u_type = typename Setup::u_type;
    using stream_f_type = typename Setup::stream_f_type;
    using p_type = typename Setup::p_type;
    using q_i_type = typename Setup::q_i_type;
    using r_i_type = typename Setup::r_i_type;
    using g_i_type = typename Setup::g_i_type;
    using d_i_type = typename Setup::d_i_type;

    using cell_aux_type = typename Setup::cell_aux_type;
    using edge_aux_type = typename Setup::edge_aux_type;
    using face_aux_type = typename Setup::face_aux_type;
    using face_aux2_type = typename Setup::face_aux2_type;
    using correction_tmp_type = typename Setup::correction_tmp_type;
    using w_1_type = typename Setup::w_1_type;
    using w_2_type = typename Setup::w_2_type;
    using u_i_type = typename Setup::u_i_type;

    static constexpr int lBuffer = 1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer = 1; ///< Lower left buffer for interpolation
    Ifherk(simulation_type* _simulation)
    : simulation_(_simulation)
    , domain_(_simulation->domain_.get())
    , psolver(_simulation)
    , lsolver(_simulation)
    {
        // parameters --------------------------------------------------------

        dx_base_ = domain_->dx_base();
        max_ref_level_ =
            _simulation->dictionary()->template get<float_type>("nLevels");
        cfl_ =
            _simulation->dictionary()->template get_or<float_type>("cfl", 0.2);
        dt_base_ =
            _simulation->dictionary()->template get_or<float_type>("dt", -1.0);
        tot_base_steps_ =
            _simulation->dictionary()->template get<int>("nBaseLevelTimeSteps");
        Re_ = _simulation->dictionary()->template get<float_type>("Re");
        output_base_freq_ = _simulation->dictionary()->template get<float_type>(
            "output_frequency");
        cfl_max_ = _simulation->dictionary()->template get_or<float_type>(
            "cfl_max", 1000);
        updating_source_max_ = _simulation->dictionary()->template get_or<bool>(
            "updating_source_max", true);
        all_time_max_ = _simulation->dictionary()->template get_or<bool>(
            "all_time_max", true);

        use_adaptation_correction = _simulation->dictionary()->template get_or<bool>(
            "use_adaptation_correction", true);

        b_f_mag = _simulation->dictionary()->template get_or<float_type>(
            "b_f_mag", 1.0);

        t_dipole = _simulation->dictionary()->template get_or<float_type>(
            "t_dipole", 1.0);

        R_dipole = _simulation->dictionary()->template get_or<float_type>(
            "R_dipole", 1.0);

        h_dipole = _simulation->dictionary()->template get_or<float_type>(
            "h_dipole", 1.0);

        l_dipole = _simulation->dictionary()->template get_or<float_type>(
            "l_dipole", 1.0);

        a_dipole = _simulation->dictionary()->template get_or<float_type>(
            "a_dipole", 1.0);

        tansteepx = _simulation->dictionary()->template get_or<float_type>(
            "tansteepx", 1.0);

        tansteepy = _simulation->dictionary()->template get_or<float_type>(
            "tansteepy", 1.0);

        tansteepz = _simulation->dictionary()->template get_or<float_type>(
            "tansteepz", 1.0);

        widthx = _simulation->dictionary()->template get_or<float_type>(
            "widthx", 1.0);

        widthy = _simulation->dictionary()->template get_or<float_type>(
            "widthy", 1.0);

        widthz = _simulation->dictionary()->template get_or<float_type>(
            "widthz", 1.0);

        speedx = _simulation->dictionary()->template get_or<float_type>(
            "speedx", 1.0);

        periody = _simulation->dictionary()->template get_or<float_type>(
            "periody", 1.0);

        periodz = _simulation->dictionary()->template get_or<float_type>(
            "periodz", 1.0);

        amplitudey = _simulation->dictionary()->template get_or<float_type>(
            "amplitudey", 1.0);

        amplitudez = _simulation->dictionary()->template get_or<float_type>(
            "amplitudez", 1.0);

        at1 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "at1", 1.0);
        at2 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "at2", 1.0);
        at3 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "at3", 1.0);
        at4 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "at4", 1.0);
        at5 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "at5", 1.0);
        at6 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "at6", 1.0);
        at7 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "at7", 1.0);
        at8 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "at8", 1.0);
        at9 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "at9", 1.0);
        at10 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at10", 1.0);
        at11 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at11", 1.0);
        at12 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at12", 1.0);
        at13 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at13", 1.0);
        at14 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at14", 1.0);
        at15 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at15", 1.0);
        at16 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at16", 1.0);
        at17 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at17", 1.0);
        at18 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at18", 1.0);
        at19 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at19", 1.0);
        at20 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at20", 1.0);
        at21 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at21", 1.0);
        at22 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at22", 1.0);
        at23 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at23", 1.0);
        at24 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at24", 1.0);
        at25 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at25", 1.0);
        at26 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at26", 1.0);
        at27 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at27", 1.0);
        at28 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at28", 1.0);
        at29 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at29", 1.0);
        at30 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at30", 1.0);
        at31 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at31", 1.0);
        at32 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at32", 1.0);
        at33 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at33", 1.0);
        at34 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at34", 1.0);
        at35 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at35", 1.0);
        at36 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at36", 1.0);
        at37 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at37", 1.0);
        at38 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at38", 1.0);
        at39 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at39", 1.0);
        at40 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at40", 1.0);
        at41 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at41", 1.0);
        at42 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at42", 1.0);
        at43 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at43", 1.0);
        at44 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at44", 1.0);
        at45 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at45", 1.0);
        at46 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at46", 1.0);
        at47 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at47", 1.0);
        at48 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at48", 1.0);
        at49 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at49", 1.0);
        at50 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at50", 1.0);
        at51 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at51", 1.0);
        at52 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at52", 1.0);
        at53 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at53", 1.0);
        at54 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at54", 1.0);
        at55 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at55", 1.0);
        at56 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at56", 1.0);
        at57 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at57", 1.0);
        at58 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at58", 1.0);
        at59 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at59", 1.0);
        at60 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at60", 1.0);
        at61 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at61", 1.0);
        at62 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at62", 1.0);
        at63 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at63", 1.0);
        at64 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at64", 1.0);
        at65 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at65", 1.0);
        at66 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at66", 1.0);
        at67 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at67", 1.0);
        at68 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at68", 1.0);
        at69 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at69", 1.0);
        at70 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at70", 1.0);
        at71 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at71", 1.0);
        at72 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at72", 1.0);
        at73 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at73", 1.0);
        at74 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at74", 1.0);
        at75 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at75", 1.0);
        at76 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at76", 1.0);
        at77 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at77", 1.0);
        at78 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at78", 1.0);
        at79 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at79", 1.0);
        at80 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at80", 1.0);
        at81 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at81", 1.0);
        at82 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at82", 1.0);
        at83 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at83", 1.0);
        at84 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at84", 1.0);
        at85 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at85", 1.0);
        at86 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at86", 1.0);
        at87 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at87", 1.0);
        at88 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at88", 1.0);
        at89 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at89", 1.0);
        at90 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at90", 1.0);
        at91 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at91", 1.0);
        at92 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at92", 1.0);
        at93 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at93", 1.0);
        at94 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at94", 1.0);
        at95 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at95", 1.0);
        at96 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at96", 1.0);
        at97 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at97", 1.0);
        at98 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at98", 1.0);
        at99 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "at99", 1.0);
        at100 = _simulation->dictionary()->template get_or<float_type>(
                                                                       "at100", 1.0);

        pt1 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pt1", 1.0);
        pt2 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pt2", 1.0);
        pt3 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pt3", 1.0);
        pt4 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pt4", 1.0);
        pt5 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pt5", 1.0);
        pt6 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pt6", 1.0);
        pt7 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pt7", 1.0);
        pt8 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pt8", 1.0);
        pt9 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pt9", 1.0);
        pt10 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt10", 1.0);
        pt11 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt11", 1.0);
        pt12 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt12", 1.0);
        pt13 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt13", 1.0);
        pt14 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt14", 1.0);
        pt15 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt15", 1.0);
        pt16 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt16", 1.0);
        pt17 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt17", 1.0);
        pt18 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt18", 1.0);
        pt19 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt19", 1.0);
        pt20 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt20", 1.0);
        pt21 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt21", 1.0);
        pt22 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt22", 1.0);
        pt23 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt23", 1.0);
        pt24 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt24", 1.0);
        pt25 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt25", 1.0);
        pt26 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt26", 1.0);
        pt27 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt27", 1.0);
        pt28 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt28", 1.0);
        pt29 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt29", 1.0);
        pt30 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt30", 1.0);
        pt31 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt31", 1.0);
        pt32 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt32", 1.0);
        pt33 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt33", 1.0);
        pt34 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt34", 1.0);
        pt35 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt35", 1.0);
        pt36 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt36", 1.0);
        pt37 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt37", 1.0);
        pt38 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt38", 1.0);
        pt39 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt39", 1.0);
        pt40 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt40", 1.0);
        pt41 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt41", 1.0);
        pt42 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt42", 1.0);
        pt43 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt43", 1.0);
        pt44 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt44", 1.0);
        pt45 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt45", 1.0);
        pt46 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt46", 1.0);
        pt47 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt47", 1.0);
        pt48 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt48", 1.0);
        pt49 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt49", 1.0);
        pt50 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt50", 1.0);
        pt51 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt51", 1.0);
        pt52 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt52", 1.0);
        pt53 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt53", 1.0);
        pt54 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt54", 1.0);
        pt55 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt55", 1.0);
        pt56 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt56", 1.0);
        pt57 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt57", 1.0);
        pt58 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt58", 1.0);
        pt59 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt59", 1.0);
        pt60 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt60", 1.0);
        pt61 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt61", 1.0);
        pt62 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt62", 1.0);
        pt63 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt63", 1.0);
        pt64 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt64", 1.0);
        pt65 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt65", 1.0);
        pt66 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt66", 1.0);
        pt67 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt67", 1.0);
        pt68 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt68", 1.0);
        pt69 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt69", 1.0);
        pt70 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt70", 1.0);
        pt71 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt71", 1.0);
        pt72 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt72", 1.0);
        pt73 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt73", 1.0);
        pt74 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt74", 1.0);
        pt75 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt75", 1.0);
        pt76 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt76", 1.0);
        pt77 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt77", 1.0);
        pt78 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt78", 1.0);
        pt79 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt79", 1.0);
        pt80 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt80", 1.0);
        pt81 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt81", 1.0);
        pt82 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt82", 1.0);
        pt83 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt83", 1.0);
        pt84 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt84", 1.0);
        pt85 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt85", 1.0);
        pt86 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt86", 1.0);
        pt87 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt87", 1.0);
        pt88 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt88", 1.0);
        pt89 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt89", 1.0);
        pt90 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt90", 1.0);
        pt91 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt91", 1.0);
        pt92 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt92", 1.0);
        pt93 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt93", 1.0);
        pt94 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt94", 1.0);
        pt95 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt95", 1.0);
        pt96 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt96", 1.0);
        pt97 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt97", 1.0);
        pt98 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt98", 1.0);
        pt99 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pt99", 1.0);
        pt100 = _simulation->dictionary()->template get_or<float_type>(
                                                                       "pt100", 1.0);

        ab1 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "ab1", 1.0);
        ab2 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "ab2", 1.0);
        ab3 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "ab3", 1.0);
        ab4 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "ab4", 1.0);
        ab5 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "ab5", 1.0);
        ab6 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "ab6", 1.0);
        ab7 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "ab7", 1.0);
        ab8 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "ab8", 1.0);
        ab9 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "ab9", 1.0);
        ab10 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab10", 1.0);
        ab11 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab11", 1.0);
        ab12 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab12", 1.0);
        ab13 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab13", 1.0);
        ab14 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab14", 1.0);
        ab15 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab15", 1.0);
        ab16 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab16", 1.0);
        ab17 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab17", 1.0);
        ab18 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab18", 1.0);
        ab19 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab19", 1.0);
        ab20 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab20", 1.0);
        ab21 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab21", 1.0);
        ab22 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab22", 1.0);
        ab23 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab23", 1.0);
        ab24 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab24", 1.0);
        ab25 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab25", 1.0);
        ab26 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab26", 1.0);
        ab27 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab27", 1.0);
        ab28 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab28", 1.0);
        ab29 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab29", 1.0);
        ab30 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab30", 1.0);
        ab31 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab31", 1.0);
        ab32 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab32", 1.0);
        ab33 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab33", 1.0);
        ab34 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab34", 1.0);
        ab35 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab35", 1.0);
        ab36 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab36", 1.0);
        ab37 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab37", 1.0);
        ab38 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab38", 1.0);
        ab39 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab39", 1.0);
        ab40 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab40", 1.0);
        ab41 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab41", 1.0);
        ab42 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab42", 1.0);
        ab43 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab43", 1.0);
        ab44 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab44", 1.0);
        ab45 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab45", 1.0);
        ab46 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab46", 1.0);
        ab47 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab47", 1.0);
        ab48 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab48", 1.0);
        ab49 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab49", 1.0);
        ab50 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab50", 1.0);
        ab51 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab51", 1.0);
        ab52 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab52", 1.0);
        ab53 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab53", 1.0);
        ab54 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab54", 1.0);
        ab55 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab55", 1.0);
        ab56 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab56", 1.0);
        ab57 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab57", 1.0);
        ab58 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab58", 1.0);
        ab59 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab59", 1.0);
        ab60 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab60", 1.0);
        ab61 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab61", 1.0);
        ab62 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab62", 1.0);
        ab63 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab63", 1.0);
        ab64 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab64", 1.0);
        ab65 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab65", 1.0);
        ab66 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab66", 1.0);
        ab67 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab67", 1.0);
        ab68 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab68", 1.0);
        ab69 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab69", 1.0);
        ab70 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab70", 1.0);
        ab71 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab71", 1.0);
        ab72 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab72", 1.0);
        ab73 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab73", 1.0);
        ab74 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab74", 1.0);
        ab75 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab75", 1.0);
        ab76 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab76", 1.0);
        ab77 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab77", 1.0);
        ab78 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab78", 1.0);
        ab79 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab79", 1.0);
        ab80 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab80", 1.0);
        ab81 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab81", 1.0);
        ab82 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab82", 1.0);
        ab83 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab83", 1.0);
        ab84 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab84", 1.0);
        ab85 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab85", 1.0);
        ab86 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab86", 1.0);
        ab87 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab87", 1.0);
        ab88 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab88", 1.0);
        ab89 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab89", 1.0);
        ab90 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab90", 1.0);
        ab91 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab91", 1.0);
        ab92 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab92", 1.0);
        ab93 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab93", 1.0);
        ab94 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab94", 1.0);
        ab95 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab95", 1.0);
        ab96 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab96", 1.0);
        ab97 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab97", 1.0);
        ab98 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab98", 1.0);
        ab99 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "ab99", 1.0);
        ab100 = _simulation->dictionary()->template get_or<float_type>(
                                                                       "ab100", 1.0);

        pb1 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pb1", 1.0);
        pb2 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pb2", 1.0);
        pb3 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pb3", 1.0);
        pb4 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pb4", 1.0);
        pb5 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pb5", 1.0);
        pb6 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pb6", 1.0);
        pb7 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pb7", 1.0);
        pb8 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pb8", 1.0);
        pb9 = _simulation->dictionary()->template get_or<float_type>(
                                                                     "pb9", 1.0);
        pb10 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb10", 1.0);
        pb11 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb11", 1.0);
        pb12 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb12", 1.0);
        pb13 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb13", 1.0);
        pb14 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb14", 1.0);
        pb15 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb15", 1.0);
        pb16 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb16", 1.0);
        pb17 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb17", 1.0);
        pb18 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb18", 1.0);
        pb19 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb19", 1.0);
        pb20 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb20", 1.0);
        pb21 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb21", 1.0);
        pb22 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb22", 1.0);
        pb23 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb23", 1.0);
        pb24 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb24", 1.0);
        pb25 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb25", 1.0);
        pb26 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb26", 1.0);
        pb27 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb27", 1.0);
        pb28 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb28", 1.0);
        pb29 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb29", 1.0);
        pb30 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb30", 1.0);
        pb31 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb31", 1.0);
        pb32 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb32", 1.0);
        pb33 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb33", 1.0);
        pb34 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb34", 1.0);
        pb35 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb35", 1.0);
        pb36 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb36", 1.0);
        pb37 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb37", 1.0);
        pb38 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb38", 1.0);
        pb39 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb39", 1.0);
        pb40 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb40", 1.0);
        pb41 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb41", 1.0);
        pb42 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb42", 1.0);
        pb43 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb43", 1.0);
        pb44 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb44", 1.0);
        pb45 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb45", 1.0);
        pb46 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb46", 1.0);
        pb47 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb47", 1.0);
        pb48 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb48", 1.0);
        pb49 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb49", 1.0);
        pb50 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb50", 1.0);
        pb51 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb51", 1.0);
        pb52 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb52", 1.0);
        pb53 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb53", 1.0);
        pb54 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb54", 1.0);
        pb55 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb55", 1.0);
        pb56 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb56", 1.0);
        pb57 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb57", 1.0);
        pb58 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb58", 1.0);
        pb59 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb59", 1.0);
        pb60 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb60", 1.0);
        pb61 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb61", 1.0);
        pb62 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb62", 1.0);
        pb63 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb63", 1.0);
        pb64 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb64", 1.0);
        pb65 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb65", 1.0);
        pb66 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb66", 1.0);
        pb67 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb67", 1.0);
        pb68 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb68", 1.0);
        pb69 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb69", 1.0);
        pb70 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb70", 1.0);
        pb71 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb71", 1.0);
        pb72 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb72", 1.0);
        pb73 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb73", 1.0);
        pb74 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb74", 1.0);
        pb75 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb75", 1.0);
        pb76 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb76", 1.0);
        pb77 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb77", 1.0);
        pb78 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb78", 1.0);
        pb79 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb79", 1.0);
        pb80 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb80", 1.0);
        pb81 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb81", 1.0);
        pb82 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb82", 1.0);
        pb83 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb83", 1.0);
        pb84 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb84", 1.0);
        pb85 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb85", 1.0);
        pb86 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb86", 1.0);
        pb87 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb87", 1.0);
        pb88 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb88", 1.0);
        pb89 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb89", 1.0);
        pb90 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb90", 1.0);
        pb91 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb91", 1.0);
        pb92 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb92", 1.0);
        pb93 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb93", 1.0);
        pb94 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb94", 1.0);
        pb95 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb95", 1.0);
        pb96 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb96", 1.0);
        pb97 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb97", 1.0);
        pb98 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb98", 1.0);
        pb99 = _simulation->dictionary()->template get_or<float_type>(
                                                                      "pb99", 1.0);
        pb100 = _simulation->dictionary()->template get_or<float_type>(
                                                                       "pb100", 1.0);

        k_dipole = _simulation->dictionary()->template get_or<float_type>(
            "k_dipole", 1.0);

        b_f_eps = _simulation->dictionary()->template get_or<float_type>(
            "b_f_eps", 1e-3);

        use_filter = _simulation->dictionary()->template get_or<bool>(
            "use_filter", true);

        dim = domain_->dimension();

        cg_threshold_ = simulation_->dictionary_->template get_or<float_type>("cg_threshold",1e-3);
        cg_max_itr_ = simulation_->dictionary_->template get_or<int>("cg_max_itr", 40);


        if (dt_base_ < 0) dt_base_ = dx_base_ * cfl_;

        // adaptivity --------------------------------------------------------
        adapt_freq_ = _simulation->dictionary()->template get_or<float_type>(
            "adapt_frequency", 1);
        T_max_ = tot_base_steps_ * dt_base_;
        update_marching_parameters();

        // support of IF in every dirrection is about 3.2 corresponding to 1e-5
        // with coefficient alpha = 1
        // so need update
        max_vel_refresh_ = floor(14/(3.3/(Re_*dx_base_*dx_base_/dt_)));
        pcout<<"maximum steps allowed without vel refresh = " << max_vel_refresh_ <<std::endl;

        // restart -----------------------------------------------------------
        write_restart_ = _simulation->dictionary()->template get_or<bool>(
            "write_restart", true);

        if (write_restart_)
            restart_base_freq_ =
                _simulation->dictionary()->template get<float_type>(
                    "restart_write_frequency");

        // IF constants ------------------------------------------------------
        fname_prefix_ = "";

        // miscs -------------------------------------------------------------
    }

  public:
    void update_marching_parameters()
    {
        nLevelRefinement_ = domain_->tree()->depth()-domain_->tree()->base_level()-1;
        dt_               = dt_base_/math::pow2(nLevelRefinement_);

        float_type tmp = Re_*dx_base_*dx_base_/dt_;
        alpha_[0]=(c_[1]-c_[0])/tmp;
        alpha_[1]=(c_[2]-c_[1])/tmp;
        alpha_[2]=(c_[3]-c_[2])/tmp;


    }
    void time_march(bool use_restart=false)
    {
        use_restart_ = use_restart;
        boost::mpi::communicator          world;
        parallel_ostream::ParallelOstream pcout =
            parallel_ostream::ParallelOstream(world.size() - 1);

        pcout
            << "Time marching ------------------------------------------------ "
            << std::endl;
        // --------------------------------------------------------------------
        if (use_restart_)
        {
            just_restarted_= true;
            Dictionary info_d(simulation_->restart_load_dir()+"/restart_info");
            T_=info_d.template get<float_type>("T");
            adapt_count_=info_d.template get<int>("adapt_count");
            T_last_vel_refresh_=info_d.template get_or<float_type>("T_last_vel_refresh", 0.0);
            source_max_[0]=info_d.template get<float_type>("cell_aux_max");
            source_max_[1]=info_d.template get<float_type>("u_max");
            pcout<<"Restart info ------------------------------------------------ "<< std::endl;
            pcout<<"T = "<< T_<< std::endl;
            pcout<<"adapt_count = "<< adapt_count_<< std::endl;
            pcout<<"cell aux max = "<< source_max_[0]<< std::endl;
            pcout<<"u max = "<< source_max_[1]<< std::endl;
            pcout<<"T_last_vel_refresh = "<< T_last_vel_refresh_<< std::endl;
            if(domain_->is_client())
            {
                //pad_velocity<u_type, u_type>(true);
            }
        }
        else
        {
            T_ = 0.0;
            adapt_count_=0;

            write_timestep();
        }

        // ----------------------------------- start -------------------------

        while(T_<T_max_-1e-10)
        {


            // -------------------------------------------------------------
            // adapt

            // clean up the block boundary of cell_aux_type for smoother adaptation

            if(domain_->is_client())
            {
                clean<cell_aux_type>(true, 2);
                clean<edge_aux_type>(true, 1);
                clean<correction_tmp_type>(true, 2);
            }
            else
            {
                //const auto& lb = domain_->level_blocks();
                /*std::vector<int> lb;
                domain_->level_blocks(lb);*/
                auto lb = domain_->level_blocks();
                std::cout<<"Blocks on each level: ";

                for (int c: lb)
                    std::cout<< c << " ";
                std::cout<<std::endl;

            }

            // copy flag correction to flag old correction
            for (auto it  = domain_->begin();
                    it != domain_->end(); ++it)
            {
                it->flag_old_correction(false);
            }

            for (auto it  = domain_->begin(domain_->tree()->base_level());
                    it != domain_->end(domain_->tree()->base_level()); ++it)
            {
                it->flag_old_correction(it->is_correction());
            }

            int c=0;

            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (!it->locally_owned()) continue;
                if (it->is_ib() || it->is_extended_ib())
                {
                    auto& lin_data =
                        it->data_r(test_type::tag(), 0).linalg_data();
                    std::fill(lin_data.begin(), lin_data.end(), 2.0);
                    c+=1;
                }

            }
            boost::mpi::communicator world;
            int c_all;
            boost::mpi::all_reduce(
                world, c, c_all, std::plus<int>());
            pcout<< "block = " << c_all << std::endl;


            if ( adapt_count_ % adapt_freq_ ==0 && adapt_count_ != 0)
            {
                if (adapt_count_==0 || updating_source_max_)
                {
                    this->template update_source_max<cell_aux_type>(0);
                    this->template update_source_max<edge_aux_type>(1);
                }

                //if(domain_->is_client())
                //{
                //    up_and_down<u>();
                //    pad_velocity<u, u>();
                //}
                if (!just_restarted_) {
                    this->adapt(false);
                    adapt_corr_time_step();
                }
                just_restarted_=false;

            }

            // balance load
            if ( adapt_count_ % adapt_freq_ ==0)
            {
                clean<u_type>(true);
                domain_->decomposition().template balance<u_type,p_type>();
            }

            adapt_count_++;

            // -------------------------------------------------------------
            // time marching

            mDuration_type ifherk_if(0);
            TIME_CODE( ifherk_if, SINGLE_ARG(
                        time_step();
                        ));
            pcout<<ifherk_if.count()<<std::endl;

            // -------------------------------------------------------------
            // update stats & output

            T_ += dt_;
            float_type tmp_n = T_ / dt_base_ * math::pow2(max_ref_level_);
            int        tmp_int_n = int(tmp_n + 0.5);

            if (write_restart_ && (std::fabs(tmp_int_n - tmp_n) < 1e-4) &&
                (tmp_int_n % restart_base_freq_ == 0))
            {
                restart_n_last_ = tmp_int_n;
                write_restart();
            }

            if ((std::fabs(tmp_int_n - tmp_n) < 1e-4) &&
                (tmp_int_n % output_base_freq_ == 0))
            {
                //test
                //

                //clean<test_type>();
                //auto f = [](std::size_t idx, float_type t, auto coord = {0, 0, 0}){return 1.0;};
                //domain::Operator::add_field_expression<test_type>(domain_, f, T_, 1.0);
                //clean_leaf_correction_boundary<test_type>(domain_->tree()->base_level(),true,2);

                n_step_ = tmp_int_n;
                write_timestep();
                // only update dt after 1 output so it wouldn't do 3 5 7 9 ...
                // and skip all outputs
                update_marching_parameters();
            }

            write_stats(tmp_n);

        }
    }
    void clean_up_initial_velocity()
    {
        if (domain_->is_client())
        {
            up_and_down<u_type>();
            auto client = domain_->decomposition().client();
            clean<edge_aux_type>();
            clean<stream_f_type>();
            for (int l = domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
            {
                client->template buffer_exchange<u_type>(l);
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned() || it->is_correction()) continue;

                    const auto dx_level =
                        dx_base_ / math::pow2(it->refinement_level());
                    domain::Operator::curl<u_type, edge_aux_type>(
                        it->data(), dx_level);
                }
            }
            //clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(), true,2);

            clean<u_type>();
            psolver.template apply_lgf<edge_aux_type, stream_f_type>();
            for (int l = domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
            {
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned()) continue;

                    const auto dx_level =
                        dx_base_ / math::pow2(it->refinement_level());
                    domain::Operator::curl_transpose<stream_f_type, u_type>(
                        it->data(), dx_level, -1.0);
                }
                client->template buffer_exchange<u_type>(l);
            }
        }
    }

    template<class Field>
    void update_source_max(int idx)
    {
        float_type max_local = 0.0;
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned()) continue;
            float_type tmp = domain::Operator::blockRootMeanSquare<Field>(it->data());

            if (tmp > max_local) max_local = tmp;
        }

        float_type new_maximum=0.0;
        boost::mpi::all_reduce(
            comm_, max_local, new_maximum, boost::mpi::maximum<float_type>());

        //source_max_[idx] = std::max(source_max_[idx], new_maximum);
        if (all_time_max_)
            source_max_[idx] = std::max(source_max_[idx], new_maximum);
        else
            source_max_[idx] = 0.5*( source_max_[idx] + new_maximum );

        if (source_max_[idx]< 1e-2)
            source_max_[idx] = 1e-2;

        pcout << "source max = "<< source_max_[idx] << std::endl;
    }

    void write_restart()
    {
        boost::mpi::communicator world;

        world.barrier();
        if (domain_->is_server() && write_restart_)
        {
            std::cout << "restart: backup" << std::endl;
            simulation_->copy_restart();
        }
        world.barrier();

        pcout << "restart: write" << std::endl;
        simulation_->write("", true);

        write_info();
        world.barrier();
    }

    void write_stats(int tmp_n)
    {
        boost::mpi::communicator world;
        world.barrier();

        // - Numeber of cells -----------------------------------------------
        int c_allc_global;
        int c_allc = domain_->num_allocations();
        boost::mpi::all_reduce(
                world, c_allc, c_allc_global, std::plus<int>());

        if (domain_->is_server())
        {
            std::cout<<"T = " << T_<<", n = "<< tmp_n << " -----------------" << std::endl;
            auto lb = domain_->level_blocks();
            std::cout<<"Blocks on each level: ";

            for (int c: lb)
                std::cout<< c << " ";
            std::cout<<std::endl;

            std::cout<<"Total number of leaf octants: "<<domain_->num_leafs()<<std::endl;
            std::cout<<"Total number of leaf + correction octants: "<<domain_->num_corrections()+domain_->num_leafs()<<std::endl;
            std::cout<<"Total number of allocated octants: "<<c_allc_global<<std::endl;
            std::cout<<" -----------------" << std::endl;
        }


        // - Forcing ------------------------------------------------
        auto& ib = domain_->ib();
        ib.clean_non_local();
        real_coordinate_type tmp_coord(0.0);

        force_type sum_f(ib.force().size(), tmp_coord);
        if (ib.size() > 0)
        {
            boost::mpi::all_reduce(world, &ib.force(0), ib.size(), &sum_f[0],
                std::plus<real_coordinate_type>());
        }
        if (domain_->is_server())
        {
            std::vector<float_type> f(domain_->dimension(), 0.);
            if (ib.size() > 0)
            {
                for (std::size_t d = 0; d < domain_->dimension(); ++d)
                    for (std::size_t i = 0; i < ib.size(); ++i)
                        f[d] += sum_f[i][d] * 1.0 / coeff_a(3, 3) / dt_ *
                                ib.force_scale();
                //f[d]+=sum_f[i][d] * 1.0 / dt_ * ib.force_scale();

                std::cout<<"ib  size: "<<ib.size()<<std::endl;
                std::cout << "Forcing = ";

                for (std::size_t d = 0; d < domain_->dimension(); ++d)
                    std::cout << f[d] << " ";
                std::cout << std::endl;

                std::cout << " -----------------" << std::endl;
            }

            std::ofstream outfile;
            int width=20;

            outfile.open("fstats.txt", std::ios_base::app); // append instead of overwrite
            outfile <<std::setw(width) << tmp_n <<std::setw(width)<<std::scientific<<std::setprecision(9);
            outfile <<std::setw(width) << T_ <<std::setw(width)<<std::scientific<<std::setprecision(9);
            for (auto& element:f)
            {
                outfile<<element<<std::setw(width);
            }
            outfile<<std::endl;
        }



        force_type All_sum_f(ib.force().size(), tmp_coord);
        force_type All_sum_f_glob(ib.force().size(), tmp_coord);
        this->template ComputeForcing<u_type, p_type, u_i_type, d_i_type>(All_sum_f);

        if (ib.size() > 0)
        {
            for (std::size_t d = 0; d < All_sum_f[0].size(); ++d)
            {
                for (std::size_t i = 0; i < All_sum_f.size(); ++i)
                {
                    if (world.rank() != domain_->ib().rank(i))
                        All_sum_f[i][d] = 0.0;
                }
            }

            boost::mpi::all_reduce(world,
                &All_sum_f[0], All_sum_f.size(), &All_sum_f_glob[0],
                std::plus<real_coordinate_type>());
        }

        if (domain_->is_server())
        {
            std::vector<float_type> f(ib_t::force_dim, 0.);
            if (ib.size() > 0)
            {
                for (std::size_t d = 0; d < ib_t::force_dim; ++d)
                    for (std::size_t i = 0; i < ib.size(); ++i)
                        f[d] += All_sum_f_glob[i][d] * ib.force_scale();
                //f[d]+=sum_f[i][d] * 1.0 / dt_ * ib.force_scale();

                std::cout << "ib  size: " << ib.size() << std::endl;
                std::cout << "New Forcing = ";

                for (std::size_t d = 0; d < domain_->dimension(); ++d)
                    std::cout << -f[d] << " ";
                std::cout << std::endl;

                std::cout << " -----------------" << std::endl;
            }

            std::ofstream outfile;
            int           width = 20;

            outfile.open("fstats_from_cg.txt",
                std::ios_base::app); // append instead of overwrite
            outfile << std::setw(width) << tmp_n << std::setw(width)
                    << std::scientific << std::setprecision(9);
            outfile << std::setw(width) << T_ << std::setw(width)
                    << std::scientific << std::setprecision(9);
            for (int  i = 0 ; i < domain_->dimension(); i++) { outfile << -f[i] << std::setw(width); }
            outfile << std::endl;
            outfile.close();
            std::cout << "finished writing new forcing" << std::endl;
        }

        world.barrier();
    }


    template<class Source_face, class Source_cell, class Target_face, class Target_cell>
    void ComputeForcing(force_type& force_target) noexcept
    {
        if (domain_->is_server())
            return;
        auto client = domain_->decomposition().client();

        boost::mpi::communicator world;

        //pad_velocity<Source_face, Source_face>(true);





        clean<face_aux2_type>();
        clean<edge_aux_type>();
        clean<w_1_type>();
        clean<w_2_type>();

        //clean<r_i_T_type>(); //use r_i as the result of applying Jcobian in the first block
        //clean<cell_aux_T_type>(); //use cell aux_type to be the second block
        clean<Target_face>();
        clean<Target_cell>();
        real_coordinate_type tmp_coord(0.0);
        auto forcing_tmp = force_target;
        std::fill(forcing_tmp.begin(), forcing_tmp.end(),
            tmp_coord);
        std::fill(force_target.begin(), force_target.end(),
            tmp_coord); //use forcing tmp to store the last block,
            //use forcing_old to store the forcing at previous Newton iteration
        //computing wii in Andre's paper
        real_coordinate_type tmp_coord1(1.0);
        auto forcing_1 = force_target;
        std::fill(forcing_1.begin(), forcing_1.end(),
            tmp_coord1);

        lsolver.template smearing<w_1_type>(forcing_1, true);
        computeWii<w_1_type>();
        //finish computing Wii



        laplacian<Source_face, face_aux2_type>();

        clean<r_i_type>();
        clean<g_i_type>();
        gradient<Source_cell, r_i_type>(1.0);

        //pcout << "Computed Laplacian " << std::endl;

        nonlinear<Source_face, g_i_type>();

        //pcout << "Computed Nonlinear Jac " << std::endl;

        add<g_i_type, Target_face>(1);

        add<face_aux2_type, Target_face>(-1.0 / Re_);

        add<r_i_type, Target_face>(1.0);

        lsolver.template projection<Target_face>(forcing_tmp);

        auto r = forcing_tmp;

        //force_target = forcing_tmp;

        float_type r2_old = dotVec(r, r);

        if (r2_old < 1e-12) {
            if (world.rank() == 1) {
                std::cout << "r0 small, exiting" << std::endl;
            }
            return;
        }

        auto p = r;

        for (int i = 0; i < cg_max_itr_; i++) {
            clean<r_i_type>();
            lsolver.template smearing<r_i_type>(p, false);
            auto Ap = r;
            cleanVec(Ap,false);
            lsolver.template projection<r_i_type>(Ap);
            r2_old = dotVec(r, r);
            float_type pAp = dotVec(p, Ap);
            float_type alpha = r2_old/pAp;
            //force_target += alpha*p;
            addVec(force_target, p, 1.0, alpha);
            //r -= alpha*Ap;
            addVec(r, Ap, 1.0, -alpha);
            float_type r2_new = dotVec(r, r);
            float_type f2 = dotVec(force_target, force_target);
            if (world.rank() == 1)
            {
                std::cout << "r2/f2 = " << r2_new / f2 << " f2 = " << f2 << std::endl;
            }

            if (std::sqrt(r2_new/f2) < cg_threshold_) {
                if (!use_filter) {
                    return;
                }
                else {
                clean<r_i_type>();
                lsolver.template smearing<r_i_type>(force_target, false);
                product<r_i_type, w_1_type, w_2_type>();
                auto Ap = r;
                cleanVec(Ap,false);
                lsolver.template projection<w_2_type>(Ap);

                force_target = Ap;

                return;
                }
            }
            float_type beta = r2_new/r2_old;
            //p = r+beta*p;
            addVec(p, r, beta, 1.0);

        }
    }


    template<class VecType>
    void addVec(VecType& a, VecType& b, float_type w1, float_type w2)
    {
        float_type s = 0;
        for (std::size_t i=0; i<a.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank()) {
                for (std::size_t d=0; d<a[0].size(); ++d) {
                    a[i][d] = 0;
                }
                continue;
            }

            for (std::size_t d=0; d<a[0].size(); ++d)
                a[i][d] =a[i][d]*w1 + b[i][d]*w2;
        }
    }

    template<class VecType>
    void cleanVec(VecType& a, bool nonloc = true)
    {

        for (std::size_t i=0; i<a.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank()) {
                for (std::size_t d=0; d<a[0].size(); ++d) {
                    a[i][d] = 0;
                }
                continue;
            }

            if (!nonloc)
            {
                for (std::size_t d = 0; d < a[0].size(); ++d) { a[i][d] = 0; }
            }
        }
    }

    template<class VecType>
    float_type dotVec(VecType& a, VecType& b)
    {
        float_type s = 0;
        for (std::size_t i=0; i<a.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank())
                continue;

            for (std::size_t d=0; d<a[0].size(); ++d)
                s+=a[i][d]*b[i][d];
        }

        float_type s_global=0.0;
        boost::mpi::all_reduce(domain_->client_communicator(), s,
                s_global, std::plus<float_type>());
        return s_global;
    }

    void write_timestep()
    {
        boost::mpi::communicator world;
        world.barrier();
        pcout << "- writing at T = " << T_ << ", n = " << n_step_ << std::endl;
        //simulation_->write(fname(n_step_));
        simulation_->writeWithTime(fname(n_step_), T_, dt_);
        //simulation_->domain()->tree()->write("tree_restart.bin");
        world.barrier();
        //simulation_->domain()->tree()->read("tree_restart.bin");
        pcout << "- output writing finished -" << std::endl;
    }

    void write_info()
    {
        if (domain_->is_server())
        {
            std::ofstream ofs(
                simulation_->restart_write_dir() + "/restart_info",
                std::ofstream::out);
            if (!ofs.is_open())
            {
                throw std::runtime_error("Could not open file for info write ");
            }

            ofs.precision(20);
            ofs<<"T = " << T_ << ";" << std::endl;
            ofs<<"adapt_count = " << adapt_count_ << ";" << std::endl;
            ofs<<"cell_aux_max = " << source_max_[0] << ";" << std::endl;
            ofs<<"u_max = " << source_max_[1] << ";" << std::endl;
            ofs<<"restart_n_last = " << restart_n_last_ << ";" << std::endl;
            ofs<<"T_last_vel_refresh = " << T_last_vel_refresh_ << ";" << std::endl;

            ofs.close();
        }
    }

    std::string fname(int _n)
    {
        return fname_prefix_+std::to_string(_n);
    }

    // ----------------------------------------------------------------------
    template<class Field>
    void up_and_down()
    {
        //claen non leafs
        clean<Field>(true);
        this->up<Field>();
        this->down_to_correction<Field>();
    }

    template<class Field>
    void up(bool leaf_boundary_only=false)
    {
        //Coarsification:
        for (std::size_t _field_idx=0; _field_idx<Field::nFields(); ++_field_idx)
            psolver.template source_coarsify<Field,Field>(_field_idx, _field_idx, Field::mesh_type(), false, false, false, leaf_boundary_only);
    }

    template<class Field>
    void down_to_correction()
    {
        // Interpolate to correction buffer
        for (std::size_t _field_idx = 0; _field_idx < Field::nFields();
             ++_field_idx)
            psolver.template intrp_to_correction_buffer<Field, Field>(
                _field_idx, _field_idx, Field::mesh_type(), true, false);
    }

    void adapt(bool coarsify_field=true)
    {
        boost::mpi::communicator world;
        auto                     client = domain_->decomposition().client();

        if (source_max_[0]<1e-10 || source_max_[1]<1e-10) return;

        //adaptation neglect the boundary oscillations
        clean_leaf_correction_boundary<cell_aux_type>(domain_->tree()->base_level(),true,2);
        clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(),true,2);

        world.barrier();

        if (coarsify_field)
        {
            pcout << "Adapt - coarsify" << std::endl;
            if (client)
            {
                //claen non leafs
                clean<u_type>(true);
                this->up<u_type>(false);
                ////Coarsification:
                //for (std::size_t _field_idx=0; _field_idx<u::nFields; ++_field_idx)
                //    psolver.template source_coarsify<u_type,u_type>(_field_idx, _field_idx, u::mesh_type);
            }
        }

        world.barrier();
        pcout<< "Adapt - communication"  << std::endl;
        auto intrp_list = domain_->adapt(source_max_, base_mesh_update_);

        world.barrier();
        pcout << "Adapt - intrp" << std::endl;
        if (client)
        {
            // Intrp
            for (std::size_t _field_idx=0; _field_idx<u_type::nFields(); ++_field_idx)
            {
                for (int l = domain_->tree()->depth() - 2;
                     l >= domain_->tree()->base_level(); --l)
                {
                    client->template buffer_exchange<u_type>(l);

                    domain_->decomposition().client()->
                    template communicate_updownward_assign
                    <u_type, u_type>(l,false,false,-1,_field_idx);
                }

                for (auto& oct : intrp_list)
                {
                    if (!oct || !oct->has_data()) continue;
                    psolver.c_cntr_nli().template nli_intrp_node<u_type, u_type>(oct, u_type::mesh_type(), _field_idx, _field_idx, false, false);
                }
            }
        }
        world.barrier();
        pcout << "Adapt - done" << std::endl;
    }

    void time_step()
    {
        // Initialize IFHERK
        // q_1 = u_type
        boost::mpi::communicator world;
        auto                     client = domain_->decomposition().client();

        T_stage_ = T_;

        ////claen non leafs

        stage_idx_=0;

        // Solve stream function to refresh base level velocity
        mDuration_type t_pad(0);
        TIME_CODE( t_pad, SINGLE_ARG(
                    pcout<< "base level mesh update = "<<base_mesh_update_<< std::endl;
                    if (    base_mesh_update_ ||
                            ((T_-T_last_vel_refresh_)/(Re_*dx_base_*dx_base_) * 3.3>7))
                    {
                        pcout<< "pad_velocity, last T_vel_refresh = "<<T_last_vel_refresh_<< std::endl;
                        T_last_vel_refresh_=T_;

                        if (!domain_->is_client())
                            return;
                        pad_velocity<u_type, u_type>(true);
                        adapt_corr_time_step();
                    }
                    else
                    {
                        if (!domain_->is_client())
                            return;
                        up_and_down<u_type>();
                    }
                    ));
        base_mesh_update_=false;
        pcout<< "pad u      in "<<t_pad.count() << std::endl;

        copy<u_type, q_i_type>();

        // Stage 1
        // ******************************************************************
        pcout << "Stage 1" << std::endl;
        T_stage_ = T_ + dt_*c_[0];
        stage_idx_ = 1;
        clean<g_i_type>();
        clean<d_i_type>();
        clean<cell_aux_type>();
        clean<face_aux_type>();

        nonlinear<u_type, g_i_type>(coeff_a(1, 1) * (-dt_));
        copy<q_i_type, r_i_type>();
        add<g_i_type, r_i_type>();
        lin_sys_with_ib_solve(alpha_[0]);


        // Stage 2
        // ******************************************************************
        pcout << "Stage 2" << std::endl;
        T_stage_ = T_ + dt_*c_[1];
        stage_idx_ = 2;
        clean<r_i_type>();
        clean<d_i_type>();
        clean<cell_aux_type>();

        //cal wii
        //r_i_type = q_i_type + dt(a21 w21)
        //w11 = (1/a11)* dt (g_i_type - face_aux_type)

        add<g_i_type, face_aux_type>(-1.0);
        copy<face_aux_type, w_1_type>(-1.0 / dt_ / coeff_a(1, 1));

        psolver.template apply_lgf_IF<q_i_type, q_i_type>(alpha_[0]);
        psolver.template apply_lgf_IF<w_1_type, w_1_type>(alpha_[0]);

        add<q_i_type, r_i_type>();
        add<w_1_type, r_i_type>(dt_ * coeff_a(2, 1));

        up_and_down<u_i_type>();
        nonlinear<u_i_type, g_i_type>(coeff_a(2, 2) * (-dt_));
        add<g_i_type, r_i_type>();

        lin_sys_with_ib_solve(alpha_[1]);

        // Stage 3
        // ******************************************************************
        pcout << "Stage 3" << std::endl;
        T_stage_ = T_ + dt_*c_[2];
        stage_idx_ = 3;
        clean<d_i_type>();
        clean<cell_aux_type>();
        clean<w_2_type>();

        add<g_i_type, face_aux_type>(-1.0);
        copy<face_aux_type, w_2_type>(-1.0 / dt_ / coeff_a(2, 2));
        copy<q_i_type, r_i_type>();
        add<w_1_type, r_i_type>(dt_ * coeff_a(3, 1));
        add<w_2_type, r_i_type>(dt_ * coeff_a(3, 2));

        psolver.template apply_lgf_IF<r_i_type, r_i_type>(alpha_[1]);

        up_and_down<u_i_type>();
        nonlinear<u_i_type, g_i_type>(coeff_a(3, 3) * (-dt_));
        add<g_i_type, r_i_type>();

        lin_sys_with_ib_solve(alpha_[2]);

        // ******************************************************************
        copy<u_i_type, u_type>();
        copy<d_i_type, p_type>(1.0 / coeff_a(3, 3) / dt_);
        // ******************************************************************
    }





    void adapt_corr_time_step()
    {
        if (!use_adaptation_correction) return;
        // Initialize IFHERK
        // q_1 = u_type
        boost::mpi::communicator world;
        auto                     client = domain_->decomposition().client();

        T_stage_ = T_;

        ////claen non leafs

        stage_idx_=0;

        // Solve stream function to refresh base level velocity
        mDuration_type t_pad(0);
        TIME_CODE( t_pad, SINGLE_ARG(
                    pcout<< "adapt_corr_time_step()" << std::endl;

                    if (!domain_->is_client())
                        return;
                    up_and_down<u_type>();
                    ));
        //base_mesh_update_=false;
        //pcout<< "pad u      in "<<t_pad.count() << std::endl;

        copy<u_type, q_i_type>();

        // Stage 1
        // ******************************************************************
        pcout << "Stage 1" << std::endl;
        T_stage_ = T_;
        stage_idx_ = 1;
        clean<g_i_type>();
        clean<d_i_type>();
        clean<cell_aux_type>();
        clean<face_aux_type>();

        //nonlinear<u_type, g_i_type>(0.0);
        copy<q_i_type, r_i_type>();
        //add<g_i_type, r_i_type>();
        lin_sys_with_ib_solve(0.0, false);


        // Stage 2
        // ******************************************************************
        pcout << "Stage 2" << std::endl;
        T_stage_ = T_;
        stage_idx_ = 2;
        clean<r_i_type>();
        clean<d_i_type>();
        clean<cell_aux_type>();

        //cal wii
        //r_i_type = q_i_type + dt(a21 w21)
        //w11 = (1/a11)* dt (g_i_type - face_aux_type)

        //add<g_i_type, face_aux_type>(-1.0);
        //copy<face_aux_type, w_1_type>(0.0);

        //psolver.template apply_lgf_IF<q_i_type, q_i_type>(0.0);
        //psolver.template apply_lgf_IF<w_1_type, w_1_type>(0.0);

        add<q_i_type, r_i_type>();
        //add<w_1_type, r_i_type>(0.0);

        up_and_down<u_i_type>();
        //nonlinear<u_i_type, g_i_type>(0.0);
        //add<g_i_type, r_i_type>();

        lin_sys_with_ib_solve(0.0, false);

        // Stage 3
        // ******************************************************************
        pcout << "Stage 3" << std::endl;
        T_stage_ = T_;
        stage_idx_ = 3;
        clean<d_i_type>();
        clean<cell_aux_type>();
        clean<w_2_type>();

        //add<g_i_type, face_aux_type>(-1.0);
        //copy<face_aux_type, w_2_type>(0.0);
        copy<q_i_type, r_i_type>();
        //add<w_1_type, r_i_type>(0.0);
        //add<w_2_type, r_i_type>(0.0);

        psolver.template apply_lgf_IF<r_i_type, r_i_type>(0.0);

        up_and_down<u_i_type>();
        //nonlinear<u_i_type, g_i_type>(0.0);
        //add<g_i_type, r_i_type>();

        lin_sys_with_ib_solve(0.0, false);

        // ******************************************************************
        copy<u_i_type, u_type>();
        copy<d_i_type, p_type>(0.0);
        // ******************************************************************
    }

    template <typename F>
    void clean(bool non_leaf_only=false, int clean_width=1) noexcept
    {
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->has_data()) continue;
            if (!it->data().is_allocated()) continue;

            for (std::size_t field_idx = 0; field_idx < F::nFields(); ++field_idx)
            {
                auto& lin_data = it->data_r(F::tag(), field_idx).linalg_data();

                if (non_leaf_only && it->is_leaf() && it->locally_owned())
                {
                    int N = it->data().descriptor().extent()[0];
                    if(domain_->dimension() == 3) {
                    view(lin_data, xt::all(), xt::all(),
                        xt::range(0, clean_width)) *= 0.0;
                    view(lin_data, xt::all(), xt::range(0, clean_width),
                        xt::all()) *= 0.0;
                    view(lin_data, xt::range(0, clean_width), xt::all(),
                        xt::all()) *= 0.0;
                    view(lin_data, xt::range(N + 2 - clean_width, N + 3),
                        xt::all(), xt::all()) *= 0.0;
                    view(lin_data, xt::all(),
                        xt::range(N + 2 - clean_width, N + 3), xt::all()) *=
                        0.0;
                    view(lin_data, xt::all(), xt::all(),
                        xt::range(N + 2 - clean_width, N + 3)) *= 0.0;
                    }
                    else {
                    view(lin_data, xt::all(), xt::range(0, clean_width)) *= 0.0;
                    view(lin_data, xt::range(0, clean_width), xt::all()) *= 0.0;
                    view(lin_data, xt::range(N + 2 - clean_width, N + 3),xt::all()) *= 0.0;
                    view(lin_data, xt::all(),xt::range(N + 2 - clean_width, N + 3)) *=0.0;
                    }
                }
                else
                {
                    //TODO whether to clean base_level correction?
                    std::fill(lin_data.begin(), lin_data.end(), 0.0);
                }
            }
        }
    }

    template <typename F>
    void clean_leaf_correction_boundary(int l, bool leaf_only_boundary=false, int clean_width=1) noexcept
    {
        for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
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

        for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
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
        if (l==domain_->tree()->base_level())
        for (auto it  = domain_->begin(l);
                it != domain_->end(l); ++it)
        {
            if(!it->locally_owned()) continue;
            if(!it->has_data() || !it->data().is_allocated()) continue;
            //std::cout<<it->key()<<std::endl;

            for(std::size_t i=0;i< it->num_neighbors();++i)
            {
                auto it2=it->neighbor(i);
                if ((!it2 || !it2->has_data()) || (leaf_only_boundary && (it2->is_correction() || it2->is_old_correction() )))
                {
                    for (std::size_t field_idx=0; field_idx<F::nFields(); ++field_idx)
                    {
                        domain::Operator::smooth2zero<F>( it->data(), i);
                    }
                }
            }
        }
    }



private:
    float_type coeff_a(int i, int j)const noexcept {return a_[i*(i-1)/2+j-1];}

    void lin_sys_solve(float_type _alpha) noexcept
    {
        auto client=domain_->decomposition().client();

        divergence<r_i_type, cell_aux_type>();

        domain_->client_communicator().barrier();
        mDuration_type t_lgf(0);
        TIME_CODE( t_lgf, SINGLE_ARG(
                    psolver.template apply_lgf<cell_aux_type, d_i_type>();
                    ));
        pcout<< "LGF solved in "<<t_lgf.count() << std::endl;

        gradient<d_i_type,face_aux_type>();
        add<face_aux_type, r_i_type>(-1.0);
        if (std::fabs(_alpha)>1e-4)
        {
            mDuration_type t_if(0);
            domain_->client_communicator().barrier();
            TIME_CODE( t_if, SINGLE_ARG(
                        psolver.template apply_lgf_IF<r_i_type, u_i_type>(_alpha);
                        ));
            pcout<< "IF  solved in "<<t_if.count() << std::endl;
        }
        else
            copy<r_i_type,u_i_type>();
    }


    void lin_sys_with_ib_solve(float_type _alpha, bool write_prev_force = true) noexcept
    {
        auto client=domain_->decomposition().client();

        divergence<r_i_type, cell_aux_type>();

        domain_->client_communicator().barrier();
        mDuration_type t_lgf(0);
        TIME_CODE( t_lgf, SINGLE_ARG(
                    psolver.template apply_lgf<cell_aux_type, d_i_type>();
                    ));
        domain_->client_communicator().barrier();
        pcout<< "LGF solved in "<<t_lgf.count() << std::endl;

        copy<r_i_type, face_aux2_type>();
        gradient<d_i_type,face_aux_type>();
        add<face_aux_type, face_aux2_type>(-1.0);

        // IB
        if (std::fabs(_alpha)>1e-14)
            psolver.template apply_lgf_IF<face_aux2_type, face_aux2_type>(_alpha, MASK_TYPE::IB2xIB);

        domain_->client_communicator().barrier();
        pcout<< "IB IF solved "<<std::endl;
        mDuration_type t_ib(0);
        domain_->ib().force() = domain_->ib().force_prev(stage_idx_);
        //domain_->ib().scales(coeff_a(stage_idx_, stage_idx_));
        TIME_CODE( t_ib, SINGLE_ARG(
                    lsolver.template ib_solve<face_aux2_type>(_alpha, T_stage_);
                    ));

        if (write_prev_force) domain_->ib().force_prev(stage_idx_) = domain_->ib().force();
        //domain_->ib().scales(1.0/coeff_a(stage_idx_, stage_idx_));

        pcout<< "IB  solved in "<<t_ib.count() << std::endl;

        // new presure field
        lsolver.template pressure_correction<d_i_type>();
        gradient<d_i_type, face_aux_type>();

        lsolver.template smearing<face_aux_type>(domain_->ib().force(), false);
        add<face_aux_type, r_i_type>(-1.0);

        if (std::fabs(_alpha)>1e-14)
        {
            mDuration_type t_if(0);
            domain_->client_communicator().barrier();
            TIME_CODE( t_if, SINGLE_ARG(
                        psolver.template apply_lgf_IF<r_i_type, u_i_type>(_alpha);
                        ));
            pcout<< "IF  solved in "<<t_if.count() << std::endl;
        }
        else
            copy<r_i_type,u_i_type>();

        // test -------------------------------------
        //force_type tmp(domain_->ib().force().size(), (0.,0.,0.));
        //lsolver.template projection<u_i_type>(tmp);
        //domain_->ib().communicator().compute_indices();
        //domain_->ib().communicator().communicate(true, tmp);
        //if (comm_.rank()==1)
        //{
        //    lsolver.printvec(tmp, "u");
        //}


    }



    template<class Velocity_in, class Velocity_out>
    void pad_velocity(bool refresh_correction_only=true)
    {
        auto client=domain_->decomposition().client();

        //up_and_down<Velocity_in>();
        clean<Velocity_in>(true);
        this->up<Velocity_in>(false);
        clean<edge_aux_type>();
        clean<stream_f_type>();

        auto dx_base = domain_->dx_base();

        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Velocity_in>(l);

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->locally_owned() || !it->has_data()) continue;
                if(it->is_correction()) continue;
                //if(!it->is_leaf()) continue;

                const auto dx_level =  dx_base/math::pow2(it->refinement_level());
                //if (it->is_leaf())
                domain::Operator::curl<Velocity_in,edge_aux_type>( it->data(),dx_level);
            }
        }

        //clean<Velocity_out>();
        clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(), true, 2);
        //clean_leaf_correction_boundary<edge_aux_type>(l, false,2+stage_idx_);
        psolver.template apply_lgf<edge_aux_type, stream_f_type>(MASK_TYPE::STREAM);

        int l_max = refresh_correction_only ?
        domain_->tree()->base_level()+1 : domain_->tree()->depth();
        for (int l  = domain_->tree()->base_level();
                l < l_max; ++l)
        {
            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->locally_owned() || !it->has_data()) continue;
                //if(!it->is_correction() && refresh_correction_only) continue;

                const auto dx_level =  dx_base/math::pow2(it->refinement_level());
                    domain::Operator::curl_transpose<stream_f_type,Velocity_out>( it->data(),dx_level, -1.0);
            }
        }

        this->down_to_correction<Velocity_out>();
    }



    //TODO maybe to be put directly intor operators:
    template<class Source, class Target>
    void nonlinear(float_type _scale = 1.0) noexcept
    {
        clean<edge_aux_type>();
        clean<Target>();
        clean<face_aux_type>();

        auto       client = domain_->decomposition().client();
        const auto dx_base = domain_->dx_base();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {

            client->template buffer_exchange<Source>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl<Source, edge_aux_type>(
                    it->data(), dx_level);
            }
        }

        clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(), true, 2);
        // add background velocity
        copy<Source, face_aux_type>();
        domain::Operator::add_field_expression<face_aux_type>(domain_, simulation_->frame_vel(), T_stage_, -1.0);

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<edge_aux_type>(l);
            clean_leaf_correction_boundary<edge_aux_type>(l, false, 2);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                domain::Operator::nonlinear<face_aux_type, edge_aux_type, Target>(
                    it->data());

                for (std::size_t field_idx = 0; field_idx < Target::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(Target::tag(), field_idx).linalg_data();
                    lin_data *= _scale;
                }
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true,3);
        }

        if (std::abs(b_f_mag) > 1e-5) {
          add_body_force<Target>(_scale, T_stage_);
        }


    }

    template<class target>
    void add_body_force(float_type scale, float_type t) noexcept {
        //float_type eps = 1e-3;
        auto dx_base = domain_->dx_base();
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
                {

                        if (!it->locally_owned()) continue;

                        auto dx_level = dx_base / std::pow(2, it->refinement_level());
                        auto scaling = std::pow(2, it->refinement_level());

                        for (auto& node : it->data())
                        {

                                const auto& coord = node.level_coordinate();


                                float_type x = static_cast<float_type>
                                        (coord[0]) * dx_level;
                                float_type y = static_cast<float_type>
                                        (coord[1]) * dx_level;
                                float_type z = y;
                                if (dim == 3) z = static_cast<float_type>(coord[2])*dx_level;

                                //node(edge_aux,0) = vor(x,y-0.5*vort_sep,0)+ vor(x,y+0.5*vort_sep,0);
                                //node(target::tag(), 0) += -scale * b_f_mag * y / (y*y + b_f_eps);
                                //node(target::tag(), 1) += -scale * b_f_mag * y / (y*y + b_f_eps);
                                //node(target::tag(), 2) += -scale * b_f_mag * y / (y*y + b_f_eps);

                                float_type R = std::pow(x*x+y*y,0.5);
                                float_type Theta = std::atan2(y,x);
                                float_type pert_top = at1*std::cos(Theta*1.0+pt1)+at2*std::cos(Theta*2.0+pt2)+at3*std::cos(Theta*3.0+pt3)+at4*std::cos(Theta*4.0+pt4)+at5*std::cos(Theta*5.0+pt5)+at6*std::cos(Theta*6.0+pt6)+at7*std::cos(Theta*7.0+pt7)+at8*std::cos(Theta*8.0+pt8)+at9*std::cos(Theta*9.0+pt9)+at10*std::cos(Theta*10.0+pt10)+at11*std::cos(Theta*11.0+pt11)+at12*std::cos(Theta*12.0+pt12)+at13*std::cos(Theta*13.0+pt13)+at14*std::cos(Theta*14.0+pt14)+at15*std::cos(Theta*15.0+pt15)+at16*std::cos(Theta*16.0+pt16)+at17*std::cos(Theta*17.0+pt17)+at18*std::cos(Theta*18.0+pt18)+at19*std::cos(Theta*19.0+pt19)+at20*std::cos(Theta*20.0+pt20)+at21*std::cos(Theta*21.0+pt21)+at22*std::cos(Theta*22.0+pt22)+at23*std::cos(Theta*23.0+pt23)+at24*std::cos(Theta*24.0+pt24)+at25*std::cos(Theta*25.0+pt25)+at26*std::cos(Theta*26.0+pt26)+at27*std::cos(Theta*27.0+pt27)+at28*std::cos(Theta*28.0+pt28)+at29*std::cos(Theta*29.0+pt29)+at30*std::cos(Theta*30.0+pt30)+at31*std::cos(Theta*31.0+pt31)+at32*std::cos(Theta*32.0+pt32)+at33*std::cos(Theta*33.0+pt33)+at34*std::cos(Theta*34.0+pt34)+at35*std::cos(Theta*35.0+pt35)+at36*std::cos(Theta*36.0+pt36)+at37*std::cos(Theta*37.0+pt37)+at38*std::cos(Theta*38.0+pt38)+at39*std::cos(Theta*39.0+pt39)+at40*std::cos(Theta*40.0+pt40)+at41*std::cos(Theta*41.0+pt41)+at42*std::cos(Theta*42.0+pt42)+at43*std::cos(Theta*43.0+pt43)+at44*std::cos(Theta*44.0+pt44)+at45*std::cos(Theta*45.0+pt45)+at46*std::cos(Theta*46.0+pt46)+at47*std::cos(Theta*47.0+pt47)+at48*std::cos(Theta*48.0+pt48)+at49*std::cos(Theta*49.0+pt49)+at50*std::cos(Theta*50.0+pt50)+at51*std::cos(Theta*51.0+pt51)+at52*std::cos(Theta*52.0+pt52)+at53*std::cos(Theta*53.0+pt53)+at54*std::cos(Theta*54.0+pt54)+at55*std::cos(Theta*55.0+pt55)+at56*std::cos(Theta*56.0+pt56)+at57*std::cos(Theta*57.0+pt57)+at58*std::cos(Theta*58.0+pt58)+at59*std::cos(Theta*59.0+pt59)+at60*std::cos(Theta*60.0+pt60)+at61*std::cos(Theta*61.0+pt61)+at62*std::cos(Theta*62.0+pt62)+at63*std::cos(Theta*63.0+pt63)+at64*std::cos(Theta*64.0+pt64)+at65*std::cos(Theta*65.0+pt65)+at66*std::cos(Theta*66.0+pt66)+at67*std::cos(Theta*67.0+pt67)+at68*std::cos(Theta*68.0+pt68)+at69*std::cos(Theta*69.0+pt69)+at70*std::cos(Theta*70.0+pt70)+at71*std::cos(Theta*71.0+pt71)+at72*std::cos(Theta*72.0+pt72)+at73*std::cos(Theta*73.0+pt73)+at74*std::cos(Theta*74.0+pt74)+at75*std::cos(Theta*75.0+pt75)+at76*std::cos(Theta*76.0+pt76)+at77*std::cos(Theta*77.0+pt77)+at78*std::cos(Theta*78.0+pt78)+at79*std::cos(Theta*79.0+pt79)+at80*std::cos(Theta*80.0+pt80)+at81*std::cos(Theta*81.0+pt81)+at82*std::cos(Theta*82.0+pt82)+at83*std::cos(Theta*83.0+pt83)+at84*std::cos(Theta*84.0+pt84)+at85*std::cos(Theta*85.0+pt85)+at86*std::cos(Theta*86.0+pt86)+at87*std::cos(Theta*87.0+pt87)+at88*std::cos(Theta*88.0+pt88)+at89*std::cos(Theta*89.0+pt89)+at90*std::cos(Theta*90.0+pt90)+at91*std::cos(Theta*91.0+pt91)+at92*std::cos(Theta*92.0+pt92)+at93*std::cos(Theta*93.0+pt93)+at94*std::cos(Theta*94.0+pt94)+at95*std::cos(Theta*95.0+pt95)+at96*std::cos(Theta*96.0+pt96)+at97*std::cos(Theta*97.0+pt97)+at98*std::cos(Theta*98.0+pt98)+at99*std::cos(Theta*99.0+pt99)+at100*std::cos(Theta*100.0+pt100);
                                //float_type pert_top = a_dipole/100.0*std::cos(Theta*16.0);
                                float_type pert_bot = ab1*std::cos(Theta*1.0+pb1)+ab2*std::cos(Theta*2.0+pb2)+ab3*std::cos(Theta*3.0+pb3)+ab4*std::cos(Theta*4.0+pb4)+ab5*std::cos(Theta*5.0+pb5)+ab6*std::cos(Theta*6.0+pb6)+ab7*std::cos(Theta*7.0+pb7)+ab8*std::cos(Theta*8.0+pb8)+ab9*std::cos(Theta*9.0+pb9)+ab10*std::cos(Theta*10.0+pb10)+ab11*std::cos(Theta*11.0+pb11)+ab12*std::cos(Theta*12.0+pb12)+ab13*std::cos(Theta*13.0+pb13)+ab14*std::cos(Theta*14.0+pb14)+ab15*std::cos(Theta*15.0+pb15)+ab16*std::cos(Theta*16.0+pb16)+ab17*std::cos(Theta*17.0+pb17)+ab18*std::cos(Theta*18.0+pb18)+ab19*std::cos(Theta*19.0+pb19)+ab20*std::cos(Theta*20.0+pb20)+ab21*std::cos(Theta*21.0+pb21)+ab22*std::cos(Theta*22.0+pb22)+ab23*std::cos(Theta*23.0+pb23)+ab24*std::cos(Theta*24.0+pb24)+ab25*std::cos(Theta*25.0+pb25)+ab26*std::cos(Theta*26.0+pb26)+ab27*std::cos(Theta*27.0+pb27)+ab28*std::cos(Theta*28.0+pb28)+ab29*std::cos(Theta*29.0+pb29)+ab30*std::cos(Theta*30.0+pb30)+ab31*std::cos(Theta*31.0+pb31)+ab32*std::cos(Theta*32.0+pb32)+ab33*std::cos(Theta*33.0+pb33)+ab34*std::cos(Theta*34.0+pb34)+ab35*std::cos(Theta*35.0+pb35)+ab36*std::cos(Theta*36.0+pb36)+ab37*std::cos(Theta*37.0+pb37)+ab38*std::cos(Theta*38.0+pb38)+ab39*std::cos(Theta*39.0+pb39)+ab40*std::cos(Theta*40.0+pb40)+ab41*std::cos(Theta*41.0+pb41)+ab42*std::cos(Theta*42.0+pb42)+ab43*std::cos(Theta*43.0+pb43)+ab44*std::cos(Theta*44.0+pb44)+ab45*std::cos(Theta*45.0+pb45)+ab46*std::cos(Theta*46.0+pb46)+ab47*std::cos(Theta*47.0+pb47)+ab48*std::cos(Theta*48.0+pb48)+ab49*std::cos(Theta*49.0+pb49)+ab50*std::cos(Theta*50.0+pb50)+ab51*std::cos(Theta*51.0+pb51)+ab52*std::cos(Theta*52.0+pb52)+ab53*std::cos(Theta*53.0+pb53)+ab54*std::cos(Theta*54.0+pb54)+ab55*std::cos(Theta*55.0+pb55)+ab56*std::cos(Theta*56.0+pb56)+ab57*std::cos(Theta*57.0+pb57)+ab58*std::cos(Theta*58.0+pb58)+ab59*std::cos(Theta*59.0+pb59)+ab60*std::cos(Theta*60.0+pb60)+ab61*std::cos(Theta*61.0+pb61)+ab62*std::cos(Theta*62.0+pb62)+ab63*std::cos(Theta*63.0+pb63)+ab64*std::cos(Theta*64.0+pb64)+ab65*std::cos(Theta*65.0+pb65)+ab66*std::cos(Theta*66.0+pb66)+ab67*std::cos(Theta*67.0+pb67)+ab68*std::cos(Theta*68.0+pb68)+ab69*std::cos(Theta*69.0+pb69)+ab70*std::cos(Theta*70.0+pb70)+ab71*std::cos(Theta*71.0+pb71)+ab72*std::cos(Theta*72.0+pb72)+ab73*std::cos(Theta*73.0+pb73)+ab74*std::cos(Theta*74.0+pb74)+ab75*std::cos(Theta*75.0+pb75)+ab76*std::cos(Theta*76.0+pb76)+ab77*std::cos(Theta*77.0+pb77)+ab78*std::cos(Theta*78.0+pb78)+ab79*std::cos(Theta*79.0+pb79)+ab80*std::cos(Theta*80.0+pb80)+ab81*std::cos(Theta*81.0+pb81)+ab82*std::cos(Theta*82.0+pb82)+ab83*std::cos(Theta*83.0+pb83)+ab84*std::cos(Theta*84.0+pb84)+ab85*std::cos(Theta*85.0+pb85)+ab86*std::cos(Theta*86.0+pb86)+ab87*std::cos(Theta*87.0+pb87)+ab88*std::cos(Theta*88.0+pb88)+ab89*std::cos(Theta*89.0+pb89)+ab90*std::cos(Theta*90.0+pb90)+ab91*std::cos(Theta*91.0+pb91)+ab92*std::cos(Theta*92.0+pb92)+ab93*std::cos(Theta*93.0+pb93)+ab94*std::cos(Theta*94.0+pb94)+ab95*std::cos(Theta*95.0+pb95)+ab96*std::cos(Theta*96.0+pb96)+ab97*std::cos(Theta*97.0+pb97)+ab98*std::cos(Theta*98.0+pb98)+ab99*std::cos(Theta*99.0+pb99)+ab100*std::cos(Theta*100.0+pb100);
                                //float_type pert_bot = a_dipole/100.0*std::cos(Theta*16.0);

                                if (dim == 2)
                                  {
                                    pert_top = 0.0;
                                    pert_bot = 0.0;
                                    Theta = 0.0;
                                    if (x < 0.0) Theta = 3.14159265358979323846;
                                  }

                                /*
                                if ( ((R>1.0) && (R<1.2)) && (t<1.0)) //&& (z>(-0.1+pert)) && (z<(0.1-pert)) )
                                  {
                                    //node(target::tag(), 0) += -scale * b_f_mag * std::cos(Theta);
                                    //node(target::tag(), 1) += -scale * b_f_mag * std::sin(Theta);
                                  node(target::tag(), 0) += -scale * b_f_mag * std::cos(Theta) * (1.0+std::tanh(100.0*(z+pert+0.05)))/2.0 * (1.0-std::tanh(100.0*(z-pert-0.05)))/2.0;
                                  node(target::tag(), 1) += -scale * b_f_mag * std::sin(Theta) * (1.0+std::tanh(100.0*(z+pert+0.05)))/2.0 * (1.0-std::tanh(100.0*(z-pert-0.05)))/2.0;
                                  node(target::tag(), 2) += -scale * b_f_mag * 0.0;
                                  }
                                else
                                  {
                                  node(target::tag(), 0) += -scale * b_f_mag * 0.0;
                                  node(target::tag(), 1) += -scale * b_f_mag * 0.0;
                                  node(target::tag(), 2) += -scale * b_f_mag * 0.0;
                                  }
                                */

                                //This is the case for making circular vortex pairs circa March 2025
                                /*
                                if (t<t_dipole)
                                  {
                                  node(target::tag(), 0) += -scale * b_f_mag * std::cos(Theta) * (1.0+std::tanh(100.0*(z+pert_bot+h_dipole/2)))/2.0 * (1.0-std::tanh(100.0*(z-pert_top-h_dipole/2)))/2.0 * (1.0+std::tanh(100.0*(R-R_dipole+l_dipole/2)))/2.0 * (1.0-std::tanh(100.0*(R-R_dipole-l_dipole/2)))/2.0;
                                  node(target::tag(), 1) += -scale * b_f_mag * std::sin(Theta) * (1.0+std::tanh(100.0*(z+pert_bot+h_dipole/2)))/2.0 * (1.0-std::tanh(100.0*(z-pert_top-h_dipole/2)))/2.0 * (1.0+std::tanh(100.0*(R-R_dipole+l_dipole/2)))/2.0 * (1.0-std::tanh(100.0*(R-R_dipole-l_dipole/2)))/2.0;
                                  if (dim == 3) node(target::tag(), 2) += -scale * b_f_mag * 0.0;
                                  }
                                else
                                  {
                                  node(target::tag(), 0) += -scale * b_f_mag * 0.0;
                                  node(target::tag(), 1) += -scale * b_f_mag * 0.0;
                                  if (dim == 3) node(target::tag(), 2) += -scale * b_f_mag * 0.0;
                                  }
                                */

                                //This is messing around with the downward vortex pair, circa March 19 2025
                                /*
                                if ( ((t<x)&&(x<(t+0.4))) && ((-1.0<y)&&(y<1.0)) && ((-0.05<z)&&(z<0.05)) )
                                  {
                                    node(target::tag(), 2) += -scale * b_f_mag * 1.0;
                                  }
                                else
                                  {
                                     node(target::tag(), 2) += -scale * b_f_mag * 0.0;
                                  }*/

                                //node(target::tag(), 2) += -scale *b_f_mag * (1.0+std::tanh(50.0*(x-t+0.4/2.0)))/2.0 * (1.0-std::tanh(50.0*(x-t-0.4/2.0)))/2.0 * (1.0+std::tanh(3.0*(y+2.0*(1.0+0.1*std::sin(1.0*t))/2.0)))/2.0 * (1.0-std::tanh(3.0*(y-2.0*(1.0+0.1*std::sin(1.0*t))/2.0)))/2.0 * (1.0+std::tanh(50.0*(z+0.1*std::sin(1.0*t)+0.1/2.0)))/2.0 * (1.0-std::tanh(50.0*(z+0.1*std::sin(1.0*t)-0.1/2.0)))/2.0;
                                node(target::tag(), 2) += -scale *b_f_mag * (1.0+std::tanh(tansteepx*(x-speedx*t+widthx/2.0)))/2.0 * (1.0-std::tanh(tansteepx*(x-speedx*t-widthx/2.0)))/2.0 * (1.0+std::tanh(tansteepy*(y+widthy*(1.0+amplitudey*std::sin(periody*t))/2.0)))/2.0 * (1.0-std::tanh(tansteepy*(y-widthy*(1.0+amplitudey*std::sin(periody*t))/2.0)))/2.0 * (1.0+std::tanh(tansteepz*(z+amplitudez*std::sin(periodz*t)+widthz/2.0)))/2.0 * (1.0-std::tanh(tansteepz*(z+amplitudez*std::sin(periodz*t)-widthz/2.0)))/2.0;


                                /*
                                if (t<t_dipole && std::abs(y)<0.5 && std::abs(x)<0.5)
                                  {
                                  node(target::tag(), 0) += -scale * b_f_mag;
                                  node(target::tag(), 1) += 0.0 * -scale * b_f_mag * std::sin(Theta) * (1.0+std::tanh(100.0*(z+pert+h_dipole/2)))/2.0 * (1.0-std::tanh(100.0*(z-pert-h_dipole/2)))/2.0 * (1.0+std::tanh(100.0*(R-R_dipole+l_dipole/2)))/2.0 * (1.0-std::tanh(100.0*(R-R_dipole-l_dipole/2)))/2.0;
                                  if (dim == 3) node(target::tag(), 2) += -scale * b_f_mag * 0.0;
                                  }
                                else
                                  {
                                  node(target::tag(), 0) += -scale * b_f_mag * 0.0;
                                  node(target::tag(), 1) += -scale * b_f_mag * 0.0;
                                  if (dim == 3) node(target::tag(), 2) += -scale * b_f_mag * 0.0;
                                  }
                                */


            }

                }
    }

    template<class Source, class Target>
    void divergence() noexcept
    {
        auto client = domain_->decomposition().client();

        up_and_down<Source>();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::divergence<Source, Target>(
                    it->data(), dx_level);
            }

            //client->template buffer_exchange<Target>(l);
            clean_leaf_correction_boundary<Target>(l, true, 2);
            //clean_leaf_correction_boundary<Target>(l, false,4+stage_idx_);
        }
    }

    template<class Source, class Target>
    void gradient(float_type _scale = 1.0) noexcept
    {
        //up_and_down<Source>();
        domain::Operator::domainClean<Target>(domain_);

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            auto client = domain_->decomposition().client();
            //client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::gradient<Source, Target>(
                    it->data(), dx_level);
                for (std::size_t field_idx = 0; field_idx < Target::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(Target::tag(), field_idx).linalg_data();

                    lin_data *= _scale;
                }
            }
            client->template buffer_exchange<Target>(l);
        }
    }


    template<class Source, class Target>
    void laplacian() noexcept
    {
        auto client = domain_->decomposition().client();

        domain::Operator::domainClean<Target>(domain_);

        clean<edge_aux_type>();

        up_and_down<Source>();

        /*for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::laplace<Source, Target>(it->data(), dx_level);
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true, 2);
            //clean_leaf_correction_boundary<Target>(l, false,4+stage_idx_);
        }*/


        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf() && !it->is_correction()) continue;
                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl<Source, edge_aux_type>(it->data(), dx_level);
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true, 2);
            //clean_leaf_correction_boundary<Target>(l, false,4+stage_idx_);
        }

        this->up<edge_aux_type>();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<edge_aux_type>(l);
            const auto dx_base = domain_->dx_base();

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf() && !it->is_correction()) continue;
                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl_transpose<edge_aux_type, Target>(it->data(), dx_level, -1.0);
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true, 2);
            //clean_leaf_correction_boundary<Target>(l, false,4+stage_idx_);
        }

        //clean_leaf_correction_boundary<Target>(domain_->tree()->base_level(), true, 2);

        //clean<Source>(true);
        //clean<Target>(true);
    }

    template<typename From, typename To>
    void add(float_type scale = 1.0) noexcept
    {
        static_assert(From::nFields() == To::nFields(),
            "number of fields doesn't match when add");
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            for (std::size_t field_idx = 0; field_idx < From::nFields();
                 ++field_idx)
            {
                it->data_r(To::tag(), field_idx)
                    .linalg()
                    .get()
                    ->cube_noalias_view() +=
                    it->data_r(From::tag(), field_idx).linalg_data() * scale;
            }
        }
    }

    template<typename Field>
    void computeWii() noexcept
    {
        auto dx_base = domain_->dx_base();
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
                {

                        if (!it->locally_owned()) continue;

                        auto dx_level = dx_base / std::pow(2, it->refinement_level());
                        auto scaling = std::pow(2, it->refinement_level());

            for (std::size_t field_idx = 0; field_idx < Field::nFields();
                 ++field_idx)
            {
                for (auto& n:it->data().node_field()) {
                    float_type val = n(Field::tag(), field_idx);

                    if (std::fabs(val) > 1e-4) {
                        n(Field::tag(), field_idx) = 1/val;
                    }
                }
            }
                }
    }

    template<typename From1, typename From2, typename To>
    void product() noexcept
    {
        static_assert(From1::nFields() == To::nFields(),
            "number of fields doesn't match when copy");

        static_assert(From2::nFields() == To::nFields(),
            "number of fields doesn't match when copy");

        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            for (std::size_t field_idx = 0; field_idx < From1::nFields();
                 ++field_idx)
            {
                for (auto& n:it->data().node_field())
                    n(To::tag(), field_idx) = n(From1::tag(), field_idx) * n(From2::tag(), field_idx);
            }
        }
    }

    template<typename From, typename To>
    void copy(float_type scale = 1.0) noexcept
    {
        static_assert(From::nFields() == To::nFields(),
            "number of fields doesn't match when copy");

        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            for (std::size_t field_idx = 0; field_idx < From::nFields();
                 ++field_idx)
            {
                for (auto& n:it->data().node_field())
                    n(To::tag(), field_idx) = n(From::tag(), field_idx) * scale;
            }
        }
    }

  private:
    simulation_type* simulation_;
    domain_type*     domain_; ///< domain
    poisson_solver_t psolver;
    linsys_solver_t lsolver;


    bool base_mesh_update_=false;
    int dim = 3;

    float_type T_, T_stage_, T_max_;
    float_type dt_base_, dt_, dx_base_;
    float_type Re_;
    float_type cfl_max_, cfl_;
    std::vector<float_type> source_max_{0.0, 0.0};

    float_type cg_threshold_;
    int  cg_max_itr_;

    float_type T_last_vel_refresh_=0.0;

    float_type b_f_mag, b_f_eps, t_dipole, R_dipole, h_dipole, l_dipole, a_dipole, k_dipole;

    float_type tansteepx, tansteepy, tansteepz, widthx, widthy, widthz, speedx, periody, periodz, amplitudey, amplitudez;

    float_type at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11, at12, at13, at14, at15, at16, at17, at18, at19, at20, at21, at22, at23, at24, at25, at26, at27, at28, at29, at30, at31, at32, at33, at34, at35, at36, at37, at38, at39, at40, at41, at42, at43, at44, at45, at46, at47, at48, at49, at50, at51, at52, at53, at54, at55, at56, at57, at58, at59, at60, at61, at62, at63, at64, at65, at66, at67, at68, at69, at70, at71, at72, at73, at74, at75, at76, at77, at78, at79, at80, at81, at82, at83, at84, at85, at86, at87, at88, at89, at90, at91, at92, at93, at94, at95, at96, at97, at98, at99, at100;

    float_type pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10, pt11, pt12, pt13, pt14, pt15, pt16, pt17, pt18, pt19, pt20, pt21, pt22, pt23, pt24, pt25, pt26, pt27, pt28, pt29, pt30, pt31, pt32, pt33, pt34, pt35, pt36, pt37, pt38, pt39, pt40, pt41, pt42, pt43, pt44, pt45, pt46, pt47, pt48, pt49, pt50, pt51, pt52, pt53, pt54, pt55, pt56, pt57, pt58, pt59, pt60, pt61, pt62, pt63, pt64, pt65, pt66, pt67, pt68, pt69, pt70, pt71, pt72, pt73, pt74, pt75, pt76, pt77, pt78, pt79, pt80, pt81, pt82, pt83, pt84, pt85, pt86, pt87, pt88, pt89, pt90, pt91, pt92, pt93, pt94, pt95, pt96, pt97, pt98, pt99, pt100;

    float_type ab1, ab2, ab3, ab4, ab5, ab6, ab7, ab8, ab9, ab10, ab11, ab12, ab13, ab14, ab15, ab16, ab17, ab18, ab19, ab20, ab21, ab22, ab23, ab24, ab25, ab26, ab27, ab28, ab29, ab30, ab31, ab32, ab33, ab34, ab35, ab36, ab37, ab38, ab39, ab40, ab41, ab42, ab43, ab44, ab45, ab46, ab47, ab48, ab49, ab50, ab51, ab52, ab53, ab54, ab55, ab56, ab57, ab58, ab59, ab60, ab61, ab62, ab63, ab64, ab65, ab66, ab67, ab68, ab69, ab70, ab71, ab72, ab73, ab74, ab75, ab76, ab77, ab78, ab79, ab80, ab81, ab82, ab83, ab84, ab85, ab86, ab87, ab88, ab89, ab90, ab91, ab92, ab93, ab94, ab95, ab96, ab97, ab98, ab99, ab100;

    float_type pb1, pb2, pb3, pb4, pb5, pb6, pb7, pb8, pb9, pb10, pb11, pb12, pb13, pb14, pb15, pb16, pb17, pb18, pb19, pb20, pb21, pb22, pb23, pb24, pb25, pb26, pb27, pb28, pb29, pb30, pb31, pb32, pb33, pb34, pb35, pb36, pb37, pb38, pb39, pb40, pb41, pb42, pb43, pb44, pb45, pb46, pb47, pb48, pb49, pb50, pb51, pb52, pb53, pb54, pb55, pb56, pb57, pb58, pb59, pb60, pb61, pb62, pb63, pb64, pb65, pb66, pb67, pb68, pb69, pb70, pb71, pb72, pb73, pb74, pb75, pb76, pb77, pb78, pb79, pb80, pb81, pb82, pb83, pb84, pb85, pb86, pb87, pb88, pb89, pb90, pb91, pb92, pb93, pb94, pb95, pb96, pb97, pb98, pb99, pb100;

    int max_vel_refresh_=1;
    int max_ref_level_=0;
    int output_base_freq_;
    int adapt_freq_;
    int tot_base_steps_;
    int n_step_ = 0;
    int restart_n_last_ = 0;
    int nLevelRefinement_;
    int stage_idx_ = 0;

    bool use_filter = true; //if use filter when computing force

    bool use_restart_=false;
    bool just_restarted_=false;
    bool write_restart_=false;
    bool updating_source_max_ = false;
    bool all_time_max_;
    bool use_adaptation_correction;
    int restart_base_freq_;
    int adapt_count_;

    std::string                       fname_prefix_;
    vector_type<float_type, 6>        a_{{1.0 / 3, -1.0, 2.0, 0.0, 0.75, 0.25}};
    vector_type<float_type, 4>        c_{{0.0, 1.0 / 3, 1.0, 1.0}};
    //vector_type<float_type, 6>        a_{{1.0 / 2, sqrt(3)/3, (3-sqrt(3))/3, (3+sqrt(3))/6, -sqrt(3)/3, (3+sqrt(3))/6}};
    //vector_type<float_type, 4>        c_{{0.0, 0.5, 1.0, 1.0}};
    vector_type<float_type, 3>        alpha_{{0.0, 0.0, 0.0}};
    parallel_ostream::ParallelOstream pcout =
        parallel_ostream::ParallelOstream(1);
    boost::mpi::communicator comm_;

};

} // namespace solver
} // namespace iblgf

#endif // IBLGF_INCLUDED_POISSON_HPP