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

#ifndef IBLGF_INCLUDED_SETUP_BASE_HELM_HPP
#define IBLGF_INCLUDED_SETUP_BASE_HELM_HPP

#include <iblgf/global.hpp>
#include <iblgf/utilities/crtp.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/lgf/lgf.hpp>
#include <iblgf/fmm/fmm.hpp>
#include <iblgf/domain/dataFields/dataBlock.hpp>
#include <iblgf/utilities/tuple_utilities.hpp>
#include <iblgf/domain/dataFields/datafield.hpp>
#include <iblgf/solver/poisson/poisson.hpp>
#include <iblgf/solver/linsys/linsys.hpp>
#include <iblgf/solver/time_integration/ifherk.hpp>
#include <iblgf/IO/parallel_ostream.hpp>

namespace iblgf
{
using namespace domain;
using namespace octree;
using namespace types;
using namespace fmm;
using namespace dictionary;

/**  @brief Base class for a setup. Provides all the neccessary default fields
 *          and aliases for datablock, domain and simulation.
 */
template<class Setup, class SetupTraits>
class SetupBase
: private crtp::Crtps<Setup, SetupBase<Setup, SetupTraits>>
, public SetupTraits
{
    const int n_ifherk_stage = 3;
    const int n_ifherk_ij = 6;

  public:
    static constexpr std::size_t Dim = SetupTraits::Dim;
    static constexpr std::size_t N_modes = SetupTraits::N_modes*2; 
    //since f(xi) = conj(f(-xi)) for real functions so the number of modes are doubled
    //still need to consider the fact that modes are complex so still keeping the same number of modes but stacking real and imaginary parts together
    //modes are f_{-N + 1}, f_{-N + 2}, ... f_{-1}, f_{0}, f_{1}, ...f_{N - 2}, f_{N - 1} N = SetupTraits::N_modes 
    //so only need f_{0}, f_{1}, ...f_{N - 2}, f_{N - 1} to be stored. With real and complex part (treating f_0 as complex as well). need 2N components 
    //data is like 
    //(Re[v_x_0],Im[v_x_0],
    // Re[v_y_0],Im[v_y_0],
    // Re[v_z_0],Im[v_z_0],
    // Re[v_x_1],Im[v_x_1],
    // Re[v_y_1],Im[v_y_1],
    // Re[v_z_1],Im[v_z_1],
    // Re[v_x_2],Im[v_x_2],
    // Re[v_y_2],Im[v_y_2], ...)

  public: //default fields
    // clang-format off
    REGISTER_FIELDS
    (Dim,
    (
      (source_tmp,          float_type,  1,  1,  1,  cell,false),
      (correction_tmp,      float_type,  1,  1,  1,  cell,false),
      (target_tmp,          float_type,  1,  1,  1,  cell,false),
      (fmm_s,               float_type,  1,  1,  1,  cell,false),
      (fmm_t,               float_type,  1,  1,  1,  cell,false),
      //flow variables
      (q_i,                 float_type,  3*N_modes,  1,  1,  face,true),
      (u_i,                 float_type,  3*N_modes,  1,  1,  face,true),
      (u_i_real,            float_type,  3*N_modes,  1,  1,  face,true),
      (vort_i_real,         float_type,  3*N_modes,  1,  1,  edge,true),
      (r_i_real,            float_type,  3*N_modes,  1,  1,  face,true),
      (d_i,                 float_type,  N_modes,    1,  1,  cell,true),
      (g_i,                 float_type,  3*N_modes,  1,  1,  face,false),
      (r_i,                 float_type,  3*N_modes,  1,  1,  face,false),
      (w_1,                 float_type,  3*N_modes,  1,  1,  face,false),
      (w_2,                 float_type,  3*N_modes,  1,  1,  face,false),
      (cell_aux,            float_type,  N_modes,    1,  1,  cell,true),
      (cell_aux2,           float_type,  N_modes,    1,  1,  cell,true),
      (face_aux,            float_type,  3*N_modes,  1,  1,  face,true),
      (face_aux2,           float_type,  3*N_modes,  1,  1,  face,true),
      (stream_f,            float_type,  N_modes,    1,  1,  edge,true),
      (edge_aux,            float_type,  3*N_modes,  1,  1,  edge,true)
    ))
    // clang-format on

    using field_tuple = fields_tuple_t;

    //Register fields gives us: source_tmp as the tag_type,

  public: //datablock
    template<class... DataFieldType>
    using db_template = domain::DataBlock<Dim, node, DataFieldType...>;
    template<class userFields>
    using datablock_template_t =
        typename tuple_utils::make_from_tuple<db_template,
            typename tuple_utils::concat<field_tuple, userFields>::type>::type;

  public: //Trait types to be used by others
    using user_fields = typename SetupTraits::fields_tuple_t;
    using datablock_t = datablock_template_t<user_fields>;
    using block_descriptor_t = typename datablock_t::block_descriptor_type;
    using coordinate_t = typename datablock_t::coordinate_type;
    using domain_t = domain::Domain<Dim, datablock_t>;
    using domaint_init_f =
        typename domain_t::template block_initialze_fct<Dictionary*>;
    using simulation_t = Simulation<domain_t>;
    using fcoord_t = coordinate_type<float_type, Dim>;

    using Fmm_t = Fmm<SetupBase>;
    using fmm_mask_builder_t = FmmMaskBuilder<domain_t>;
    using poisson_solver_t = solver::PoissonSolver<SetupBase>;
    using time_integration_t = solver::Ifherk<SetupBase>;
    using linsys_solver_t = solver::LinSysSolver<SetupBase>;

  public: //Ctors
    SetupBase(Dictionary* _d)
    : simulation_(_d->get_dictionary("simulation_parameters"))
    , domain_(simulation_.domain())
    {
        domain_->initialize(simulation_.dictionary()->get_dictionary("domain"));
    }

    SetupBase(Dictionary* _d, domaint_init_f _fct,
        std::string restart_tree_dir = "")
    : simulation_(_d->get_dictionary("simulation_parameters"))
    , domain_(simulation_.domain())
    {
        auto d = _d->get_dictionary("simulation_parameters");
        use_restart_ = d->template get_or<bool>("use_restart", true);

        if (restart_tree_dir == "")
        {
            if (!simulation_.restart_dir_exist()) { use_restart_ = false; }
            else
            {
                restart_tree_dir = simulation_.restart_tree_info_dir();
            }
        }
        else
        {
            use_restart_ = true;
        }

        if (!use_restart_)
        {
            domain_->initialize(
                simulation_.dictionary()->get_dictionary("domain").get(), _fct);
        }
        else
        {
            domain_->initialize_with_keys(
                simulation_.dictionary()->get_dictionary("domain").get(),
                restart_tree_dir);
        }

        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);

        nLevels_ = simulation_.dictionary_->template get_or<int>("nLevels", 0);
        global_refinement_ = simulation_.dictionary_->template get_or<int>(
            "global_refinement", 0);
    }

  public: //memebers
    bool use_restart() { return use_restart_; }

    /** @brief Compute L2 and LInf errors */
    template<class Numeric, class Exact, class Error>
    float_type compute_errors(std::string _output_prefix = "",
        int                               field_idx = 0)
    {
        const float_type dx_base = domain_->dx_base();
        float_type       L2 = 0.;
        float_type       LInf = -1.0;
        int              count = 0;
        float_type       L2_exact = 0;
        float_type       LInf_exact = -1.0;

        std::vector<float_type> L2_perLevel(nLevels_ + 1 + global_refinement_,
            0.0);
        std::vector<float_type> L2_exact_perLevel(
            nLevels_ + 1 + global_refinement_, 0.0);
        std::vector<float_type> LInf_perLevel(nLevels_ + 1 + global_refinement_,
            0.0);
        std::vector<float_type> LInf_exact_perLevel(
            nLevels_ + 1 + global_refinement_, 0.0);

        std::vector<int> counts(nLevels_ + 1 + global_refinement_, 0);

        std::ofstream                     ofs, ofs_global;
        parallel_ostream::ParallelOstream pofs(
            io::output().dir() + "/" + _output_prefix + "level_error.txt", 1,
            ofs);
        parallel_ostream::ParallelOstream pofs_global(
            io::output().dir() + "/" + _output_prefix + "global_error.txt", 1,
            ofs_global);

        if (domain_->is_server()) return -1.0;

        for (auto it_t = domain_->begin_leaves(); it_t != domain_->end_leaves();
             ++it_t)
        {
            if (!it_t->locally_owned() || !it_t->has_data()) continue;
            if (it_t->is_correction()) continue;

            int    refinement_level = it_t->refinement_level();
            double dx = dx_base / std::pow(2.0, refinement_level);

            for (auto& node : it_t->data())
            {
                float_type tmp_exact = node(Exact::tag(), field_idx);
                float_type tmp_num = node(Numeric::tag(), field_idx);
                //if (std::fabs(tmp_exact)<1e-6) continue;

                float_type error_tmp = tmp_num - tmp_exact;

                node(Error::tag(), field_idx) = error_tmp;

                // clean inside spehre
                const auto& coord = node.level_coordinate();
                float_type  x = static_cast<float_type>(coord[0]) * dx;
                float_type  y = static_cast<float_type>(coord[1]) * dx;
                float_type  z = 0.0;
                if (domain_->dimension() == 3)
                    z = static_cast<float_type>(coord[2]) * dx;

                if (domain_->dimension() == 3)
                {
                    if (field_idx == 0)
                    {
                        y += 0.5 * dx;
                        z += 0.5 * dx;
                    }
                    else if (field_idx == 1)
                    {
                        x += 0.5 * dx;
                        z += 0.5 * dx;
                    }
                    else
                    {
                        x += 0.5 * dx;
                        y += 0.5 * dx;
                    }

                    float_type r2 = x * x + y * y + z * z;
                    /*if (std::fabs(r2) <= .25)
                {
                    node(Error::tag(), field_idx)=0.0;
                    error_tmp = 0;
                }*/
                }

                if (domain_->dimension() == 2)
                {
                    if (field_idx == 0) { y += 0.5 * dx; }
                    if (field_idx == 1) { x += 0.5 * dx; }
                    float_type r2 = x * x + y * y;
                    /*if (std::fabs(r2) <= 0.25) {
		    	node(Error::tag(), field_idx)=0.0;
			error_tmp = 0.0;
		    }*/
                }
                // clean inside spehre
                float_type weight = std::pow(dx, domain_->dimension());
                L2 += error_tmp * error_tmp * weight;
                L2_exact += tmp_exact * tmp_exact * weight;

                L2_perLevel[refinement_level] += error_tmp * error_tmp * weight;
                L2_exact_perLevel[refinement_level] +=
                    tmp_exact * tmp_exact * weight;
                ++counts[refinement_level];

                if (std::fabs(tmp_exact) > LInf_exact)
                    LInf_exact = std::fabs(tmp_exact);

                if (std::fabs(error_tmp) > LInf) LInf = std::fabs(error_tmp);

                if (std::fabs(error_tmp) > LInf_perLevel[refinement_level])
                    LInf_perLevel[refinement_level] = std::fabs(error_tmp);

                if (std::fabs(tmp_exact) >
                    LInf_exact_perLevel[refinement_level])
                    LInf_exact_perLevel[refinement_level] =
                        std::fabs(tmp_exact);

                ++count;
            }
        }

        float_type L2_global(0.0);
        float_type LInf_global(0.0);

        float_type L2_exact_global(0.0);
        float_type LInf_exact_global(0.0);

        boost::mpi::all_reduce(client_comm_, L2, L2_global,
            std::plus<float_type>());
        boost::mpi::all_reduce(client_comm_, L2_exact, L2_exact_global,
            std::plus<float_type>());

        boost::mpi::all_reduce(client_comm_, LInf, LInf_global,
            boost::mpi::maximum<float_type>());
        boost::mpi::all_reduce(client_comm_, LInf_exact, LInf_exact_global,
            boost::mpi::maximum<float_type>());

        pcout_c << "Glabal " << _output_prefix
                << "L2_exact = " << std::sqrt(L2_exact_global) << std::endl;
        pcout_c << "Global " << _output_prefix
                << "LInf_exact = " << LInf_exact_global << std::endl;

        pcout_c << "Glabal " << _output_prefix
                << "L2 = " << std::sqrt(L2_global) << std::endl;
        pcout_c << "Global " << _output_prefix << "LInf = " << LInf_global
                << std::endl;

        ofs_global << std::sqrt(L2_exact_global) << " " << LInf_exact_global
                   << " " << std::sqrt(L2_global) << " " << LInf_global
                   << std::endl;

        //Level wise errros
        std::vector<float_type> L2_perLevel_global(
            nLevels_ + 1 + global_refinement_, 0.0);
        std::vector<float_type> LInf_perLevel_global(
            nLevels_ + 1 + global_refinement_, 0.0);

        std::vector<float_type> L2_exact_perLevel_global(
            nLevels_ + 1 + global_refinement_, 0.0);
        std::vector<float_type> LInf_exact_perLevel_global(
            nLevels_ + 1 + global_refinement_, 0.0);

        //files

        std::vector<int> counts_global(nLevels_ + 1 + global_refinement_, 0);
        for (std::size_t i = 0; i < LInf_perLevel_global.size(); ++i)
        {
            boost::mpi::all_reduce(client_comm_, counts[i], counts_global[i],
                std::plus<float_type>());
            boost::mpi::all_reduce(client_comm_, L2_perLevel[i],
                L2_perLevel_global[i], std::plus<float_type>());
            boost::mpi::all_reduce(client_comm_, LInf_perLevel[i],
                LInf_perLevel_global[i], boost::mpi::maximum<float_type>());

            boost::mpi::all_reduce(client_comm_, L2_exact_perLevel[i],
                L2_exact_perLevel_global[i], std::plus<float_type>());
            boost::mpi::all_reduce(client_comm_, LInf_exact_perLevel[i],
                LInf_exact_perLevel_global[i],
                boost::mpi::maximum<float_type>());

            pcout_c << _output_prefix << "L2_" << i << " "
                    << std::sqrt(L2_perLevel_global[i]) << std::endl;
            pcout_c << _output_prefix << "LInf_" << i << " "
                    << LInf_perLevel_global[i] << std::endl;
            pcout_c << "count_" << i << " " << counts_global[i] << std::endl;

            pofs << i << " " << std::sqrt(L2_perLevel_global[i]) << " "
                 << LInf_perLevel_global[i] << std::endl;
        }

        return LInf_global;
    }

  protected:
    simulation_t              simulation_; ///< simulation
    std::shared_ptr<domain_t> domain_ =
        nullptr;                             ///< Domain reference for convience
    boost::mpi::communicator client_comm_;   ///< Communicator for clients only
    boost::mpi::communicator world_;         ///< World Communicator
    parallel_ostream::ParallelOstream pcout; ///< parallel cout on master
    parallel_ostream::ParallelOstream pcout_c =
        parallel_ostream::ParallelOstream(1);

    bool use_restart_ = false;
    int  nLevels_ = 0;
    int  global_refinement_;
};

} // namespace iblgf
#endif // IBLGF_INCLUDED_SETUP_BASE_HPP
