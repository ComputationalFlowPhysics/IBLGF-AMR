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

#ifndef IBLGF_INCLUDED_VORTEXRINGS_HPP
#define IBLGF_INCLUDED_VORTEXRINGS_HPP
#define USE_OMP

//#ifdef IBLGF_VORTEX_RUN_ALL

#define POISSON_TIMINGS

#include <iostream>
#include <chrono>
#include <omp.h>
//#include <iblgf/lgf/lgf.hpp>
//#include <iblgf/lgf/helmholtz.hpp>

// IBLGF-specific
#include "../../setups/setup_helmholtz_OMP.hpp"

namespace iblgf
{
const int Dim = 2;

struct parameters
{
    static constexpr std::size_t Dim = 2;
	static constexpr std::size_t N_modes = 32;
	static constexpr std::size_t PREFAC  = 2; //2 for complex values 
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
     (
         //name, type, nFields, l/h-buf,mesh_obj, output(optional)
         (phi_num          ,float_type, 1*2*N_modes,    1, 1,   cell),
         (source           ,float_type, 1*2*N_modes,    1, 1,   cell),
         (phi_exact        ,float_type, 1*2*N_modes,    1, 1,   cell),
         (phi_num_f        ,float_type, 3*2*N_modes,    1, 1,   face),
         (source_f         ,float_type, 3*2*N_modes,    1, 1,   face),
         (phi_exact_f      ,float_type, 3*2*N_modes,    1, 1,   face),
         (amr_lap_source_f ,float_type, 3*2*N_modes,    1, 1,   face),
         (error_f          ,float_type, 3*2*N_modes,    1, 1,   face),
         (err_lap_source_f ,float_type, 3*2*N_modes,    1, 1,   face),
         (error            ,float_type, 1*2*N_modes,    1, 1,   cell),
         (amr_lap_source   ,float_type, 1*2*N_modes,    1, 1,   cell),
         (error_lap_source ,float_type, 1*2*N_modes,    1, 1,   cell),
         (decomposition    ,float_type, 1*2*N_modes,    1, 1,   cell)
    ))
    // clang-format on
};

struct vortex_ring
{
    float_type vorticity(
        float_type x, float_type y) const noexcept
    {
        x -= center[0];
        y -= center[1];
        float_type r = std::sqrt(x*x + y*y);
        float_type r2 = x*x + y*y;
        return (4.0*r2 - 4.0 - c*c)*std::exp(-r2);
    }

    float_type vorticity_c(
        float_type x, float_type y, float_type c_) const noexcept
    {
        x -= center[0];
        y -= center[1];
        float_type r = std::sqrt(x*x + y*y);
        float_type r2 = x*x + y*y;
        return (4.0*r2 - 4.0 - c_*c_)*std::exp(-r2);
    }

    float_type psi(float_type x, float_type y) const noexcept
    {
        x -= center[0];
        y -= center[1];
        const float_type r = std::sqrt(x * x + y * y);
        return std::exp(-r*r);
    }

  public:
    coordinate_type<float_type, Dim> center;
    float_type                       c;
};

struct VortexRingTest : public Setup_helmholtz_OMP<VortexRingTest, parameters>
{
    using super_type = Setup_helmholtz_OMP<VortexRingTest, parameters>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

    using domain_type = typename super_type::domain_t;

    using tree_t = typename domain_type::tree_t;
    using octant_t = typename tree_t::octant_type;
    using MASK_TYPE = typename octant_t::MASK_TYPE;

    using lgf_lap_t = typename lgf::LGF_GL<2>;
    using lgf_if_t = typename lgf::LGF_GE<2>;

    static constexpr int lBuffer = 1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer = 1; ///< Lower left buffer for interpolation

    VortexRingTest(Dictionary* _d)
    : super_type(_d, [this](auto _d, auto _domain) {
        return this->initialize_domain(_d, _domain);
    })
    , domain_loc(simulation_.domain_.get())
    , psolver(&this->simulation_, N_modes - 1)
    //, H_lgf(0.1)
    {

        float_type dx_base = domain_->dx_base();
        int nLevels = _d->get_dictionary("simulation_parameters")->template get_or<int>("nLevels", 0);
        c_value     = _d->get_dictionary("simulation_parameters")->template get_or<float_type>("c_value", 2.0*M_PI);
        float_type c_z = simulation_.dictionary_->template get_or<float_type>("L_z", 1.0);
        dx_fine     = dx_base*std::pow(0.5, nLevels);
        pcout << "dxfine " << dx_fine << std::endl;
        float_type c_helm = c_value*dx_base;

        domain_->decomposition().OMP_initialize();

        Omegas.resize(N_modes);
        for (int i = 0; i < N_modes; i++) {
            float_type Omega = static_cast<float_type>(i) * 2.0 * M_PI / c_z;
            Omegas[i] = Omega;
        }

        //H_lgf.resize(phi_num_type::nFields());

        for (int i = 0; i < (N_modes - 1); i++) {
            float_type c_v = Omegas[i + 1] * dx_base;
            H_lgf.emplace_back((new lgf::Helmholtz<2>(c_v)));
        }

        for (int i = 0; i < phi_num_type::nFields(); i++) {
            lgf_if_vec.emplace_back((new lgf_if_t()));
            //lgf_if_vec[i]->alpha_base_level() = 0.1;
            //lgf_if_vec[i]->change_level(0);
            //H_lgf.emplace_back((new lgf::Helmholtz<2>(c_helm)));
        }

        pcout << "c_helm is " << c_helm << std::endl;



        vrings_ = this->read_vrings(simulation_.dictionary_.get());
        float_type max_vort = 0.0;
        for (auto& vr : vrings_)
        {
            const auto center = vr.center;
        }

        refinement_factor_ =simulation_.dictionary_->
            template get<float_type>("refinement_factor");
        pcout<<"Refienment factor "<<refinement_factor_<<std::endl;
        /*subtract_non_leaf_ =simulation_.dictionary_->
            template get_or<bool>("subtract_non_leaf", true);*/

        use_correction_ =simulation_.dictionary_->
            template get_or<bool>("correction", true);

        domain_->correction_buffer()= simulation_.dictionary_->
            template get_or<bool>("correction_buffer", true);

        pcout << "\n Setup:  Test - Vortex rings \n" << std::endl;
        pcout << "Number of refinement levels: " << nLevels_ << std::endl;

        domain_->register_refinement_condition() = [this](auto octant,
                                                       int     diff_level) {
            return this->refinement(octant, diff_level);
        };

        domain_->init_refine(_d->get_dictionary("simulation_parameters")
                                 ->template get_or<int>("nLevels", 0),
            global_refinement_, 0);

        

        pcout << "Using correction = " << std::boolalpha << use_correction_
              << std::endl;
        pcout << "Subtract non leaf = " << std::boolalpha << subtract_non_leaf_
              << std::endl;

        domain_->decomposition().subtract_non_leaf() = subtract_non_leaf_;
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        pcout << "Initial distribution done" << std::endl;
        this->initialize();

        boost::mpi::communicator world;
        if (world.rank() == 0)
            std::cout << "on Simulation: \n" << simulation_ << std::endl;
    }

    template<class Dict>
    std::vector<vortex_ring> read_vrings(Dict* _dict)
    {
        std::vector<vortex_ring> vrings;
        auto                     dicts = _dict->get_all_dictionaries("vortex");
        for (auto& d : dicts)
        {
            vortex_ring v_tmp;
            v_tmp.center = d->template get<float_type, 2>("center");
            n            = d->template get<int>("n");
            N            = d->template get<int>("N");
            v_tmp.c      = c_value;
            pcout << "value of c" << v_tmp.c << std::endl;
            vrings.push_back(v_tmp);
        }
        return vrings;
    }

    float_type solve()
    {
        std::ofstream                     ofs, ofs_level, ofs_timings;
        parallel_ostream::ParallelOstream pofs(
            io::output().dir() + "/" + "global_timings.txt", 1, ofs),
            pofs_level(
                io::output().dir() + "/" + "level_timings.txt", 1, ofs_level);

        boost::mpi::communicator world;

        auto pts = domain_->get_nPoints();

        if (domain_->is_client())
        {
            auto             pts = domain_->get_nPoints();
            pcout_c << "Initializing FMM ---------------------------------"
                    << std::endl;
            

            client_comm_.barrier();

            pcout_c << "Finished Initializing FMM ---------------------------------"
                    << std::endl;

            int base_level = domain_->tree()->base_level();
            for (int field_idx = 0; field_idx < amr_lap_source_type::nFields(); field_idx++) {
                psolver.template clean_field<amr_lap_source_type>(field_idx);
            }

            for (int field_idx = 0; field_idx < amr_lap_source_f_type::nFields(); field_idx++) {
                psolver.template clean_field<amr_lap_source_f_type>(field_idx);
            }

            //psolver.use_correction() = use_correction_;
            //psolver.subtract_non_leaf() = subtract_non_leaf_;

            mDuration_type solve_duration(0), solve_duration_if(0), solve_duration_f(0), solve_duration_if2(0);
            client_comm_.barrier();

            pcout_c << "Helmholtz equation ---------------------------------"
                    << std::endl;

            lgf_if_.alpha_base_level() = 0.01;


            //fmm_.template apply_LGF<source_type, phi_num_type>(domain_loc, &lgf_lap_, 1, H_lgf, 1, base_level, true, 1.0, MASK_TYPE::AMR2AMR);
            /*TIME_CODE(solve_duration,
                SINGLE_ARG(psolver.template apply_lgf<source_type, phi_num_type>(&lgf_lap_, H_lgf, 0, MASK_TYPE::AMR2AMR);
                           client_comm_.barrier();))*/

            TIME_CODE(solve_duration,
                SINGLE_ARG(psolver.template apply_lgf_and_helm<source_type, phi_num_type>(N_modes, 1, MASK_TYPE::AMR2AMR);
                           client_comm_.barrier();))
            pcout_c << "Elapsed time for Poisson Solver " << solve_duration.count() / 1.0e3
                    << " Rate " << pts.back() / (solve_duration.count() / 1.0e3)
                    << std::endl;

            TIME_CODE(solve_duration_f,
                SINGLE_ARG(psolver.template apply_lgf_and_helm<source_f_type, phi_num_f_type>(N_modes, 3, MASK_TYPE::AMR2AMR);
                           client_comm_.barrier();))
            TIME_CODE(solve_duration_if,
                SINGLE_ARG(psolver.template apply_helm_if<source_f_type, amr_lap_source_f_type>(0.1, N_modes, 3, MASK_TYPE::AMR2AMR);
                           client_comm_.barrier();))

            TIME_CODE(solve_duration_if2,
                SINGLE_ARG(psolver.template apply_helm_if<source_f_type, amr_lap_source_f_type>(0.1, N_modes, 3, MASK_TYPE::IB2xIB);
                           client_comm_.barrier();))


            

            pcout_c << "Elapsed time for IFHERK Solver " << solve_duration_if.count() / 1.0e3
                    << " Rate " << pts.back() / (solve_duration_if.count() / 1.0e3)
                    << std::endl;

            pcout_c << "Elapsed time for IFHERK Solver IB2xIB " << solve_duration_if2.count() / 1.0e3
                    << " Rate " << pts.back() / (solve_duration_if2.count() / 1.0e3)
                    << std::endl;
            
            pcout_c << "End applying FMM" << std::endl;

#ifdef POISSON_TIMINGS
            //psolver.print_timings(pofs, pofs_level);
#endif
            //psolver.apply_laplace<phi_num_type, amr_lap_source_type>();
        }

        float_type inf_error =
            this->compute_errors<phi_num_type, phi_exact_type, error_type>();

        float_type inf_error2 =
            this->compute_errors<phi_num_type, phi_exact_type, error_type>("phi_err_2", 2);
        float_type inf_errorN =
            this->compute_errors<phi_num_type, phi_exact_type, error_type>("phi_err_N", N_modes - 1);
        float_type inf_error2N =
            this->compute_errors<phi_num_type, phi_exact_type, error_type>("phi_err_2N", 2*N_modes - 1);


        for (int NthField = 0; NthField < 3; NthField++) {
            this->compute_errors<amr_lap_source_f_type, source_f_type, err_lap_source_f_type>("laplace_f_", NthField*2*N_modes);
            this->compute_errors<amr_lap_source_f_type, source_f_type, err_lap_source_f_type>("laplace_f_", NthField*2*N_modes + N_modes-1);
            this->compute_errors<amr_lap_source_f_type, source_f_type, err_lap_source_f_type>("laplace_f_", NthField*2*N_modes + 2 * N_modes-1);

            this->compute_errors<phi_num_f_type, phi_exact_f_type, error_f_type>("error_f_", NthField*2*N_modes);
            this->compute_errors<phi_num_f_type, phi_exact_f_type, error_f_type>("error_f_", NthField*2*N_modes + N_modes-1);
            this->compute_errors<phi_num_f_type, phi_exact_f_type, error_f_type>("error_f_", NthField*2*N_modes + 2 * N_modes-1);
        }

        //simulation_.write("mesh.hdf5");

        return inf_error;
    }

    double run()
    {
        simulation_.write("mesh.hdf5");
        float_type Inf_error = this->solve();
        pcout_c << "Solve 1st time done" << std::endl;
        simulation_.write("mesh.hdf5");
        //pcout_c<<"write" <<std::endl;
        //domain_->decomposition().balance<source_type,phi_exact_type>();
        //pcout_c<<"decompositiondone" <<std::endl;

        this->solve();
        //simulation_.write("mesh_new.hdf5");
        return Inf_error;
    }

    /** @brief Initialization of poisson problem.
     *  @detail Testing poisson with manufactured solutions: exp
     */
    void initialize()
    {
        boost::mpi::communicator world;
        if (domain_->is_server()) return;
        auto center =
            (domain_->bounding_box().max() - domain_->bounding_box().min()) /
                2.0 +
            domain_->bounding_box().min();

        // Adapt center to always have peak value in a cell-center
        //center+=0.5/std::pow(2,nRef);
        const float_type dx_base = domain_->dx_base();

        // Loop through leaves and assign values
        int nLocally_owned = 0;
        int nGhost = 0;
        int nAllocated = 0;
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (it.ptr())
            {
                if (it->locally_owned() && it->has_data()) { ++nLocally_owned; }
                else if (it->has_data())
                {
                    ++nGhost;
                    if (it->data().is_allocated()) ++nAllocated;
                }
            }
        }

        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
             ++it)
        {
            if (!it->locally_owned()) continue;
            if (!(*it && it->has_data())) continue;
            auto dx_level = dx_base / std::pow(2, it->refinement_level());
            auto scaling = std::pow(2, it->refinement_level());

            for (auto& node : it->data())
            {
                for (int field_idx = 0; field_idx < source_type::nFields(); field_idx++) {
                    node(source, field_idx) = 0.0;
                    node(phi_num, field_idx) = 0.0;

                    float_type factor_v = 1.0 + static_cast<float_type>(field_idx) / static_cast<float_type>(source_type::nFields());

                    const auto& coord = node.level_coordinate();

                    // manufactured solution:
                    float_type x = static_cast<float_type>(
                                    coord[0] - center[0] * scaling + 0.5) *
                                dx_level;
                    float_type y = static_cast<float_type>(
                                    coord[1] - center[1] * scaling + 0.5) *
                                dx_level;

                    float_type x_f = static_cast<float_type>(
                                    coord[0] - center[0] * scaling) *
                                dx_level;
                    float_type y_f = static_cast<float_type>(
                                    coord[1] - center[1] * scaling) *
                                dx_level;

                    float_type c_v_ = Omegas[field_idx / 2];
                    node(source, field_idx) = vorticity_c(x, y, c_v_) * factor_v;
                    node(phi_exact, field_idx) = psi(x, y) * factor_v;

                    node(source_f, field_idx) = vorticity_c(x_f, y, c_v_) * factor_v;
                    node(source_f, field_idx + source_type::nFields()) = vorticity_c(x, y_f, c_v_) * factor_v;
                    node(source_f, field_idx + source_type::nFields() * 2) = vorticity_c(x, y, c_v_) * factor_v;

                    node(phi_exact_f, field_idx) = psi(x_f, y) * factor_v;
                    node(phi_exact_f, field_idx + source_type::nFields()) = psi(x, y_f) * factor_v;
                    node(phi_exact_f, field_idx + source_type::nFields() * 2) = psi(x, y) * factor_v;
                }
                /***********************************************************/
            }
        }

        auto       client = domain_->decomposition().client();
        for (int l = domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<source_type>(l);
        }
    }

    float_type vorticity(
        float_type x, float_type y) const noexcept
    {
        float_type vort = 0.0;
        for (auto& vr : vrings_) { vort += vr.vorticity(x, y); }
        return vort;
    }

    float_type vorticity_c(
        float_type x, float_type y, float_type c_) const noexcept
    {
        float_type vort = 0.0;
        for (auto& vr : vrings_) { vort += vr.vorticity_c(x, y, c_); }
        return vort;
    }

    float_type psi(float_type x, float_type y) const noexcept
    {
        float_type psi = 0.0;
        for (auto& vr : vrings_) { psi += vr.psi(x, y); }
        return psi;
    }

    int get_nPoints() const noexcept
    {
        if (!domain_->is_client()) return 0;

        int nPts = 0;
        int nPts_global = 0;
        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
             ++it)
        {
            if (it->has_data()) nPts += it->data().node_field().size();
        }
        boost::mpi::all_reduce(
            client_comm_, nPts, nPts_global, std::plus<int>());
        return nPts_global;
    }

    /** @brief  Refienment conditon for octants.  */
    template<class OctantType>
    bool refinement(
        OctantType* it, int diff_level, bool use_all = false) const noexcept
    {
        auto b = it->data().descriptor();
        b.level() = it->refinement_level();
        const float_type dx_base = domain_->dx_base();

        return refinement(b, dx_base, vorticity_max_, diff_level, use_all);
    }

    /** @brief  Refienment conditon for blocks.  */
    bool refinement(block_descriptor_t b, float_type dx_base,
        float_type vorticity_max, int diff_level,
        bool use_all = false) const noexcept
    {
        auto center =
            (domain_->bounding_box().max() - domain_->bounding_box().min()) /
                2.0 +
            domain_->bounding_box().min();

        auto scaling = std::pow(2, b.level());
        center *= scaling;
        auto dx_level = dx_base / std::pow(2, b.level());

        b.grow(2, 2);
        auto corners = b.get_corners();
        for (int i = b.base()[0]; i <= b.max()[0]; ++i)
        {
            for (int j = b.base()[1]; j <= b.max()[1]; ++j)
            {
                const float_type x =
                    static_cast<float_type>(i - center[0] + 0.5) * dx_level;
                const float_type y =
                    static_cast<float_type>(j - center[1] + 0.5) * dx_level;

                const auto vort = vorticity(x, y);
                if (std::fabs(vort) >
                    vorticity_max_ * pow(refinement_factor_, diff_level))
                {
                    return true;
                }
            }
        }
        return false;
    }

    /** @brief  Initialization of the domain blocks. This is registered in the
     *          domain through the base setup class, passing it to the domain ctor. */
    std::vector<coordinate_t> initialize_domain(
        Dictionary* _d, domain_t* _domain)
    {
        auto res =
            _domain->construct_basemesh_blocks(_d, _domain->block_extent());
        domain_->read_parameters(_d);
        return res;
    }

  private:
    float_type               vorticity_max_ = 1;
    domain_type*             domain_loc;
    std::vector<vortex_ring> vrings_;
    float_type               refinement_factor_ = 1. / 8;
    float_type               dx_fine = 0.0;
    bool                     use_correction_ = true;
    bool                     subtract_non_leaf_ = false;
    int                      n;
    int                      N;
    std::vector<std::unique_ptr<lgf::Helmholtz<2>>>        H_lgf;
    lgf::Helmholtz<2>        H_lgf_s;
    lgf_lap_t                lgf_lap_;
    std::vector<std::unique_ptr<lgf_if_t>> lgf_if_vec;
    lgf_if_t                 lgf_if_;
    std::vector<float_type>  Omegas;
    int                      c_value;

    //poisson_solver_t psolver(&this->simulation_, N_modes - 1);
    poisson_solver_t psolver;
};

//#endif

double vortex_run(std::string input, int argc = 0, char** argv = nullptr);

} // namespace iblgf
#endif // IBLGF_INCLUDED_POISSON_HPP
