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

#ifndef IBLGF_INCLUDED_STABILITY_SOLVER_HPP
#define IBLGF_INCLUDED_STABILITY_SOLVER_HPP




#include <iblgf/solver/DirectSolver/MKLPardiso_solve.hpp>
//need c2f
//#include "mpi.h"
//#include "/home/root/intel-oneAPI/oneAPI/mkl/latest/include/mkl.h"
//#include "/home/root/intel-oneAPI/oneAPI/mkl/latest/include/mkl_cluster_sparse_solver.h"

//need those defined so that xTensor does not load its own CBLAS and resulting in conflicts
//#define CXXBLAS_DRIVERS_MKLBLAS_H
//#define CXXBLAS_DRIVERS_CBLAS_H
#define CXXLAPACK_CXXLAPACK_CXX
#define WITH_MKLBLAS 1
#define CXXBLAS_DRIVERS_MKLBLAS_H 1
#define UNDEF_XT_CBLAS
//Need to add line to undef HAVE_CBLAS at driver.h in xtensor-blas

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
#include <iblgf/solver/Newton/NewtonMethod.hpp>




namespace iblgf
{
//const int Dim = 2;

/*struct parameters
{
    static constexpr std::size_t Dim = 2;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
     (
        //name               type        Dim   lBuffer  hBuffer, storage type
         (fu               , float_type, 2,    1,       1,     face,true  ),
		 (fp               , float_type, 1,    1,       1,     cell,true  ),
         (fw               , float_type, 1,    1,       1,     edge,true  ),
         (u                , float_type, 2,    1,       1,     face,true  ),
		 (p                , float_type, 1,    1,       1,     cell,true  ),
         (test             , float_type, 1,    1,       1,     cell,false )
    ))
    // clang-format on
};*/
namespace solver
{
using namespace domain;

template<class Setup>
class Stability
{
    public:
    
    using solver_t = iblgf::solver::IntelPardisoSolve<float_type>;
    using time_integration_t = typename Setup::time_integration_t;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

	using simulation_type = typename Setup::simulation_t;
	using domain_type = typename simulation_type::domain_type;
	using ib_t = typename domain_type::ib_t;
    using force_type = typename ib_t::force_type;

	using real_coordinate_type = typename domain_type::real_coordinate_type;

    using fu_i_type = typename Setup::fu_i_type;
    using fp_i_type = typename Setup::fp_i_type;
    using fw_i_type = typename Setup::fw_i_type;

    using cell_aux_type = typename Setup::cell_aux_type;
    using face_aux_type = typename Setup::face_aux_type;
    using cell_aux2_type = typename Setup::cell_aux2_type;
    using face_aux2_type = typename Setup::face_aux2_type;
    
	Stability(simulation_type* _simulation)
    : simulation_(_simulation)
    , domain_(_simulation->domain_.get())
    , ifherk(simulation_) 
	{

        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);
        
        clean_p_tar = simulation_->dictionary()->template get_or<bool>("clean_p_tar", false);
        num_input = simulation_->dictionary()->template get_or<bool>("num_input", false);

        Newton_max_itr_ = simulation_->dictionary_->template get_or<int>("Newton_max_itr", 100);
        Newton_threshold_ = simulation_->dictionary_->template get_or<float_type>("Newton_threshold",1e-3);

        
	}

    

    template<class U_old>
    int Init_Construct_Newton_Matrix() {
        //Initially, need to construct all the matrices
        //Jac_p is the sum of everything except DN
        boost::mpi::communicator world;

        

		//time_integration_t ifherk(&this->simulation_);

        ifherk.Assigning_idx();

        if (world.rank() != 0) ifherk.template pad_velocity<U_old, U_old>(true);

        
        
		world.barrier();

        ifherk.construct_Jac_p();
        world.barrier();
        if (world.rank() == 1) {
            std::cout << "start construct DN" << std::endl;
        }
        ifherk.template construct_Jac_from_Jac_p<U_old>();
        ifherk.Jac.clean_entry(1e-10);

        int ndim = ifherk.total_dim();

        std::vector<MKL_INT> iparm;

        
        iparm.resize(64);

        for (int i = 0; i < 64; i++) {
            iparm[i] = 0;
        }        

		iparm[0] = 1; /* Solver default parameters overriden with provided by iparm */
        //iparm[1] = 0;  /* Use METIS for fill-in reordering */
        iparm[1] = 2;  /* Use METIS for fill-in reordering */
        //iparm[1] = 10;  /* Use MPI METIS for fill-in reordering */
        //iparm[1] = 3;  /* Use Parallel METIS for fill-in reordering */
        iparm[5] = 0;  /* Write solution into x */
        iparm[7] = 5;  /* Max number of iterative refinement steps, negative means using quad precision */
        iparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
        iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
        iparm[12] = 1; /* Switch on Maximum Weighted Matching algorithm */
        //iparm[12] = 0; /* Switch off Maximum Weighted Matching algorithm */
        iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
        iparm[18] = -1; /* Output: Mflops for LU factorization */
        //iparm[23] = 10;
        iparm[26] = 1;  /* Check input data for correctness */ 
        iparm[39] = 2; /* Input: matrix/rhs/solution are distributed between MPI processes  */

        Pardiso.set_parameters(iparm, ndim);

        return 0;
    }

    template<class U_old>
    int Update_Newton_Matrix() {
        //only update DN since it involves U_old
        boost::mpi::communicator world;

		//time_integration_t ifherk(&this->simulation_);

        if (world.rank() != 0) ifherk.template pad_velocity<U_old, U_old>(true);

        ifherk.template construct_Jac_from_Jac_p<U_old>();
        ifherk.Jac.clean_entry(1e-12);

        return 0;
    }

    template<class Face, class Cell, class Edge>
    float_type NewtonUpdate(force_type& forcing_vec, float_type& res_inf, float_type& state_err) {
        //update using Newton Iteration
        boost::mpi::communicator world;
        //if (world.rank() != 0) ifherk.template clean_up_initial_velocity<Face>();
        /* RHS and solution vectors. */
        float_type* b = NULL;
        float_type* x = NULL;

        float_type* x_old = NULL;

        force_type forcing_tmp;

        real_coordinate_type tmp_coord(0.0);
        forcing_tmp.resize(domain_->ib().size());

        ifherk.template NewtonRHS<Face, Cell, fu_i_type, fp_i_type>(forcing_vec, forcing_tmp);

        //if (world.rank() != 0) ifherk.template Curl_access<Face, fw_i_type>();

        ifherk.template clean<face_aux_type>();
        ifherk.template clean<cell_aux_type>();

        ifherk.template clean<face_aux2_type>();
        ifherk.template clean<cell_aux2_type>();

        float_type u1_inf = this->compute_errors<fu_i_type, face_aux_type, face_aux2_type>(
			std::string("fu_0_"), 0);
		float_type u2_inf = this->compute_errors<fu_i_type, face_aux_type, face_aux2_type>(
			std::string("fu_1_"), 1);
		float_type p_inf = this->compute_errors<fp_i_type, cell_aux_type, cell_aux2_type>(
			std::string("fp_0_"), 0);

        if (world.rank() != 0) {
            int begin_row = ifherk.num_start();
            int end_row = ifherk.num_end();

            int loc_size = end_row - begin_row+1;
            int check_size = ifherk.Jac.numRow_loc();

            if (loc_size != check_size) {
                std::cout << "Rank " << world.rank() << " local matrix size does not match " << loc_size << " vs " << check_size << std::endl;
                return 1;
            }

            int tot_size = ifherk.Jac.tot_size();
            x     = (float_type*)MKL_malloc(sizeof(float_type) * loc_size, 64);
            x_old = (float_type*)MKL_malloc(sizeof(float_type) * loc_size, 64);
            b     = (float_type*)MKL_malloc(sizeof(float_type) * loc_size, 64);
            for (int k =0 ; k < loc_size; k++) {
                x_old[k] = 0;
                x[k] = 0;
                b[k] = 0;
            }

            
            ifherk.template Grid2CSR<Face, Cell, Edge>(x_old, forcing_vec, false);
            ifherk.template Grid2CSR<fu_i_type, fp_i_type, fw_i_type>(b, forcing_tmp);

        }

        Pardiso.load_matrix(ifherk.Jac, ifherk);
        world.barrier();
        if (world.rank() == 1) {
            std::cout << "matrix loaded" << std::endl;
        }
        
        Pardiso.load_RHS(ifherk.Jac, ifherk, b);
        world.barrier();
        if (world.rank() == 1) {
            std::cout << "RHSloaded" << std::endl;
        }
        Pardiso.reordering();
        world.barrier();
        if (world.rank() == 1) {
            std::cout << "Reordering complete" << std::endl;
        }
        Pardiso.factorization();
        Pardiso.back_substitution();
        Pardiso.getSolution(ifherk.Jac, ifherk, x);
        Pardiso.release_internal_mem();
        Pardiso.FreeSolver();

        float_type res_loc = 0.0;
        float_type linf_loc = -1.0;
        float_type f2_loc = 0.0;

        if (world.rank() != 0) {
            
            int begin_row = ifherk.num_start();
            int end_row = ifherk.num_end();

            int loc_size = end_row - begin_row+1;
            int check_size = ifherk.Jac.numRow_loc();

            for (int k =0 ; k < loc_size; k++) {
                //if (std::abs(x[k]) > 1e6) continue;
                x_old[k] -= x[k];
                res_loc += (x[k]*x[k]);
                f2_loc += x_old[k]*x_old[k];
                if (std::abs(x[k]) > linf_loc) {
                    linf_loc = std::abs(x[k]);
                }
            }
        }

        float_type res_val = ifherk.GetStateMag(x);
        float_type state_val = ifherk.GetStateMag(x_old);

        state_err = res_val/state_val;

        ifherk.template CSR2Grid<Face, Cell, Edge>(x_old, forcing_vec);

        float_type res_glob = 0.0;
        float_type linf_glob = 0.0;
        float_type f2_glob = 0.0;

        boost::mpi::all_reduce(world, res_loc, res_glob, std::plus<float_type>());
        boost::mpi::all_reduce(world, f2_loc, f2_glob, std::plus<float_type>());
        boost::mpi::all_reduce(world, linf_loc, linf_glob, boost::mpi::maximum<float_type>());

        res_inf = linf_glob;

        return (res_glob/f2_glob);
    }

    template<class Face, class Cell, class Edge>
    void NewtonIteration(force_type& forcing_vec) {
        //Newton iteration
        boost::mpi::communicator world;
        float_type linf_err;
        float_type state_err;

        if (forcing_vec.size() > 0)
        {
            real_coordinate_type    tmp_coord(0.0);
            std::vector<float_type> f(domain_->dimension(), 0.);
            force_type sum_f(domain_->ib().force().size(), tmp_coord);
            //std::vector<float_type> f(domain_->dimension(), 0.);
            for (std::size_t d = 0; d < domain_->dimension(); ++d)
            {
                for (std::size_t i = 0; i < forcing_vec.size(); ++i)
                {
                    if (world.rank() != domain_->ib().rank(i))
                        forcing_vec[i][d] = 0.0;
                }
            }

            boost::mpi::all_reduce(domain_->client_communicator(),
                &forcing_vec[0], forcing_vec.size(), &sum_f[0],
                std::plus<real_coordinate_type>());

            for (std::size_t d = 0; d < domain_->dimension(); ++d)
            {
                for (std::size_t i = 0; i < forcing_vec.size(); ++i)
                {
                    //if (world.rank() != domain_->ib().rank(i)) continue;
                    f[d] += sum_f[i][d] * domain_->ib().force_scale();
                }
            }
            //boost::mpi::all_reduce(world, &ib.force(0), ib.size(), &sum_f[0], std::plus<std::vector<float_type>>());
            if (world.rank() == 1)
            {
                std::cout << "ib  size: " << forcing_vec.size() << std::endl;
                std::cout << "Forcing = ";

                for (std::size_t d = 0; d < domain_->dimension(); ++d)
                    std::cout << f[d] << " ";
                std::cout << std::endl;

                std::cout << " -----------------" << std::endl;
            }
        }

        this->template Init_Construct_Newton_Matrix<Face>();

        for (int i = 0; i < Newton_max_itr_;i++) {
            float_type err = this->template NewtonUpdate<Face, Cell, Edge>(forcing_vec, linf_err, state_err);

            if (forcing_vec.size() > 0)
            {
                real_coordinate_type tmp_coord(0.0);
                std::vector<float_type> f(domain_->dimension(), 0.);
                force_type sum_f(domain_->ib().force().size(), tmp_coord);
                //std::vector<float_type> f(domain_->dimension(), 0.);
                for (std::size_t d = 0; d < domain_->dimension(); ++d) {
                    for (std::size_t i = 0; i < forcing_vec.size(); ++i) {
                        if (world.rank() != domain_->ib().rank(i)) forcing_vec[i][d] = 0.0;
                    }
                }

                boost::mpi::all_reduce(domain_->client_communicator(), &forcing_vec[0], forcing_vec.size(), &sum_f[0], std::plus<real_coordinate_type>());

                for (std::size_t d = 0; d < domain_->dimension(); ++d) {
                    for (std::size_t i = 0; i < forcing_vec.size(); ++i) {
                        //if (world.rank() != domain_->ib().rank(i)) continue;
                        f[d] += sum_f[i][d] * domain_->ib().force_scale();
                    }
                }
                //boost::mpi::all_reduce(world, &ib.force(0), ib.size(), &sum_f[0], std::plus<std::vector<float_type>>());
                if (world.rank() == 1) {
                    std::cout << "ib  size: " << forcing_vec.size() << std::endl;
                    std::cout << "Forcing = ";

                    for (std::size_t d = 0; d < domain_->dimension(); ++d)
                        std::cout << f[d] << " ";
                    std::cout << std::endl;

                    std::cout << " -----------------" << std::endl;
                }
            }

            if (world.rank() == 1) {
                std::cout << "L2  Res of Newton Iteration is " << err << std::endl;
                std::cout << "L2  Res physical var of Newton Iteration is " << state_err << std::endl;
                std::cout << "Inf Res of Newton Iteration is " << linf_err << std::endl;
            }
            if (std::sqrt(state_err) < Newton_threshold_) {
                write_restart();
                break;
            }


            std::string prefix_v = "Newton_itr";
            std::string idx_str = std::to_string(i);
            std::string post_fix = ".hdf5";
            std::string dest_name = prefix_v+idx_str;
            simulation_->write(dest_name);
            this->template Update_Newton_Matrix<Face>();
        }
    }

    void write_restart()
    {
        boost::mpi::communicator world;

        world.barrier();
        if (domain_->is_server())
        {
            std::cout << "restart: backup" << std::endl;
            simulation_->copy_restart();
        }
        world.barrier();

        if (world.rank() == 1) std::cout << "restart: write" << std::endl;
        simulation_->write("", true);

        //write_info();
        world.barrier();
    }

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

                

                if (std::fabs(tmp_exact) > LInf_exact)
                    LInf_exact = std::fabs(tmp_exact);

                if (std::fabs(error_tmp) > LInf) LInf = std::fabs(error_tmp);

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

        pcout_c << "Global " << _output_prefix
                << "L2 = " << std::sqrt(L2_global) << std::endl;
        pcout_c << "Global " << _output_prefix << "LInf = " << LInf_global
                << std::endl;



        return LInf_global;
    }


    private:

    boost::mpi::communicator client_comm_;
    solver_t Pardiso;

    simulation_type* simulation_;
    domain_type*     domain_;

    time_integration_t ifherk;
    int Newton_max_itr_;
    float_type Newton_threshold_;

    parallel_ostream::ParallelOstream pcout_c =
        parallel_ostream::ParallelOstream(1);


    bool num_input;

    bool clean_p_tar;
};
} // namespace solver
} // namespace iblgf

#endif
