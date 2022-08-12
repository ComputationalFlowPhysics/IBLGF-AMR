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

#ifndef IBLGF_INCLUDED_SOLVER_LINSYS_HELM_HPP
#define IBLGF_INCLUDED_SOLVER_LINSYS_HELM_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <complex>
#include <string>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>

#include <cstddef>
#include <iblgf/tensor/vector.hpp>

#include <iblgf/solver/time_integration/ScaLAPACK_ib.hpp>

namespace iblgf
{
namespace solver
{
using namespace domain;

template<class Setup>
class LinSysSolver_helm
{
  public: //member types
    static constexpr std::size_t Dim = Setup::Dim;

    using simulation_type = typename Setup::simulation_t;
    using poisson_solver_t = typename Setup::poisson_solver_t;
    //use all these to obtain values of LGF and IF
    using lgf_lap_t = typename poisson_solver_t::lgf_lap_t;
    using lgf_if_t = typename poisson_solver_t::lgf_if_t;
    using helm_t = typename poisson_solver_t::helm_t;

    using domain_type = typename simulation_type::domain_type;
    using datablock_type = typename domain_type::datablock_t;
    using tree_t = typename domain_type::tree_t;
    using octant_t = typename tree_t::octant_type;
    using MASK_TYPE = typename octant_t::MASK_TYPE;
    using block_type = typename datablock_type::block_descriptor_type;
    using ib_t = typename domain_type::ib_t;
    using real_coordinate_type = typename domain_type::real_coordinate_type;

    using key_2D = std::tuple<int, int>;
    //using force_type = typename ib_t::force_type;
    
    using coordinate_type = typename domain_type::coordinate_type;
    // FMM
    using Fmm_t = typename Setup::Fmm_t;
    using u_type = typename Setup::u_type;
    using r_i_type = typename Setup::r_i_type;
    using cell_aux2_type = typename Setup::cell_aux2_type;
    using face_aux_type = typename Setup::face_aux_type;
    using face_aux2_type = typename Setup::face_aux2_type;

    static constexpr std::size_t N_modes = Setup::N_modes; //number of complex modes

    using ModesVector_type = types::vector_type<float_type, N_modes>;


    using point_force_type = types::vector_type<float_type, 3 * 2 * N_modes>;
    using force_type = std::vector<point_force_type>;

  public:
    LinSysSolver_helm(simulation_type* simulation)
    : simulation_(simulation)
    , domain_(simulation->domain_.get())
    , ib_(&domain_->ib())
    , psolver_(simulation, N_modes - 1)
    {
        cg_threshold_ = simulation_->dictionary_->template get_or<float_type>("cg_threshold",1e-3);
        cg_max_itr_ = simulation_->dictionary_->template get_or<int>("cg_max_itr", 40);
        c_z = simulation_->dictionary()->template get_or<float_type>(
            "L_z", 1.0);
        /*float_type dx_base_ = domain_->dx_base();
        const int l_max = domain_->tree()->depth();
        const int l_min = domain_->tree()->base_level();
        const int nLevels = l_max - l_min;
        c_z = dx_base_*N_modes*2/std::pow(2.0, nLevels - 1);*/

        additional_modes = simulation_->dictionary()->template get_or<int>("add_modes", N_modes - 1);

        usingDirectSolve = simulation_->dictionary()->template get_or<bool>("useDirectSolve", false);

        support_ext_factor = simulation_->dictionary()->template get_or<int>("support_ext_factor", 2);

        _compute_Modes.resize(N_modes);
        for (int i = 0; i < N_modes; i++) {
            this->_compute_Modes[i] = true;
        }

    }

    void Initializing_solvers(vector_type<float_type, 3> alpha_)
    {
        point_force_type tmp(0.0);

        force_type tmp_f(domain_->ib().size(), tmp);

        int force_dim = tmp_f.size()*(u_type::nFields())/N_modes;
        for (int i = 0; i < 3; i++)
        {
            //this->creating_matrix(matrix_ib, alpha_[i]);
            if (i == 0) { DirectSolvers.emplace_back(simulation_); }
            else
            {
                std::vector<int> localModes_ =
                    DirectSolvers[0].getlocal_modes();
                DirectSolvers.emplace_back(simulation_, localModes_, false);
            }

            int err = DirectSolvers[i].settingUpMatrix();

            for (int row_n = 0; row_n < force_dim; row_n++) {
                force_type row;
                creating_matrix_row(row, alpha_[i], row_n);
                err = DirectSolvers[i].load_matrix_row(row, row_n);
            }
            err = DirectSolvers[i].final_assemble();
            //DirectSolvers[i].load_matrix(matrix_ib);
        }
        //matrix_ib.clear();
        //matrix_ib.shrink_to_fit(); //resizing to free up the memory
    }

    void DirectSolve(int stage_idx, force_type& forcing, force_type& res) {
        DirectSolvers[stage_idx].load_RHS(forcing, false);
        DirectSolvers[stage_idx].getSolution(res);

        boost::mpi::communicator world;

        if (world.rank() != 0)
        {
            for (int i = 0; i < res.size(); i++)
            {
                if (domain_->ib().rank(i) != world.rank())
                {
                    for (int j = 0; j < res[0].size(); j++) { res[i][j] = 0; }
                }
            }
        }
    }


    void creating_matrix_row(force_type& matrix_force_glob, float_type alpha, int k) {
        //k is the row to construct
        boost::mpi::communicator world;

        if (world.rank() == 0) return;

        point_force_type tmp(0.0);

        force_type tmp_f(domain_->ib().size(), tmp);

        int force_dim = tmp_f.size()*(u_type::nFields())/N_modes;

        

        int num = k;

        force_type Ap(domain_->ib().size(), tmp);

        matrix_force_glob = Ap;

        if (world.rank() != 0) {
            if (world.rank() == 1)
            {
                if (num % ((u_type::nFields())/N_modes) == 0)
                {
                    std::cout << "Constructing matrix for IB, alpha = " << alpha << " number solved " << num << " over "
                              << force_dim << std::endl;
                }
            }
            for (int i = 0; i < tmp_f.size(); i++)
            {
                for (int j = 0; j < tmp_f[0].size(); j++) { tmp_f[i][j] = 0.0; }
            }
            int ib_idx = num / ((u_type::nFields())/N_modes);
            int field_idx = num % ((u_type::nFields())/N_modes);
            int idx_complex = field_idx/2; //the number of components (zero for u, one for v, two for w)
            int realcomp = field_idx % 2;  //zero if real part but one if complex part

            if (domain_->ib().rank(ib_idx)==world.rank()) {
                for (int ModeN = 0; ModeN < N_modes; ModeN++) {
                    int field_idx_now = idx_complex*N_modes*2 + ModeN*2 + realcomp;
                    tmp_f[ib_idx][field_idx_now] = 1.0;
                }
                //tmp_f[ib_idx][field_idx] = 1.0;
            }
            /*for (int i = 0; i < tmp_f.size(); i++)
            {
                if (domain_->ib().rank(i)!=world.rank()) continue;
                tmp_f[i][num] = 1.0;
            }*/
            //force_type Ap(domain_->ib().size(), tmp);
            for (int i = 0; i < Ap.size(); i++)
            {
                for (int j = 0; j < Ap[0].size(); j++) { Ap[i][j] = 0.0; }
            }
            this->template ET_H_S_E<face_aux2_type>(tmp_f, Ap, alpha);
            for (int i = 0; i < Ap.size(); i++)
            {
                if (domain_->ib().rank(i)!=world.rank()) {
                    for (int j = 0; j < Ap[0].size(); j++) { Ap[i][j] = 0.0; }
                }
            }
        }

        boost::mpi::all_reduce(domain_->client_communicator(), &Ap[0], domain_->ib().size(), &matrix_force_glob[0], std::plus<point_force_type>());
    }

    void creating_matrix_direct(std::vector<force_type>& matrix_force_glob, float_type alpha) {
        /*boost::mpi::communicator world;
        std::map<key_2D, float_type> lgf_mat; //will store LGF for each Fourier mode
        std::map<key_2D, float_type> if_mat; //will store integrating factor for each Fourier mode
        std::map<key_2D, float_type> grad, div;
        std::vector<std::map<key_2D, float_type>> smearing;

        float_type safe_dist_ib = domain_->ib().safety_dis_;
        float_type ib_radius = domain_->ib().ddf_radius();
        float_type added_radius = ib_radius+safety_dist_ib+10.0; //copied directly from ib_overlap function in ib.hpp

        int conv_range = std::ceil(added_radius) * 2; 
        //times 2 just to be safe since we are no longer have all the redundent points in ib extended blocks

        float_type dx_ib = domain_->ib().dx_ib_;

        lgf_lap_t   lgf_lap_;
        lgf_if_t    lgf_if_;
        lgf_if_.alpha_base_level() = alpha;
        //helm_t      lgf_helm;
        Conv_matrix if_mat(conv_range, conv_range, dx_ib, lgf_if_);
        Conv_matrix lgf_mat(conv_range, conv_range, dx_ib, lgf_lap_);

        const float_type dx_base = domain_->dx_base();

        std::vector<std::map<key_2D, float_type>> smearing_loc_x = smearing_matrix(true, 0);
        std::vector<std::map<key_2D, float_type>> smearing_loc_y = smearing_matrix(true, 1);
        std::vector<std::map<key_2D, float_type>> smearing_loc_z = smearing_matrix(true, 2);

        std::vector<std::map<key_2D, float_type>> smearing_all_x = smearing_matrix(false, 0);
        std::vector<std::map<key_2D, float_type>> smearing_all_y = smearing_matrix(false, 1);
        std::vector<std::map<key_2D, float_type>> smearing_all_z = smearing_matrix(false, 2);

        std::string div = "div";
        std::string grad = "grad";

        auto dxP0 = dx_matrix(smearing_loc_x, 0, div, dx_ib);
        auto dxP1 = dx_matrix(smearing_loc_y, 1, div, dx_ib);
        auto div_all_0 = dxP1;

        add_matrix(dxP0, div_all_0, 1, 1);
        //add_matrix(dxP1, div_all_0);

        auto L_inv_app = lgf_mat.apply_mat(div_all_0);

        auto grad_0 = dx_matrix(L_inv_app, 0, grad, dx_ib);
        auto grad_1 = dx_matrix(L_inv_app, 1, grad, dx_ib);

        auto pp0 = smearing_loc_x;
        auto pp1 = smearing_loc_y;
        auto pp2 = smearing_loc_z;
        add_matrix(grad_0, pp0, -1, 1);
        add_matrix(grad_1, pp1, -1, 1);

        auto h0 = if_mat.apply_mat(pp0);
        auto h1 = if_mat.apply_mat(pp1);
        auto h2 = if_mat.apply_mat(pp2);

        auto res0 = denseProd(smearing_all_x, h0);
        auto res1 = denseProd(smearing_all_y, h1);
        auto res2 = denseProd(smearing_all_z, h2);*/

        /*for (int i = 1; i < N_modes; i++) {
            float_type c = (static_cast<float_type>(i)) * 2.0 * M_PI * dx_base/c_z;
            helm_t lgf_helm(c);
            Conv_matrix helm_mat(conv_range, conv_range, dx_ib, lgf_helm);
        }*/
    }


    void testing_nth(std::vector<std::vector<std::vector<float_type>>>& res, float_type alpha, int N=0) {
        //not really worth doing, much slower than direct construct
        boost::mpi::communicator world;

        res.resize(9);
        
        float_type safe_dist_ib = domain_->ib().safety_dis_;
        float_type ib_radius = domain_->ib().ddf_radius();
        float_type added_radius = ib_radius+safe_dist_ib+10.0; //copied directly from ib_overlap function in ib.hpp

        int ib_range = std::ceil(added_radius) * support_ext_factor; 
        
        //times 2 just to be safe since we are no longer have all the redundent points in ib extended blocks

        float_type dx_ib = domain_->ib().dx_ib_;


        int conv_range = ib_range + std::ceil(1.0/dx_ib);

        const float_type dx_base = domain_->dx_base();

        float_type c = static_cast<float_type>(N) * 2.0 * M_PI * dx_base/c_z;

        if (N == 0) {
            c = 0.1;
        }

        lgf_lap_t   lgf_lap_;
        lgf_if_t    lgf_if_;
        helm_t      helm(c);
        lgf_if_.alpha_base_level() = alpha;
        //helm_t      lgf_helm;
        Conv_matrix if_mat(ib_range, ib_range, 1.0, lgf_if_, simulation_);
        Conv_matrix lgf_mat(conv_range, conv_range, dx_ib, lgf_lap_, simulation_);
        Conv_matrix helm_mat(conv_range, conv_range, dx_ib, helm, simulation_);

        

        std::vector<std::map<key_2D, float_type>> smearing_loc_x = smearing_matrix(true, 0);
        std::vector<std::map<key_2D, float_type>> smearing_loc_y = smearing_matrix(true, 1);
        std::vector<std::map<key_2D, float_type>> smearing_loc_z = smearing_matrix(true, 2);


        std::vector<std::map<key_2D, float_type>> smearing_all_x = smearing_matrix(false, 0);
        std::vector<std::map<key_2D, float_type>> smearing_all_y = smearing_matrix(false, 1);
        std::vector<std::map<key_2D, float_type>> smearing_all_z = smearing_matrix(false, 2);


        auto h0 = if_mat.apply_mat(smearing_loc_x);
        auto h1 = if_mat.apply_mat(smearing_loc_y);
        auto h2 = if_mat.apply_mat(smearing_loc_z);

        std::string div = "div";
        std::string grad = "grad";

        float_type omega = 2.0*M_PI*static_cast<float_type>(N)/c_z;

        float_type alpha_level = lgf_if_.alpha_;
        float_type helm_weights = std::exp(-omega*omega*alpha_level * dx_ib * dx_ib); 

        if (N != 0) {
            h0 = scale_matrix_real(h0, helm_weights);
            h1 = scale_matrix_real(h1, helm_weights);
            h2 = scale_matrix_real(h2, helm_weights);
        }
        //this needs to be called after initializing conv_matrix so that appropriate alpha is set

        //float_type omega = static_cast<float_type>(idx + 1) * 2.0 * M_PI / L_z;

        auto dxP0 = dx_matrix(h0, 0, div, dx_ib);
        auto dxP1 = dx_matrix(h1, 1, div, dx_ib);
        auto dxP2 = scale_matrix_real(h2, -omega);

        std::vector<std::map<key_2D, float_type>> L_inv_app_0;
        std::vector<std::map<key_2D, float_type>> L_inv_app_1;
        std::vector<std::map<key_2D, float_type>> L_inv_app_2;

        if (N == 0)
        {
            L_inv_app_0 = lgf_mat.apply_mat(dxP0);
            L_inv_app_1 = lgf_mat.apply_mat(dxP1);
            L_inv_app_2 = lgf_mat.apply_mat(dxP2);
        }
        else {
            L_inv_app_0 = helm_mat.apply_mat(dxP0);
            L_inv_app_1 = helm_mat.apply_mat(dxP1);
            L_inv_app_2 = helm_mat.apply_mat(dxP2);
        }

        auto grad_00 = dx_matrix(L_inv_app_0, 0, grad, dx_ib);
        auto grad_10 = dx_matrix(L_inv_app_0, 1, grad, dx_ib);
        
        auto grad_01 = dx_matrix(L_inv_app_1, 0, grad, dx_ib);
        auto grad_11 = dx_matrix(L_inv_app_1, 1, grad, dx_ib);

        auto grad_02 = dx_matrix(L_inv_app_2, 0, grad, dx_ib);
        auto grad_12 = dx_matrix(L_inv_app_2, 1, grad, dx_ib);
        
        auto grad_20 = scale_matrix_real(L_inv_app_0, omega);
        auto grad_21 = scale_matrix_real(L_inv_app_1, omega);
        auto grad_22 = scale_matrix_real(L_inv_app_2, omega);


        auto pp0 = h0;
        auto pp1 = h1;
        auto pp2 = h2;
        add_matrix(grad_00, pp0, -1, 1);
        add_matrix(grad_11, pp1, -1, 1);
        add_matrix(grad_22, pp2, -1, 1);


        grad_01 = scale_matrix_real(grad_01, -1);
        grad_02 = scale_matrix_real(grad_02, -1);
        grad_10 = scale_matrix_real(grad_10, -1);
        grad_12 = scale_matrix_real(grad_12, -1);
        grad_20 = scale_matrix_real(grad_20, -1);
        grad_21 = scale_matrix_real(grad_21, -1);

        /*auto h00 = if_mat.apply_mat(pp0);
        auto h01 = if_mat.apply_mat(grad_01);
        auto h02 = if_mat.apply_mat(grad_02);

        auto h10 = if_mat.apply_mat(grad_10);
        auto h11 = if_mat.apply_mat(pp1);
        auto h12 = if_mat.apply_mat(grad_12);

        auto h20 = if_mat.apply_mat(grad_20);
        auto h21 = if_mat.apply_mat(grad_21);
        auto h22 = if_mat.apply_mat(pp2);*/


        res[0] = denseProd(smearing_all_x, pp0); //00
        res[1] = denseProd(smearing_all_y, grad_10); //10
        res[2] = denseProd(smearing_all_z, grad_20); //20

        res[3] = denseProd(smearing_all_x, grad_01); //01
        res[4] = denseProd(smearing_all_y, pp1); //11
        res[5] = denseProd(smearing_all_z, grad_21); //21

        res[6] = denseProd(smearing_all_x, grad_02); //02
        res[7] = denseProd(smearing_all_y, grad_12); //12
        res[8] = denseProd(smearing_all_z, pp2); //22

        /*res[0] = denseProd(smearing_all_x, h00); //00
        res[1] = denseProd(smearing_all_y, h10); //10
        res[2] = denseProd(smearing_all_z, h20); //20

        res[3] = denseProd(smearing_all_x, h01); //01
        res[4] = denseProd(smearing_all_y, h11); //11
        res[5] = denseProd(smearing_all_z, h21); //21

        res[6] = denseProd(smearing_all_x, h02); //02
        res[7] = denseProd(smearing_all_y, h12); //12
        res[8] = denseProd(smearing_all_z, h22); //22*/

        /*for (int i = 1; i < N_modes; i++) {
            float_type c = (static_cast<float_type>(i)) * 2.0 * M_PI * dx_base/c_z;
            helm_t lgf_helm(c);
            Conv_matrix helm_mat(conv_range, conv_range, dx_ib, lgf_helm);
        }*/
    }

    void creating_matrix(std::vector<force_type>& matrix_force_glob, float_type alpha) {
        boost::mpi::communicator world;

        point_force_type tmp(0.0);

        force_type tmp_f(domain_->ib().size(), tmp);

        int force_dim = tmp_f.size()*(u_type::nFields())/N_modes;

        matrix_force_glob.resize(0);

        std::vector<force_type> matrix_force = matrix_force_glob; 

        matrix_force.resize(0);

        if (world.rank() != 0) {
        for (int num = 0; num < force_dim;num++) {
            if (world.rank() == 1)
            {
                if (num % ((u_type::nFields())/N_modes) == 0)
                {
                    std::cout << "Constructing matrix for IB, alpha = " << alpha << " number solved " << num << " over "
                              << force_dim << std::endl;
                }
            }
            for (int i = 0; i < tmp_f.size(); i++)
            {
                for (int j = 0; j < tmp_f[0].size(); j++) { tmp_f[i][j] = 0.0; }
            }
            int ib_idx = num / ((u_type::nFields())/N_modes);
            int field_idx = num % ((u_type::nFields())/N_modes);
            int idx_complex = field_idx/2; //the number of components (zero for u, one for v, two for w)
            int realcomp = field_idx % 2;  //zero if real part but one if complex part

            if (domain_->ib().rank(ib_idx)==world.rank()) {
                for (int ModeN = 0; ModeN < N_modes; ModeN++) {
                    int field_idx_now = idx_complex*N_modes*2 + ModeN*2 + realcomp;
                    tmp_f[ib_idx][field_idx_now] = 1.0;
                }
                //tmp_f[ib_idx][field_idx] = 1.0;
            }
            /*for (int i = 0; i < tmp_f.size(); i++)
            {
                if (domain_->ib().rank(i)!=world.rank()) continue;
                tmp_f[i][num] = 1.0;
            }*/
            force_type Ap(domain_->ib().size(), tmp);
            for (int i = 0; i < Ap.size(); i++)
            {
                for (int j = 0; j < Ap[0].size(); j++) { Ap[i][j] = 0.0; }
            }
            this->template ET_H_S_E<face_aux2_type>(tmp_f, Ap, alpha);
            for (int i = 0; i < Ap.size(); i++)
            {
                if (domain_->ib().rank(i)!=world.rank()) {
                    for (int j = 0; j < Ap[0].size(); j++) { Ap[i][j] = 0.0; }
                }
            }
            matrix_force.emplace_back(Ap);
        }

        if (world.rank() == 1) {
            std::cout << "finished constructing" << std::endl;
            std::cout << "number of rows " << matrix_force.size() << std::endl;
        }
        }

        matrix_force_glob = matrix_force;

        for (int i = 0; i < matrix_force_glob.size(); i++) {
            boost::mpi::all_reduce(domain_->client_communicator(), &matrix_force[i][0], domain_->ib().size(), &matrix_force_glob[i][0], std::plus<point_force_type>());
        }
    }

    float_type test()
    {
        float_type alpha =  0.01;
        if (domain_->is_server())
            return 0;

        for (std::size_t i=0; i<ib_->size(); ++i)
            ib_->force(i,0)=1;

        //force_type uc(ib_->force());

        //this->template CG_solve<u_type>(uc, alpha);

        //force_type tmp(ib_->size(), (0.,0.,0.));
        //this->template ET_H_S_E<u_type>(ib_->force(), tmp, MASK_TYPE::IB2AMR, alpha);

        //ib_->communicator().compute_indices();
        //ib_->communicator().communicate(true, ib_->force());

        //if (comm_.rank()==1)
        //{
        //    printvec(ib_->force(), "forces");
        //    printvec(tmp, "u");
        //}

        domain::Operator::domainClean<face_aux_type>(domain_);
        domain::Operator::add_field_expression_complex_helmholtz<face_aux_type>(domain_, N_modes, simulation_->frame_vel(), 1.0);

        point_force_type tmp(0.0);

        force_type tmp2(ib_->size(), tmp);
        this->projection<face_aux_type>(tmp2);
        ib_->communicator().compute_indices();
        ib_->communicator().communicate(true, tmp2);


        

        if (comm_.rank()==1)
        {
            std::cout<< " Projection test" << std::endl;
            for (std::size_t i=0; i<ib_->size(); ++i)
                std::cout<<ib_->coordinate(i) << ", " << tmp2[i] << std::endl;
        }

        return 0;
    }

    template<class Field>
    void ib_solve(float_type alpha, float_type t, int stage_idx)
    {
        // right hand side
        point_force_type tmp(0.0);
        force_type uc(ib_->size(), tmp);

        domain_->client_communicator().barrier();
        //std::cout<<"projection" << std::endl;
        this->projection<Field>(uc);
        domain_->client_communicator().barrier();
        //std::cout<<"subtract_boundary_vel" << std::endl;
        this->subtract_boundary_vel(uc, t);

        domain_->client_communicator().barrier();
        domain::Operator::domainClean<face_aux2_type>(domain_);
        if (!usingDirectSolve) this->template CG_solve_mode<face_aux2_type>(uc, alpha);
        else {
            auto& f = ib_->force();
            this->DirectSolve((stage_idx-1), uc, f);
        }
    }

    template<class Field>
    void pressure_correction()
    {
        domain::Operator::domainClean<face_aux2_type>(domain_);
        domain::Operator::domainClean<cell_aux2_type>(domain_);
        this->smearing<face_aux2_type>(ib_->force());

        int l = domain_->tree()->depth()-1;
        domain::Operator::levelDivergence_helmholtz_complex<face_aux2_type, cell_aux2_type>(domain_, l, (1 + additional_modes), c_z, this->_compute_Modes);

        // apply L^-1
        psolver_.template apply_lgf_and_helm_ib<cell_aux2_type,cell_aux2_type>(N_modes, this->_compute_Modes, 1, MASK_TYPE::IB2AMR);

        domain::Operator::add<cell_aux2_type, Field>(domain_, -1.0);
    }

    template<class ForceType>
    void subtract_boundary_vel(ForceType& uc, float_type t)
    {
        int sep = 2*N_modes;
        auto& frame_vel = simulation_->frame_vel();
        for (std::size_t i=0; i<uc.size(); ++i) {
            uc[i][0]    -=frame_vel(0, t, ib_->coordinate(i));
            uc[i][sep]  -=frame_vel(1, t, ib_->coordinate(i));
            uc[i][2*sep]-=frame_vel(2, t, ib_->coordinate(i));
            
        }
    }

    template<class ForceType>
    ForceType boundaryVel(ForceType x)
    {
        point_force_type tmp(0.0);
        tmp[0] = 1.0;
        return tmp;
    }

    template<class Ftmp, class UcType>
    void CG_solve(UcType& uc, float_type alpha)
    {
        auto& f = ib_->force();

        
        for (int i = 0; i < N_modes; i++) {
            this->_compute_Modes[i] = true;
        }

        point_force_type tmp(0.0);
        force_type Ax(ib_->size(), tmp);
        force_type r (ib_->size(), tmp);
        force_type Ap(ib_->size(), tmp);

        if (domain_->is_server())
            return;

        // Ax
        this->template ET_H_S_E<Ftmp>(f, Ax, alpha);
        //printvec(Ax, "Ax");

        //  res = uc - Ax
        for (int i=0; i<ib_->size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                r[i]=0;
            else
                r[i]=uc[i]-Ax[i];
        }

        // p = res
        auto p = r;

        // rold = r'* r;
        float_type rsold = dot(r, r);

        for (int k=0; k<cg_max_itr_; k++)
        {
            // Ap = A(p)
            this->template ET_H_S_E<Ftmp>(p, Ap, alpha );
            // alpha = rsold / p'*Ap
            float_type pAp = dot(p,Ap);
            if (pAp == 0.0)
            {
                return;
            }

            float_type alpha = rsold / dot(p, Ap);
            // f = f + alpha * p;
            add(f, p, 1.0, alpha);
            // r = r - alpha*Ap
            add(r, Ap, 1.0, -alpha);
            // rsnew = r' * r
            float_type rsnew = dot(r, r);
            float_type f2 = dot(f,f);
            //auto ModeError = dot_Mode(r,r);
            if (comm_.rank()==1)
                std::cout<< "residue square = "<< rsnew/f2<<std::endl;;
            if (sqrt(rsnew/f2)<cg_threshold_)
                break;

            // p = r + (rsnew / rsold) * p;
            add(p, r, rsnew/rsold, 1.0);
            rsold = rsnew;
        }
    }


    template<class Ftmp, class UcType>
    void CG_solve_mode(UcType& uc, float_type alpha)
    {
        auto& f = ib_->force();

        
        for (int i = 0; i < N_modes; i++) {
            this->_compute_Modes[i] = true;
        }

        point_force_type tmp(0.0);
        force_type Ax(ib_->size(), tmp);
        force_type r (ib_->size(), tmp);
        force_type Ap(ib_->size(), tmp);

        if (domain_->is_server())
            return;

        // Ax
        this->template ET_H_S_E<Ftmp>(f, Ax, alpha);
        //printvec(Ax, "Ax");

        //  res = uc - Ax
        for (int i=0; i<ib_->size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                r[i]=0;
            else
                r[i]=uc[i]-Ax[i];
        }

        // p = res
        auto p = r;

        // rold = r'* r;
        ModesVector_type rsold = dot_mode(r, r);

        for (int k=0; k<cg_max_itr_; k++)
        {
            // Ap = A(p)
            this->template ET_H_S_E<Ftmp>(p, Ap, alpha );
            // alpha = rsold / p'*Ap
            ModesVector_type pAp = dot_mode(p,Ap);
            /*if (comm_.rank()==1){
            for (int i = 0; i < N_modes; i++) {
                std::cout << pAp[i] << " ";
            }
            std::cout << std::endl;
            }*/
            bool if_return = true;
            for (int i = 0; i < N_modes;i++) {
                if (pAp[i] == 0) {
                    this->_compute_Modes[i] = false;
                }
                else {
                    if_return = false;
                }
            }
            if (if_return)
            {
                return;
            }
            
            ModesVector_type alpha;
            for (int i = 0; i < rsold.size();i++) {
                float_type ratio = rsold[i]/pAp[i];
                if (this->_compute_Modes[i])
                {
                    alpha[i] = ratio;
                }
                else
                {
                    alpha[i] = 0;
                }
            }

            ModesVector_type n_alpha;
            for (int i = 0; i < rsold.size();i++) {
                float_type ratio = -rsold[i]/pAp[i];
                if (this->_compute_Modes[i])
                {
                    n_alpha[i] = ratio;
                }
                else
                {
                    n_alpha[i] = 0;
                }
                //n_alpha[i] = ratio;
            }
            /*if (comm_.rank()==1){
            for (int i = 0; i < N_modes; i++) {
                std::cout << alpha[i] << " ";
            }
            std::cout << std::endl;
            }*/
            // f = f + alpha * p;
            add_forcing_weighted_2(f, p, 1.0, alpha);
            // r = r - alpha*Ap
            add_forcing_weighted_2(r, Ap, 1.0, n_alpha);
            // rsnew = r' * r
            auto rsnew = dot_mode(r, r);
            auto f2 = dot_mode(f,f);

            //auto ModeError = dot_Mode(r,r);
            //auto Modef = dot_Mode(f,f);
            ModesVector_type ratio_i;
            float_type max_i = -1;

            float_type f2_sum = 0;
            for (int i = 0; i < f2.size();i++){
                f2_sum+=f2[i];
            }
            for (int i = 0; i < rsnew.size(); i++)
            {
                float_type ratio = rsnew[i] / f2_sum;
                if (this->_compute_Modes[i])
                {
                    ratio_i[i] = ratio;
                    if (ratio > max_i) max_i = ratio;
                    if (sqrt(ratio) < cg_threshold_) this->_compute_Modes[i] = false;
                }
                else
                {
                    ratio_i[i] = 0;
                }
            }

            /*if (comm_.rank()==1){
            for (int i = 0; i < N_modes; i++) {
                std::cout << ratio_i[i] << " ";
            }
            std::cout << std::endl;
            }*/
            /*if (comm_.rank()==1)
                std::cout<< "residue square = "<< rsnew/f2<<std::endl;;
            if (sqrt(rsnew/f2)<cg_threshold_)
                break;*/

            if (comm_.rank()==1)
                std::cout<< "residue square max = "<< max_i <<std::endl;;
            if (sqrt(max_i) < cg_threshold_)
            {
                for (int i = 0; i < N_modes; i++)
                {
                    this->_compute_Modes[i] = true;
                }
                break;
            }

            // p = r + (rsnew / rsold) * p;
            //auto add_ratio = divide_mode(rsnew, rsold);
            ModesVector_type add_ratio;
            for (int i = 0; i < rsold.size();i++) {
                float_type ratio = rsnew[i]/rsold[i];

                if (this->_compute_Modes[i]) {
                    add_ratio[i] = ratio;
                }
                else {
                    add_ratio[i] = 0;
                }
            }
            add_forcing_weighted(p, r, add_ratio, 1.0);
            rsold = rsnew;
        }
    }

    template<class Ftmp, class UcType>
    void BCG_solve(UcType& uc, float_type alpha_)
    {
        auto& f = ib_->force();

        
        for (int i = 0; i < N_modes; i++) {
            this->_compute_Modes[i] = true;
        }

        point_force_type tmp(0.0);
        force_type Ax(ib_->size(), tmp);
        force_type r(ib_->size(), tmp);
        //force_type v(ib_->size(), tmp); //sort of like a residue in QMR but not really
        //force_type v_prev(ib_->size(), tmp); //store previous v
        force_type Ap(ib_->size(), tmp);
        force_type Error(ib_->size(), tmp);

        if (domain_->is_server())
            return;

        // Ax
        this->template ET_H_S_E<Ftmp>(f, Ax, alpha_);
        //printvec(Ax, "Ax");

        //  res = uc - Ax
        for (int i=0; i<ib_->size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                r[i]=0;
            else
                r[i]=uc[i]-Ax[i];
        }

        //float_type w = dot(v,v);

        //std::complex<float_type> beta_sq = product(v,v);
        //std::complex<float_type> beta = std::sqrt(beta_sq);

        //std::complex<float_type> tao = beta*w;

        // p = res
        auto p = r;

        // rold = r'* r;
        std::complex<float_type> rsold = product(r, r);

        const std::complex<float_type> OneComp(1.0, 0.0);

        for (int k=0; k<cg_max_itr_; k++)
        {
            // Ap = A(p)
            this->template ET_H_S_E<Ftmp>(p, Ap, alpha_);
            // alpha = rsold / p'*Ap
            std::complex<float_type> pAp = product(p,Ap);
            if (std::abs(pAp) == 0.0 || std::abs(rsold) == 0.0)
            {
                return;
            }

            std::complex<float_type> alpha = rsold / pAp;
            // f = f + alpha * p;
            add_complex(f, p, OneComp, alpha);
            // r = r - alpha*Ap
            add_complex(r, Ap, OneComp, -alpha);
            // rsnew = r' * r
            std::complex<float_type> rsnew = product(r, r);
            float_type f2 = dot(f,f);
            float_type rs_mag = std::abs(rsnew);

            this->template ET_H_S_E<Ftmp>(f, Ax, alpha_);
            for (int i = 0; i < ib_->size(); ++i)
            {
                if (ib_->rank(i) != comm_.rank()) Error[i] = 0;
                else
                    Error[i] = uc[i] - Ax[i];
            }
            float_type errorMag = dot(Error, Error);
            if (comm_.rank()==1)
                std::cout<< "BCG residue square = "<< rs_mag/f2<< " Error is " << errorMag << std::endl;
            if (sqrt(rs_mag/f2)<cg_threshold_)
                break;

            // p = r + (rsnew / rsold) * p;
            add_complex(p, r, rsnew/rsold, OneComp);
            rsold = rsnew;
        }
    }

    template<class Ftmp, class UcType>
    void BCGstab_solve(UcType& uc, float_type alpha_)
    {
        auto& f = ib_->force();

        
        for (int i = 0; i < N_modes; i++) {
            this->_compute_Modes[i] = true;
        }

        point_force_type tmp(0.0);
        force_type Ax(ib_->size(), tmp);
        force_type As(ib_->size(), tmp);
        force_type r(ib_->size(), tmp);
        force_type v(ib_->size(), tmp);
        force_type p(ib_->size(), tmp);
        //force_type r_old(ib_->size(), tmp);
        force_type r_hat(ib_->size(), tmp);
        //force_type v(ib_->size(), tmp); //sort of like a residue in QMR but not really
        //force_type v_prev(ib_->size(), tmp); //store previous v
        //force_type Ap(ib_->size(), tmp);
        force_type Error(ib_->size(), tmp);

        if (domain_->is_server())
            return;

        // Ax
        this->template ET_H_S_E<Ftmp>(f, Ax, alpha_);
        //printvec(Ax, "Ax");

        //  res = uc - Ax
        for (int i=0; i<ib_->size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                r[i]=0;
            else
                r[i]=uc[i]-Ax[i];
        }

        r_hat = r;

        // rold = r'* r;
        float_type rsold = dot(r, r);
        float_type rho = 1;
        float_type rho_old = rho;
        float_type w = 1;
        float_type alpha = 1;

        for (int k=0; k<cg_max_itr_; k++)
        {
            rho = dot(r_hat, r);
            float_type beta = (rho/rho_old)*(alpha/w);
            add(p, v, 1.0, -w);
            add(p, r, beta, 1.0);
            // Ap = A(p)
            this->template ET_H_S_E<Ftmp>(p, v, alpha_);
            alpha = rho/dot(r_hat, v);

            add(f, p, 1.0, alpha);

            auto s = r;
            add(s, v, 1.0, -alpha);
            this->template ET_H_S_E<Ftmp>(s, As, alpha_);
            w = dot(As, s)/dot(As, As);
            add(f, s, 1.0, w);
            r = s;
            add(r, As, 1.0, -w);

            std::complex<float_type> rsnew = product(r, r);
            float_type f2 = dot(f,f);
            float_type rs_mag = std::abs(rsnew);

            /*this->template ET_H_S_E<Ftmp>(f, Ax, alpha_);
            for (int i = 0; i < ib_->size(); ++i)
            {
                if (ib_->rank(i) != comm_.rank()) Error[i] = 0;
                else
                    Error[i] = uc[i] - Ax[i];
            }
            float_type errorMag = dot(Error, Error);*/
            if (comm_.rank()==1)
                std::cout<< "BCGstab residue square = "<< rs_mag/f2/*<< " Error is " << errorMag*/ << std::endl;
            if (sqrt(rs_mag/f2)<cg_threshold_)
                break;

            // p = r + (rsnew / rsold) * p;
            rho_old = rho;
        }
    }

    template<class Ftmp, class UcType>
    void QMR_solve(UcType& uc, float_type alpha_)
    {
        auto& f = ib_->force();

        const std::complex<float_type> OneComp(1.0, 0.0);

        
        for (int i = 0; i < N_modes; i++) {
            this->_compute_Modes[i] = true;
        }

        point_force_type tmp(0.0);
        force_type Ax(ib_->size(), tmp);
        //force_type r(ib_->size(), tmp);
        force_type v(ib_->size(), tmp); //v_tilde sort of like a residue in QMR but not really
        force_type v_prev(ib_->size(), tmp); //store previous v_tilde
        force_type v_k(ib_->size(), tmp); //actual v
        force_type h(ib_->size(), tmp); //something to keep track of residue
        force_type v_k_old(ib_->size(), tmp);
        force_type Av(ib_->size(), tmp);

        if (domain_->is_server())
            return;

        // Ax
        this->template ET_H_S_E<Ftmp>(f, Ax, alpha_);
        //printvec(Ax, "Ax");

        //  res = uc - Ax
        for (int i=0; i<ib_->size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                v[i]=0;
            else
                v[i]=uc[i]-Ax[i];
        }


        std::complex<float_type> beta_sq = product(v,v);
        std::complex<float_type> beta = std::sqrt(beta_sq);


        auto tmp_v_k = v;
        scale_complex(tmp_v_k, OneComp/beta);
        float_type w = dot(tmp_v_k,tmp_v_k);
        w = std::sqrt(w);
        float_type w_old = w;
        float_type w_new = w;



        h = v;

        std::complex<float_type> tao_tilde = beta*w;
        std::complex<float_type> tao_tilde_old = beta*w;

        std::complex<float_type> s(0.0,0.0);

        std::complex<float_type> s_old = s;
        std::complex<float_type> s_oold = s; //s_{k-2}

        std::complex<float_type> c(1.0,0.0);
        std::complex<float_type> c_old = c;
        std::complex<float_type> c_oold = c;

        // p = res
        auto p = v_k;
        auto p_old = p;
        auto p_oold = p_old;

        std::complex<float_type> s_cum(1.0, 0.0);//value to find the accumulative product of s

        // rold = r'* r;
        

        

        for (int k=0; k<cg_max_itr_; k++)
        {
            if (std::abs(beta) == 0.0) {
                return;
            }
            v_k_old = v_k;
            v_k = v;
            scale_complex(v_k, OneComp/beta);
            //updating w, w_old
            w_old=w;
            w = w_new;
            //compute Av_k
            this->template ET_H_S_E<Ftmp>(v_k, Av, alpha_);
            std::complex<float_type> alpha = product(v_k, Av);
            //v_tilde = Av_k - alpha_k*v_k - beta_k*v_{k-1}
            v = Av;
            add_complex(v, v_k, OneComp, -alpha);
            add_complex(v, v_k_old, OneComp, -beta);
            std::complex<float_type> beta_new_sq = product(v, v);
            std::complex<float_type> beta_new    = std::sqrt(beta_new_sq);
            //updating w
            auto v_tmp = v; //use this v as the v for next step to compute w
            scale_complex(v_tmp, OneComp/beta_new);
            w_new = sqrt(dot(v_tmp, v_tmp));
            //compute theta, eta, xi, c, and s
            std::complex<float_type> theta_k = std::conj(s_oold)*beta*w_old;
            std::complex<float_type> eta     = c_old*c_oold*beta*w_old + std::conj(s_old)*alpha*w;
            std::complex<float_type> xi_tilde= c_old*alpha*w - s_old*c_oold*beta*w_old;
            float_type xi_mag = std::sqrt(std::abs(xi_tilde)*std::abs(xi_tilde)
                                        + w_new * w_new * std::abs(beta_new) * std::abs(beta_new));
            std::complex<float_type> xi(xi_mag, 0.0);
            if (std::abs(xi_tilde) != 0.0) {
                xi = xi_mag * xi_tilde / std::abs(xi_tilde);
            } 
            c = xi_tilde/xi;
            s = beta_new/xi*w_new;
            //updating c_oold c_old s_oold s_old
            c_oold = c_old;
            c_old = c;
            s_oold = s_old;
            s_old = s;

            auto p_tmp = v_k;
            add_complex(p_tmp, p_old, OneComp, -eta);
            add_complex(p_tmp, p_oold, OneComp, -theta_k);
            scale_complex(p_tmp, OneComp/xi);

            //updating p_old and p_oold
            p_oold = p_old;
            p_old = p;
            p = p_tmp;

            //tao_k = c_k tao_tilde_k
            std::complex<float_type> tao = c*tao_tilde;
            tao_tilde = -s*tao_tilde;
            //updating f
            add_complex(f, p, OneComp, tao);
            //updating beta and updating h and s_cum
            s_cum *= s;
            beta = beta_new;
            auto h_tmp = v;
            //get v_{k+1} = v_tilde/beta_new
            scale_complex(h_tmp, OneComp/beta_new);
            auto scaleVal = c*tao_tilde/(std::abs(s_cum)*std::abs(s_cum))*w_new;
            add_complex(h, h_tmp, OneComp, scaleVal);
            float_type f2 = dot(f,f);
            float_type rs_mag = (std::abs(s_cum)*std::abs(s_cum)) * (std::abs(s_cum)*std::abs(s_cum))*dot(h,h);

            if (comm_.rank()==1)
                std::cout<< "residue square = "<< rs_mag/f2<<std::endl;;
            if (sqrt(rs_mag/f2)<cg_threshold_) 
                break;
        }
    }


    template <class F, class S>
    void printvec(F& f, S message)
    {
        std::cout<<"-- "<< message << std::endl;
        for (int i=0; i<f.size(); ++i)
            std::cout<<f[i][0] << " " << f[i][2*N_modes] << " " << f[i][4*N_modes] << std::endl;
    }


    //template <class VecType>
    //void ET_S_E(VecType& fin, VecType& fout, int fmm_type = MASK_TYPE::IB2IB)
    //{

    //    force_type ftmp(ib_->size(), (0.,0.,0.));

    //    if (domain_->is_server())
    //        return;

    //    this->smearing<u_type>(fin);
    //    this->projection<u_type>(fout);
    //    this->template apply_Schur<u_type, u_type>(fmm_type);
    //    this->projection<u_type>(ftmp);

    //    add(fout, ftmp, 1, -1);
    //}

    template <class Field, class VecType>
    void ET_H_S_E(VecType& fin, VecType& fout, float_type alpha)
    {
        auto client = domain_->decomposition().client();

        domain::Operator::domainClean<Field>(domain_);
        domain::Operator::domainClean<face_aux_type>(domain_);

        this->smearing<Field>(fin);

        //this->template apply_Schur<Field, face_aux_type>(MASK_TYPE::IB2xIB);

        //domain::Operator::add<face_aux_type, Field>(domain_, -1.0);

        //if (std::fabs(alpha)>1e-4)
        //    psolver_.template apply_lgf_IF<Field, Field>(alpha, MASK_TYPE::xIB2IB);

        if (std::fabs(alpha)>1e-14)
            psolver_.template apply_helm_if_ib<Field, Field>(alpha, N_modes, c_z, this->_compute_Modes, 3, MASK_TYPE::IB2xIB);

        this->template apply_Schur<Field, face_aux_type>(MASK_TYPE::xIB2IB);

        domain::Operator::add<face_aux_type, Field>(domain_, -1.0);

        this->projection<Field>(fout);

    }


    template<class Source, class Target>
    void apply_Schur(int fmm_type)
    {

        // finest level only
        int l = domain_->tree()->depth()-1;

        // div
        domain::Operator::levelDivergence_helmholtz_complex<Source, cell_aux2_type>(domain_, l, (1 + additional_modes), c_z, this->_compute_Modes);

        // apply L^-1
        psolver_.template apply_lgf_and_helm_ib<cell_aux2_type, cell_aux2_type>(N_modes, this->_compute_Modes, 1, fmm_type);

        // apply Gradient
        const int l_max = (fmm_type != MASK_TYPE::STREAM) ?
                    domain_->tree()->depth() : domain_->tree()->base_level()+1;

        const int l_min = (fmm_type !=  MASK_TYPE::IB2xIB && fmm_type !=  MASK_TYPE::xIB2IB) ?
                    domain_->tree()->base_level() : domain_->tree()->depth()-1;

        for (int l = l_min; l < l_max; ++l)
            domain::Operator::levelGradient_helmholtz_complex<cell_aux2_type, Target>(domain_, l, (1 + additional_modes), c_z, this->_compute_Modes);
    }

    template<class U, class ForceType,
        typename std::enable_if<(U::mesh_type() == MeshObject::face), void>::type* = nullptr>
    void smearing(ForceType& f, bool cleaning=true)
    {
        int sep = 2*N_modes; //distance between different components in the vector

        ib_->communicator().compute_indices();
        ib_->communicator().communicate(true, f);

        //cleaning
        if (cleaning)
            domain::Operator::domainClean<U>(domain_);

        for (std::size_t i=0; i<ib_->size(); ++i)
        {
            std::size_t oct_i=0;
            for (auto it: ib_->influence_list(i))
            {
                if (!it->locally_owned()) continue;
                auto ib_coord = ib_->scaled_coordinate(i, it->refinement_level());

                domain::Operator::ib_smearing_helmholtz<U>
                    (ib_coord, f[i], ib_->influence_pts(i, oct_i), ib_->delta_func(), N_modes, this->_compute_Modes);
                oct_i+=1;
            }
        }

    }

    template<class U, class ForceType,
        typename std::enable_if<(U::mesh_type() == MeshObject::face), void>::type* = nullptr>
    void projection(ForceType& f)
    {
        int sep = 2*N_modes;

        if (domain_->is_server())
            return;

        // clean f
        for (std::size_t i=0; i<ib_->size(); ++i)
            f[i]=0.0;

        for (std::size_t i=0; i<ib_->size(); ++i)
        {
            std::size_t oct_i=0;

            for (auto it: ib_->influence_list(i))
            {
                if (!it->locally_owned()) continue;
                auto ib_coord = ib_->scaled_coordinate(i, it->refinement_level());

                domain::Operator::ib_projection_helmholtz<U>
                    (ib_coord, f[i], ib_->influence_pts(i, oct_i), ib_->delta_func(), N_modes, this->_compute_Modes);

                oct_i+=1;
            }
        }

        ib_->communicator().compute_indices();
        ib_->communicator().communicate(false, f);

    }

    template<class VecType>
    float_type dot(VecType& a, VecType& b)
    {
        float_type s = 0;
        for (std::size_t i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            for (std::size_t d=0; d<a[0].size(); ++d)
                s+=a[i][d]*b[i][d];
        }

        float_type s_global=0.0;
        boost::mpi::all_reduce(domain_->client_communicator(), s,
                s_global, std::plus<float_type>());
        return s_global;
    }
    /*product of v^Tv for complex vectors*/
    template<class VecType>
    std::complex<float_type> product(VecType& a, VecType& b)
    {
        
        float_type real = 0;
        float_type imag = 0;
        for (std::size_t i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            int Num = a[i].size()/2;
            if (a[i].size() % 2 != 0) throw std::runtime_error("Number of elements in vector is not even (in QMR solve product)");
            for (std::size_t d=0; d<Num; ++d) {
                real+= (a[i][d*2]*b[i][d*2] - a[i][d*2 + 1]*b[i][d*2 + 1]);
                imag+= (a[i][d*2]*b[i][d*2 + 1] + b[i][d*2]*a[i][d*2 + 1]);
            }
        }
        float_type real_global = 0.0;
        float_type imag_global = 0.0;
        
        boost::mpi::all_reduce(domain_->client_communicator(), real,
                real_global, std::plus<float_type>());
        boost::mpi::all_reduce(domain_->client_communicator(), imag,
                imag_global, std::plus<float_type>());
        std::complex<float_type> s_global(real_global,imag_global);
        return s_global;
    }

    template<class VecType>
    ModesVector_type dot_mode(VecType& a, VecType& b)
    {
        ModesVector_type s(0.0);
        for (std::size_t i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            for (std::size_t n = 0; n < N_modes; n++) {
                if (!this->_compute_Modes[n]) continue;
                s[n] += (a[i][2*n] * b[i][2*n] + 
                         a[i][2*n + 1] * b[i][2*n + 1] +
                         a[i][2*N_modes + 2*n] * b[i][2*N_modes + 2*n] + 
                         a[i][2*N_modes + 2*n + 1] * b[i][2*N_modes + 2*n + 1] + 
                         a[i][4*N_modes + 2*n] * b[i][4*N_modes + 2*n] + 
                         a[i][4*N_modes + 2*n + 1] * b[i][4*N_modes + 2*n + 1]);
            }
        }

        ModesVector_type s_global(0.0);
        boost::mpi::all_reduce(domain_->client_communicator(), s,
                s_global, std::plus<ModesVector_type>());
        return s_global;
    }

    template <class VecType>
    void add(VecType& a, VecType& b,
            float_type scale1=1.0, float_type scale2=1.0)
    {
        for (int i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            for (int d=0; d<a[0].size(); ++d)
                a[i][d] = a[i][d]*scale1 + b[i][d]*scale2;
        }
    }

    template <class VecType>
    void add_forcing(VecType& a, VecType& b,
            float_type scale1=1.0, float_type scale2=1.0)
    {
        for (int i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            for (std::size_t n = 0; n < N_modes; n++) {
                if (!_compute_Modes[n]) continue;
                for (int k = 0; k < 3; k++) {
                    a[i][2*n + 2*N_modes*k]   = a[i][2*n + 2*N_modes*k] *   scale1 + b[i][2*n + 2*N_modes*k] *   scale2;
                    a[i][2*n+1 + 2*N_modes*k] = a[i][2*n+1 + 2*N_modes*k] * scale1 + b[i][2*n+1 + 2*N_modes*k] * scale2;
                }

            }
        }
    }

    template <class VecType, class VecType2>
    void add_forcing_weighted(VecType& a, VecType& b,
            VecType2& scale1, float_type scale2=1.0)
    {
        for (int i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            for (std::size_t n = 0; n < N_modes; n++) {
                if (!_compute_Modes[n]) continue;
                for (int k = 0; k < 3; k++) {
                    a[i][2*n + 2*N_modes*k]   = a[i][2*n + 2*N_modes*k] *   scale1[n] + b[i][2*n + 2*N_modes*k] *   scale2;
                    a[i][2*n+1 + 2*N_modes*k] = a[i][2*n+1 + 2*N_modes*k] * scale1[n] + b[i][2*n+1 + 2*N_modes*k] * scale2;
                }

            }
        }
    }

    template <class VecType, class VecType2>
    void add_forcing_weighted_2(VecType& a, VecType& b,
            float_type scale1, VecType2& scale2)
    {
        for (int i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            for (std::size_t n = 0; n < N_modes; n++) {
                if (!_compute_Modes[n]) continue;
                for (int k = 0; k < 3; k++) {
                    a[i][2*n + 2*N_modes*k]   = a[i][2*n + 2*N_modes*k] *   scale1 + b[i][2*n + 2*N_modes*k] *   scale2[n];
                    a[i][2*n+1 + 2*N_modes*k] = a[i][2*n+1 + 2*N_modes*k] * scale1 + b[i][2*n+1 + 2*N_modes*k] * scale2[n];
                }

            }
        }
    }

    template <class VecType>
    void add_mode(VecType& a, VecType& b,
            float_type scale1=1.0, float_type scale2=1.0)
    {
        for (int i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            for (std::size_t n = 0; n < N_modes; n++) {
                if (!_compute_Modes[n]) continue;
                a[i][n]   = a[i][n] *  scale1 + b[i][n] * scale2;
            }
        }
    }


    template <class VecType>
    void add_complex(VecType& a, VecType& b,
            std::complex<float_type> scale1, std::complex<float_type> scale2)
    {
        for (int i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            int Num = a[i].size()/2;
            if (a[i].size() % 2 != 0) throw std::runtime_error("Number of elements in vector is not even (in QMR solve product)");
            VecType a_tmp = a;
            for (std::size_t d=0; d<Num; ++d) {
                float_type real = a[i][d*2];
                float_type imag = a[i][d*2+1];
                a[i][d*2]   = (real*scale1.real() - imag*scale1.imag()
                             + b[i][d*2]*scale2.real() - b[i][d*2 + 1]*scale2.imag());
                a[i][d*2+1] = (imag*scale1.real() + real*scale1.imag() 
                             + b[i][d*2 + 1]*scale2.real() + b[i][d*2]*scale2.imag());
            }
        }
    }


    template <class VecType>
    void scale_complex(VecType& a, std::complex<float_type> scale1)
    {
        for (int i=0; i<a.size(); ++i)
        {
            if (ib_->rank(i)!=comm_.rank())
                continue;

            int Num = a[i].size()/2;
            if (a[i].size() % 2 != 0) throw std::runtime_error("Number of elements in vector is not even (in QMR solve product)");
            VecType a_tmp = a;
            for (std::size_t d=0; d<Num; ++d) {
                float_type real = a[i][d*2];
                float_type imag = a[i][d*2+1];
                a[i][d*2]   = (real*scale1.real() - imag*scale1.imag());
                a[i][d*2+1] = (imag*scale1.real() + real*scale1.imag());
            }
        }
    }
    

  private:
    boost::mpi::communicator comm_;
    std::vector<bool> _compute_Modes;
    simulation_type* simulation_;
    domain_type*     domain_; ///< domain
    ib_t* ib_;
    poisson_solver_t psolver_;

    int support_ext_factor = 2;

    std::vector<force_type> matrix_ib; //matrix for IB terms
    std::vector<DirectIB<Setup>> DirectSolvers;
    bool usingDirectSolve = false;

    int additional_modes = 0;

    float_type c_z; //length in the homogeneous direction

    float_type cg_threshold_;
    int  cg_max_itr_;

    void add_matrix(const std::vector<std::map<key_2D, float_type>>& A, std::vector<std::map<key_2D, float_type>>& B, float_type a, float_type b) {
        //compute the matrix aA + bB, and store in B
        //requires size of A and B to be the same
        //this operation is specially designed for computing P - GL^(-1)DP where P is the smearing operator
        if (A.size() != B.size()) {
            std::cout << "size not equal in matrix add in linsys" << std::endl;
        }
        for (int i = 0; i < B.size(); i++) {
            std::map<key_2D, float_type> colB = B[i];
            std::map<key_2D, float_type> colA = A[i];
            for (const auto& [key, val] : colB) {
                auto it = colA.find(key);
                if (it == colA.end()) {
                    B[i][key] = val * b;
                }
                else {
                    B[i][key] = val * b + it->second * a;
                }
            }
        }
    }

    std::vector<std::map<key_2D, float_type>> scale_matrix_real(const std::vector<std::map<key_2D, float_type>>& A, float_type a) {
        //compute the matrix aA 
        std::vector<std::map<key_2D, float_type>> res;
        res.resize(A.size());
        for (int i = 0; i < A.size(); i++) {
            std::map<key_2D, float_type> colA = A[i];
            for (const auto& [key, val] : colA) {
                float_type val_s = a*val;
                res[i].emplace(key, val_s);
            }
        }
        return res;
    }

    void scale_matrix(std::vector<std::map<key_2D, float_type>>& A, std::vector<std::map<key_2D, float_type>>& B, float_type a, float_type b) {
        //compute the matrix (a + ib)(A + iB), and store back in A, B
        //requires size of A and B to be the same
        //this operation is specially designed for computing d/dz part of P - GL^(-1)DP where P is the smearing operator
        if (A.size() != B.size()) {
            std::cout << "size not equal in complex matrix scale in linsys" << std::endl;
        }
        auto A_tmp = A;
        //real part is aA - bB
        add_matrix(B, A,, -b, a);
        //imag part is bA + aB
        add_matrix(A_tmp, B, b, a);
    }

    std::vector<std::map<key_2D, float_type>> dx_matrix(const std::vector<std::map<key_2D, float_type>>& A, int comp, std::string op, float_type dx) {
        //compute the matrix derivative A from divergence/grad, and return
        //the comp arg means the direction on which the derivative is taken
        boost::mpi::communicator world;
        std::string div = "div";
        std::string grad = "grad";
        bool is_div = false;
        bool is_grad = false;
        if (op == div) {
            is_div = true;
            if (world.rank() == 1) std::cout << "computing divergence" << std::endl;
        }
        else if (op == grad) {
            is_grad = true;
            if (world.rank() == 1) std::cout << "compute gradient" << std::endl;
        }
        else {
            if (world.rank() == 1) std::cout << "op given is not available" << std::endl;
        }
        std::vector<std::map<key_2D, float_type>> res;
        res.resize(A.size());
        for (int i = 0; i < A.size(); i++) {
            std::map<key_2D, float_type> colA = A[i];
            for (const auto& [key, val] : colA) {
                
                int idx0 = std::get<0>(key);
                int idx1 = std::get<1>(key);
                if (is_div && comp == 0) {
                    idx0 += 1;
                }
                else if (is_div && comp == 1) {
                    idx1 += 1;
                }
                else if (is_grad && comp == 0) {
                    idx0 -= 1;
                }
                else if (is_grad && comp == 1) {
                    idx1 -= 1;
                }
                key_2D tmp(idx0, idx1);
                auto it = colA.find(tmp);
                if (it == colA.end()) {
                    //if (is_div) { res[i][key] = -val / dx; }
                    //else if (is_grad) { res[i][key] = val / dx; }
                    continue;
                }
                if (is_div) {
                    res[i][key] = (it->second - val)/dx;
                }
                else if (is_grad) {
                    res[i][key] = (val - it->second)/dx;
                }
            }
        }
        return res;
    }


    std::vector<std::map<key_2D, float_type>> smearing_matrix(bool local, int dir) {
        //compute the smearing matrix
        //if local is true, only compute the smearing matrix associated with local ib points
        //if local is false, compute the smearing matrix for all the ib points
        boost::mpi::communicator world;

        std::vector<std::map<key_2D, float_type>> res;

        const int l_max = domain_->tree()->depth() - 1;
        const int l_min = domain_->tree()->base_level();

        const int ref_l = l_max - l_min;

        auto ddf = ib_->delta_func();
        for (int i = 0; i < domain_->ib().size();i++) {
            if ((domain_->ib().rank(i) != world.rank()) && local) continue;

            std::map<key_2D, float_type> col;

            auto ib_coord = ib_->scaled_coordinate(i, ref_l);

            float_type ib_radius = domain_->ib().ddf_radius();

            int l_0 = std::floor((ib_coord[0] - ib_radius - 1));
            int r_0 = std::ceil((ib_coord[0] + ib_radius + 1));
            int l_1 = std::floor((ib_coord[1] - ib_radius - 1));
            int r_1 = std::ceil((ib_coord[1] + ib_radius + 1));

            for (int idx0 = l_0; idx0 < (r_0 + 1); idx0++) {
                for (int idx1 = l_1; idx1 < (r_1 + 1); idx1++) {
                    key_2D loc(idx0, idx1);
                    float_type x_loc = static_cast<float_type>(idx0) - ib_coord[0] + 0.5;
                    float_type y_loc = static_cast<float_type>(idx1) - ib_coord[1] + 0.5;
                    if (dir == 0) x_loc -= 0.5;
                    if (dir == 1) y_loc -= 0.5;

                    real_coordinate_type tmp_coord;
                    tmp_coord[0] = x_loc;
                    tmp_coord[1] = y_loc;

                    float_type val = ddf(tmp_coord);

                    if (std::abs(val) > 1e-12) col.emplace(loc, val);
                }
            }

            res.emplace_back(col);
        }

        return res;
        
    }


    std::vector<std::vector<float_type>> denseProd(const std::vector<std::map<key_2D, float_type>>& A, const std::vector<std::map<key_2D, float_type>>& B) {
        //compute a dense matrix product, this is designed for P^T H(I - GL^{-1}D)P
        //where P^T = A^T, and H(I - GL^{-1}D)P = B
        boost::mpi::communicator world;

        std::vector<std::vector<float_type>> res;

        res.resize(A.size());

        for (int i = 0; i < A.size(); i++) {
            res[i].resize(B.size());
        }

        
        for (int i = 0; i < A.size(); i++) {
            std::map<key_2D, float_type> row = A[i];
            for (int j = 0; j < B.size(); j++) {
                std::map<key_2D, float_type> col = B[j];
                float_type sum = 0;
                //search according to col since A is much sparser
                //so that finding element in row is much faster
                for (const auto& [key, val] : col) {
                    auto it = row.find(key);
                    if (it != row.end()) {
                        sum += val * (it->second);
                    }
                }
                res[i][j] = sum;
            }
        }
        return res;
    }

    class Conv_matrix {
        public:
        template<class Kernel>
        Conv_matrix(int _t_l, int _s_l, float_type dx, Kernel& kernel, simulation_type* simulation) 
        : _domain_(simulation->domain_.get())
        {
            //need to set alpha before calling constructor if constructing IF, and dx = 1.0
            if (_t_l <= 0 || _s_l <= 0) {
                std::cout << "range or source negative" << std::endl;
            }
            t_l = -_t_l;
            t_r = _t_l;
            s_l = -_s_l;
            s_r = _s_l;

            mat.resize(t_r);
            for (int i = 0; i < mat.size(); i++) {
                mat[i].resize(t_r);
            }

            const int l = _domain_->tree()->depth() - 1;

            kernel.change_level(l - _domain_->tree()->base_level());

            if (kernel.neighbor_only()) {
                kernel.derived().build_lt();
            }

            for (int i = 0; i < mat.size(); i++) {
                for (int j = 0; j < t_r; j++) {
                    float_type val = kernel.derived().get(coordinate_type({i, j}));
                    mat[i][j] = val*std::pow(dx, Dim);
                }
            }
        }


        std::vector<std::map<key_2D, float_type>> apply_mat(const std::vector<std::map<key_2D, float_type>>& B) {
            //compute the matrix res = conv(mat,B)
            std::vector<std::map<key_2D, float_type>> res;
            //res.resize(B.size());
            for (int j = 0; j < B.size(); j++) {
                std::map<key_2D, float_type> col = B[j]; //jth local column in B
                std::map<key_2D, float_type> col_res;
                key_2D tmp_coord = col.begin()->first;
                int min_0 = std::get<0>(tmp_coord);
                int max_0 = std::get<0>(tmp_coord);
                int min_1 = std::get<1>(tmp_coord);
                int max_1 = std::get<1>(tmp_coord);
                for (const auto& [key, val] : col) {
                    int idx0 = std::get<0>(key);
                    int idx1 = std::get<1>(key);

                    if (idx0 > max_0) {
                        max_0 = idx0;
                    }
                    if (idx0 < min_0) {
                        min_0 = idx0;
                    }
                    if (idx1 > max_1) {
                        max_1 = idx1;
                    }
                    if (idx1 < min_1) {
                        min_1 = idx1;
                    }
                }
                for (int i_x = (s_l + min_0); i_x <= (s_r + max_0); i_x++) {
                    for (int i_y = (s_l + min_1); i_y <= (s_r + max_1); i_y++)
                    {
                        float_type sum = 0.0;
                        for (const auto& [key, val] : col)
                        {
                            int id_0 = std::get<0>(key);
                            int id_1 = std::get<1>(key);
                            int num_i = std::abs((i_x - id_0));
                            int num_j = std::abs((i_y - id_1));
                            if (num_i >= t_r || num_j >= t_r) continue;
                            float_type tmp = mat[num_i][num_j] * val;
                            sum += tmp;
                        }
                        col_res.emplace(key_2D(i_x, i_y), sum);
                    }
                }
                res.emplace_back(col_res);
            }
            return res;
        }

        std::vector<std::vector<float_type>> mat;
        domain_type*     _domain_;
        int t_l; //left bound of target range
        int t_r; //right bound of target range

        int s_l; //left bound of source range
        int s_r; //right bound of source range
    };

    
};

} // namespace solver
} // namespace iblgf

#endif
