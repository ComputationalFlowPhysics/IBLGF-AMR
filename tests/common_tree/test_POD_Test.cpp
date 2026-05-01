#include <gtest/gtest.h>

#include "POD_2D.hpp"
#include <iblgf/dictionary/dictionary.hpp>

using namespace iblgf;

std::vector<double> read_all_values(const std::string& path)
{
    std::ifstream in(path);
    std::vector<double> vals;
    double              value = 0.0;
    while (in >> value) { vals.push_back(value); }
    return vals;
}

std::vector<double> read_coeff_real_vec(const std::string& path)
{
    std::ifstream       in(path);
    std::vector<double> real_vals;
    double              vr = 0.0;
    double              vi = 0.0;
    while (in >> vr >> vi) { real_vals.push_back(vr); }
    return real_vals;
}
struct PodOutputs
{
    std::string sv_sym;
    std::string sv_asym;
    std::string coeff0_sym;
    std::string coeff1_sym;
    std::string coeff2_sym;
    std::string coeff0_asym;
    std::string coeff1_asym;
    std::string coeff2_asym;
    std::string mode0_sym;
    std::string mode0_asym;
    std::string snap0;
};

PodOutputs make_outputs(const std::string& out_dir, int idx_start)
{
    PodOutputs out;
    out.sv_sym = "./" + out_dir + "/singular_values_sym_r.txt";
    out.sv_asym = "./" + out_dir + "/singular_values_asym_r.txt";
    out.coeff0_sym = "./" + out_dir + "/coeff_real_sym_0000.txt";
    out.coeff1_sym = "./" + out_dir + "/coeff_real_sym_0001.txt";
    out.coeff2_sym = "./" + out_dir + "/coeff_real_sym_0002.txt";
    out.coeff0_asym = "./" + out_dir + "/coeff_real_asym_0000.txt";
    out.coeff1_asym = "./" + out_dir + "/coeff_real_asym_0001.txt";
    out.coeff2_asym = "./" + out_dir + "/coeff_real_asym_0002.txt";
    out.mode0_sym = "./" + out_dir + "/flow_podmode_sym_0.hdf5";
    out.mode0_asym = "./" + out_dir + "/flow_podmode_asym_0.hdf5";
    out.snap0 = "./" + out_dir + "/flow_adapted_to_ref_" + std::to_string(idx_start) + ".hdf5";
    return out;
}

template<std::size_t N>
void expect_mode_coeffs_up_to_sign(
    const std::vector<double>& coeffs,
    const std::array<float_type, N>& ref,
    float_type norm,
    double tol,
    double zero_tol = 1e-2)
{
    ASSERT_EQ(coeffs.size(), ref.size());
    double dot = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) { dot += coeffs[i] * (ref[i] / norm); }
    const double sgn = (dot >= 0.0) ? 1.0 : -1.0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        const double expected = sgn * (ref[i] / norm);
        if (std::abs(expected) < 1e-12)
        {
            EXPECT_LE(std::abs(coeffs[i]), zero_tol)
                << "Coefficient " << i << " expected near zero.";
        }
        else
        {
            EXPECT_NEAR(coeffs[i], expected, tol)
                << "Coefficient " << i << " does not match expected value up to sign.";
        }
    }
}

TEST(POD3DTest, SyntheticCaseKnownModes)
{
    boost::mpi::communicator world;
    if (world.size() < 2)
    {
        GTEST_SKIP() << "common_tree_overlap_Test requires at least 2 MPI ranks; got "
                     << world.size() << " ranks.";
    }

    Dictionary dictionary("./configs/config_pod3D", 0, nullptr);
    auto pod_setup = std::make_unique<POD2D>(&dictionary);
    auto       sim_dict = dictionary.get_dictionary("simulation_parameters");
    const int idx_start = sim_dict->template get_or<int>("nStart", 100);
    const int n_total = sim_dict->template get_or<int>("nTotal", 6);
    const int nskip = sim_dict->template get_or<int>("nskip", 1);
    const PetscErrorCode ierr_init = SlepcInitialize(nullptr, nullptr, (char*)0, NULL);
    ASSERT_EQ(ierr_init, PETSC_SUCCESS);
    pod_setup->write_fake_snapshots_known_modes(idx_start, n_total, nskip);

    pod_setup->run(0, nullptr);
    const PetscErrorCode ierr_fini = SlepcFinalize();
    ASSERT_EQ(ierr_fini, PETSC_SUCCESS);



    const std::string out_dir = sim_dict->get_dictionary("output")->template get<std::string>("directory");

    // print pod coefficients for debugging
    const auto outputs = make_outputs(out_dir, idx_start);

    auto coeff0 = read_coeff_real_vec(outputs.coeff0_sym);
    auto coeff1 = read_coeff_real_vec(outputs.coeff1_sym);
    auto coeff2 = read_coeff_real_vec(outputs.coeff2_sym);
    if(world.rank() == 0)
    {
        std::cout << "POD coeffs for symmetric branch:\n";
        std::cout << "Mode 0 coeffs: ";
        for (const auto& c : coeff0) { std::cout << c << " "; }
        std::cout << "\nMode 1 coeffs: ";
        for (const auto& c : coeff1) { std::cout << c << " "; }
        std::cout << "\nMode 2 coeffs: ";
        for (const auto& c : coeff2) { std::cout << c << " "; }
        std::cout << std::endl;
    }
    // expected values
    const std::array<float_type, 5> r1 = {-2.0, -1.0, 0.0, 1.0, 2.0};
        const std::array<float_type, 5> r2 = {1.0, -2.0, 0.0, 2.0, -1.0};
        const std::array<float_type, 5> r3 = {1.0, 0.0, -2.0, 0.0, 1.0};
        const float_type n1 = std::sqrt(static_cast<float_type>(10.0));
        const float_type n2 = std::sqrt(static_cast<float_type>(10.0));
        const float_type n3 = std::sqrt(static_cast<float_type>(6.0));
    if(world.rank() == 0)
    {
        std::cout << "Expected mode 0 coeffs (normalized): ";
        for (const auto& c : r1) { std::cout << c/n1 << " "; }
        std::cout << "\nExpected mode 1 coeffs (normalized): ";
        for (const auto& c : r2) { std::cout << c/n2 << " "; }
        std::cout << "\nExpected mode 2 coeffs (normalized): ";
        for (const auto& c : r3) { std::cout << c/n3 << " "; }
        std::cout << std::endl;
    }
    //make sure they are close
    expect_mode_coeffs_up_to_sign(coeff0, r1, n1, 1e-2);
    expect_mode_coeffs_up_to_sign(coeff1, r2, n2, 1e-2);
    expect_mode_coeffs_up_to_sign(coeff2, r3, n3, 1e-2);

    coeff0=read_coeff_real_vec(outputs.coeff0_asym);
    coeff1=read_coeff_real_vec(outputs.coeff1_asym);
    coeff2=read_coeff_real_vec(outputs.coeff2_asym);
    expect_mode_coeffs_up_to_sign(coeff0, r1, n1, 1e-2);
    expect_mode_coeffs_up_to_sign(coeff1, r2, n2, 1e-2);
    expect_mode_coeffs_up_to_sign(coeff2, r3, n3, 1e-2);

    auto singular_values_sym = read_all_values(outputs.sv_sym);
    auto singular_values_asym = read_all_values(outputs.sv_asym);
    if(world.rank() == 0)
    {
        std::cout << "Singular values for symmetric branch:\n";
        for (const auto& sv : singular_values_sym) { std::cout << sv << " "; }
        std::cout << "\nSingular values for asymmetric branch:\n";
        for (const auto& sv : singular_values_asym) { std::cout << sv << " "; }
        std::cout << std::endl;
    }
            const float_type s1 = static_cast<float_type>(3.0);
        const float_type s2 = static_cast<float_type>(2.0);
        const float_type s3 = static_cast<float_type>(1.0);
    // singular calues should be si*norm(phi_i)
    // assuming norm(phi_i) is constant then computed[i]/s_i should be approximately constant across modes
    EXPECT_NEAR(singular_values_sym[0]/s1, singular_values_sym[1]/s2, 1e-1);
    EXPECT_NEAR(singular_values_sym[1]/s2, singular_values_sym[2]/s3, 1e-1);
    EXPECT_NEAR(singular_values_sym[0]/s1, singular_values_sym[2]/s3, 1e-1);

    //assuming norm(phi_i) is constant across modes, the singular values should be in the ratio of s_i
    EXPECT_NEAR(singular_values_sym[0]/singular_values_sym[1], s1/s2, 1e-1);
    EXPECT_NEAR(singular_values_sym[1]/singular_values_sym[2], s2/s3, 1e-1);
    EXPECT_NEAR(singular_values_sym[0]/singular_values_sym[2], s1/s3, 1e-1);
    //assuming norm(phi_i) is 1, the singular values should be approximately equal to s_i
    EXPECT_NEAR(singular_values_sym[0],s1, 1e-1);
    EXPECT_NEAR(singular_values_sym[1],s2, 1e-1);
    EXPECT_NEAR(singular_values_sym[2],s3, 1e-1);

    EXPECT_EQ(1, 1) << "This test is designed for single-rank execution.";
    
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI before any tests run
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;  // optional, can use in main if needed

    return RUN_ALL_TESTS(); // now MPI is already initialized
}
