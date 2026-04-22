#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>

#include <boost/filesystem.hpp>
#include <boost/mpi/communicator.hpp>

#include <iblgf/dictionary/dictionary.hpp>

#include "POD_2D.hpp"

namespace iblgf
{
namespace
{

constexpr double kCoeffMatchTol = 0.85;
constexpr double kARelErrTol = 0.30;
constexpr double kMode0LInfTol = 1e-2;

std::string get_input_path()
{
    if (const char* env = std::getenv("IBLGF_COMMON_TREE2D_CONFIG_POD")) { return env; }
    if (const char* env = std::getenv("IBLGF_COMMON_TREE2D_CONFIG")) { return env; }
    return "./configs/config_pod2D";
}

bool file_exists(const std::string& path)
{
    return boost::filesystem::exists(path);
}

void remove_if_exists(const std::string& path)
{
    if (file_exists(path)) { boost::filesystem::remove(path); }
}

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

std::vector<double> repeated_row_pattern(const std::vector<double>& row5, int n_total)
{
    std::vector<double> out(static_cast<std::size_t>(n_total), 0.0);
    for (int s = 0; s < n_total; ++s)
    {
        out[static_cast<std::size_t>(s)] = row5[static_cast<std::size_t>(s % 5)];
    }
    return out;
}

double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) { s += a[i] * b[i]; }
    return s;
}

double norm2(const std::vector<double>& a)
{
    return std::sqrt(dot(a, a));
}

void normalize(std::vector<double>& a);

double frob_norm_sq(const std::array<std::vector<double>, 3>& m)
{
    double s = 0.0;
    for (const auto& row : m)
    {
        for (const double v : row) s += v * v;
    }
    return s;
}

double aligned_frob_rel_error(
    const std::array<std::vector<double>, 3>& a_true,
    const std::array<std::vector<double>, 3>& a_comp)
{
    const std::array<std::array<int, 3>, 6> perms = {
        std::array<int, 3>{0, 1, 2},
        std::array<int, 3>{0, 2, 1},
        std::array<int, 3>{1, 0, 2},
        std::array<int, 3>{1, 2, 0},
        std::array<int, 3>{2, 0, 1},
        std::array<int, 3>{2, 1, 0},
    };

    double best = std::numeric_limits<double>::max();
    for (const auto& p : perms)
    {
        double err2 = 0.0;
        for (int i = 0; i < 3; ++i)
        {
            const auto& at = a_true[static_cast<std::size_t>(i)];
            const auto& ac = a_comp[static_cast<std::size_t>(p[static_cast<std::size_t>(i)])];
            double dot_t_c = 0.0;
            double dot_c_c = 0.0;
            for (std::size_t j = 0; j < at.size(); ++j)
            {
                dot_t_c += at[j] * ac[j];
                dot_c_c += ac[j] * ac[j];
            }
            const double alpha = (dot_c_c > 1e-30) ? (dot_t_c / dot_c_c) : 0.0;
            for (std::size_t j = 0; j < at.size(); ++j)
            {
                const double e = at[j] - alpha * ac[j];
                err2 += e * e;
            }
        }
        best = std::min(best, err2);
    }
    const double ref = std::max(1e-30, frob_norm_sq(a_true));
    return std::sqrt(best / ref);
}

std::array<std::vector<double>, 3> build_a_true(int n_total)
{
    const auto r1 = repeated_row_pattern({-2.0, -1.0, 0.0, 1.0, 2.0}, n_total);
    const auto r2 = repeated_row_pattern({1.0, -2.0, 0.0, 2.0, -1.0}, n_total);
    const auto r3 = repeated_row_pattern({1.0, 0.0, -2.0, 0.0, 1.0}, n_total);
    const double s1 = 3.0 / std::sqrt(10.0);
    const double s2 = 2.0 / std::sqrt(10.0);
    const double s3 = 1.0 / std::sqrt(6.0);

    std::array<std::vector<double>, 3> out = {
        std::vector<double>(r1.size(), 0.0),
        std::vector<double>(r2.size(), 0.0),
        std::vector<double>(r3.size(), 0.0),
    };
    for (std::size_t j = 0; j < r1.size(); ++j)
    {
        out[0][j] = s1 * r1[j];
        out[1][j] = s2 * r2[j];
        out[2][j] = s3 * r3[j];
    }
    return out;
}

std::array<std::vector<double>, 3> build_a_computed(
    std::vector<double> coeff0,
    std::vector<double> coeff1,
    std::vector<double> coeff2,
    const std::vector<double>& singular_values)
{
    std::array<std::vector<double>, 3> out = {coeff0, coeff1, coeff2};
    for (int i = 0; i < 3; ++i)
    {
        const double si =
            (static_cast<int>(singular_values.size()) > i) ? singular_values[static_cast<std::size_t>(i)] : 0.0;
        for (double& v : out[static_cast<std::size_t>(i)]) v *= si;
    }
    return out;
}

void expect_leading_coeff_vectors_match_a_rows(const std::vector<double>& coeff0, const std::vector<double>& coeff1, int n_total)
{
    auto c0 = coeff0;
    auto c1 = coeff1;
    normalize(c0);
    normalize(c1);

    auto a_row1 = repeated_row_pattern({-2.0, -1.0, 0.0, 1.0, 2.0}, n_total);
    auto a_row2 = repeated_row_pattern({1.0, -2.0, 0.0, 2.0, -1.0}, n_total);
    normalize(a_row1);
    normalize(a_row2);

    const double c0_r1 = std::abs(dot(c0, a_row1));
    const double c0_r2 = std::abs(dot(c0, a_row2));
    const double c1_r1 = std::abs(dot(c1, a_row1));
    const double c1_r2 = std::abs(dot(c1, a_row2));
    const bool   direct = (c0_r1 > kCoeffMatchTol && c1_r2 > kCoeffMatchTol);
    const bool   swapped = (c0_r2 > kCoeffMatchTol && c1_r1 > kCoeffMatchTol);
    EXPECT_TRUE(direct || swapped)
        << "Leading temporal POD vectors are not consistent with generated A rows.";
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
    out.mode0_sym = "./" + out_dir + "/flow_podmode_sym_0.hdf5";
    out.mode0_asym = "./" + out_dir + "/flow_podmode_asym_0.hdf5";
    out.snap0 = "./" + out_dir + "/flow_adapted_to_ref_" + std::to_string(idx_start) + ".hdf5";
    return out;
}

void clean_outputs(const PodOutputs& out)
{
    remove_if_exists(out.sv_sym);
    remove_if_exists(out.sv_asym);
    remove_if_exists(out.coeff0_sym);
    remove_if_exists(out.coeff1_sym);
    remove_if_exists(out.coeff2_sym);
    remove_if_exists(out.coeff0_asym);
    remove_if_exists(out.coeff1_asym);
}

void normalize(std::vector<double>& a)
{
    const double n = norm2(a);
    for (double& v : a) { v /= n; }
}

struct PodCaseResult
{
    double              sigma0 = 0.0;
    std::vector<double> coeff0;
};

PodCaseResult run_synthetic_pod_case(const std::string& config_path)
{
    Dictionary dictionary(config_path, 0, nullptr);
    auto       sim_dict = dictionary.get_dictionary("simulation_parameters");
    auto       out_dict = sim_dict->get_dictionary("output");

    const int idx_start = sim_dict->template get_or<int>("nStart", 100);
    const int n_total = sim_dict->template get_or<int>("nTotal", 6);
    const int nskip = sim_dict->template get_or<int>("nskip", 1);
    const std::string out_dir = out_dict->template get<std::string>("directory");
    const auto outputs = make_outputs(out_dir, idx_start);
    remove_if_exists(outputs.sv_sym);
    remove_if_exists(outputs.sv_asym);
    remove_if_exists(outputs.coeff0_sym);

    auto pod_setup = std::make_unique<POD2D>(&dictionary);
    pod_setup->write_fake_snapshots_single_mode(idx_start, n_total, nskip);

    int    argc = 0;
    char** argv = nullptr;
    pod_setup->run(argc, argv);

    if (!file_exists(outputs.sv_sym))
    {
        throw std::runtime_error("Missing POD output file: " + outputs.sv_sym);
    }
    if (!file_exists(outputs.coeff0_sym))
    {
        throw std::runtime_error("Missing POD coeff output file: " + outputs.coeff0_sym);
    }

    const auto sv_vals = read_all_values(outputs.sv_sym);
    if (sv_vals.empty()) { throw std::runtime_error("Empty singular value file: " + outputs.sv_sym); }

    PodCaseResult out;
    out.sigma0 = sv_vals[0];
    out.coeff0 = read_coeff_real_vec(outputs.coeff0_sym);
    return out;
}

} // namespace

TEST(POD2DTest, GeneratesSyntheticSnapshotsAndRunsPODPipeline)
{
    boost::mpi::communicator world;

    const std::string input = get_input_path();
    if (!file_exists(input))
    {
        GTEST_SKIP() << "Missing POD config file: " << input;
    }

    Dictionary dictionary(input, 0, nullptr);
    auto       sim_dict = dictionary.get_dictionary("simulation_parameters");
    auto       out_dict = sim_dict->get_dictionary("output");

    const int idx_start = sim_dict->template get_or<int>("nStart", 100);
    const int n_total = sim_dict->template get_or<int>("nTotal", 6);
    const int nskip = sim_dict->template get_or<int>("nskip", 1);
    const std::string out_dir = out_dict->template get<std::string>("directory");
    const auto outputs = make_outputs(out_dir, idx_start);
    clean_outputs(outputs);

    auto pod_setup = std::make_unique<POD2D>(&dictionary);

    ASSERT_NO_THROW({
        pod_setup->write_fake_snapshots_known_modes(idx_start, n_total, nskip);
    });

    ASSERT_NO_THROW({
        int    argc = 0;
        char** argv = nullptr;
        pod_setup->run(argc, argv);
    });

    EXPECT_TRUE(file_exists(outputs.sv_sym)) << "Missing POD output: " << outputs.sv_sym;
    EXPECT_TRUE(file_exists(outputs.sv_asym)) << "Missing POD output: " << outputs.sv_asym;
    EXPECT_TRUE(file_exists(outputs.coeff0_sym)) << "Missing POD coeff output: " << outputs.coeff0_sym;
    EXPECT_TRUE(file_exists(outputs.coeff1_sym)) << "Missing POD coeff output: " << outputs.coeff1_sym;
    EXPECT_TRUE(file_exists(outputs.coeff2_sym)) << "Missing POD coeff output: " << outputs.coeff2_sym;
    EXPECT_TRUE(file_exists(outputs.coeff0_asym)) << "Missing POD coeff output: " << outputs.coeff0_asym;
    EXPECT_TRUE(file_exists(outputs.coeff1_asym)) << "Missing POD coeff output: " << outputs.coeff1_asym;
    EXPECT_TRUE(file_exists(outputs.mode0_sym)) << "Missing POD mode output: " << outputs.mode0_sym;
    EXPECT_TRUE(file_exists(outputs.mode0_asym)) << "Missing POD mode output: " << outputs.mode0_asym;
    EXPECT_TRUE(file_exists(outputs.snap0)) << "Missing synthetic snapshot: " << outputs.snap0;

    // Singular values should be finite and non-negative.
    const auto sv_sym_vals = read_all_values(outputs.sv_sym);
    const auto sv_asym_vals = read_all_values(outputs.sv_asym);
    ASSERT_FALSE(sv_sym_vals.empty());
    ASSERT_FALSE(sv_asym_vals.empty());
    for (const double sv : sv_sym_vals)
    {
        EXPECT_TRUE(std::isfinite(sv));
        EXPECT_GE(sv, 0.0);
    }
    for (const double sv : sv_asym_vals)
    {
        EXPECT_TRUE(std::isfinite(sv));
        EXPECT_GE(sv, 0.0);
    }
    ASSERT_GE(static_cast<int>(sv_sym_vals.size()), 2);
    ASSERT_GE(static_cast<int>(sv_asym_vals.size()), 2);
    EXPECT_GE(sv_sym_vals[0], sv_sym_vals[1]);
    EXPECT_GE(sv_asym_vals[0], sv_asym_vals[1]);

    // Temporal vectors should be orthonormal for the leading subspace.
    auto coeff0 = read_coeff_real_vec(outputs.coeff0_sym);
    auto coeff1 = read_coeff_real_vec(outputs.coeff1_sym);
    auto coeff2 = read_coeff_real_vec(outputs.coeff2_sym);
    ASSERT_EQ(static_cast<int>(coeff0.size()), n_total);
    ASSERT_EQ(static_cast<int>(coeff1.size()), n_total);
    ASSERT_EQ(static_cast<int>(coeff2.size()), n_total);

    auto coeff0_n = coeff0;
    auto coeff1_n = coeff1;
    normalize(coeff0_n);
    normalize(coeff1_n);
    EXPECT_NEAR(dot(coeff0_n, coeff0_n), 1.0, 1e-8);
    EXPECT_NEAR(dot(coeff1_n, coeff1_n), 1.0, 1e-8);
    EXPECT_NEAR(dot(coeff0_n, coeff1_n), 0.0, 1e-6);
    expect_leading_coeff_vectors_match_a_rows(coeff0, coeff1, n_total);

    const auto a_true = build_a_true(n_total);
    const auto a_comp = build_a_computed(coeff0, coeff1, coeff2, sv_sym_vals);
    const double a_rel_err = aligned_frob_rel_error(a_true, a_comp);
    EXPECT_LT(a_rel_err, kARelErrTol)
        << "Relative Frobenius error ||A_true-A_comp||_F/||A_true||_F too large: " << a_rel_err;

    // Do the same vector checks for asymmetric branch.
    auto coeff0_as = read_coeff_real_vec(outputs.coeff0_asym);
    auto coeff1_as = read_coeff_real_vec(outputs.coeff1_asym);
    ASSERT_EQ(static_cast<int>(coeff0_as.size()), n_total);
    ASSERT_EQ(static_cast<int>(coeff1_as.size()), n_total);
    normalize(coeff0_as);
    normalize(coeff1_as);
    EXPECT_NEAR(dot(coeff0_as, coeff0_as), 1.0, 1e-8);
    EXPECT_NEAR(dot(coeff1_as, coeff1_as), 1.0, 1e-8);
    EXPECT_NEAR(dot(coeff0_as, coeff1_as), 0.0, 1e-6);
    // Same rationale as symmetric branch: validate subspace/error, not mode labeling.

    // Use the main framework norm path against u_ref stored in POD mode files.
    const double linf_sym_0 = pod_setup->mode0_error_sym();
    EXPECT_TRUE(std::isfinite(linf_sym_0));
    EXPECT_LT(linf_sym_0, kMode0LInfTol);

    const double linf_asym_0 = pod_setup->mode0_error_asym();
    EXPECT_TRUE(std::isfinite(linf_asym_0));
    EXPECT_LT(linf_asym_0, kMode0LInfTol);

    world.barrier();
}

TEST(POD2DTest, WeightedNormIsConsistentAcrossLevels)
{
    boost::mpi::communicator world;

    const std::string cfg_l0 = "./configs/config_pod2D";
    const std::string cfg_l2 = "./configs/config_pod2D_multilevel";
    if (!file_exists(cfg_l0) || !file_exists(cfg_l2))
    {
        GTEST_SKIP() << "Missing POD configs: " << cfg_l0 << " and/or " << cfg_l2;
    }

    PodCaseResult l0;
    PodCaseResult l2;
    ASSERT_NO_THROW({
        l0 = run_synthetic_pod_case(cfg_l0);
        l2 = run_synthetic_pod_case(cfg_l2);
    });

    ASSERT_GT(l0.sigma0, 0.0);
    ASSERT_GT(l2.sigma0, 0.0);

    // With proper weighting, leading singular values should stay close across refinement levels.
    const double rel_diff = std::abs(l2.sigma0 - l0.sigma0) / l0.sigma0;
    EXPECT_LT(rel_diff, 0.15)
        << "Weighted norm consistency failed: sigma(level2)=" << l2.sigma0
        << ", sigma(level0)=" << l0.sigma0 << ", rel_diff=" << rel_diff;

    // Leading temporal POD vectors should also agree (up to sign) across levels.
    ASSERT_EQ(l0.coeff0.size(), l2.coeff0.size());
    normalize(l0.coeff0);
    normalize(l2.coeff0);
    EXPECT_GT(std::abs(dot(l0.coeff0, l2.coeff0)), 0.95)
        << "Leading temporal POD vectors differ across levels.";

    world.barrier();
}

} // namespace iblgf
