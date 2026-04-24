#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <set>
#include <sstream>
#include <string>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/environment.hpp>

#include "common_tree.hpp"
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/solver/modal_analysis/merge_trees.hpp>

namespace iblgf
{
namespace
{

template<class T>
std::set<T> set_diff(const std::set<T>& a, const std::set<T>& b)
{
    std::set<T> out;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
        std::inserter(out, out.begin()));
    return out;
}

template<class T>
std::string first_n(const std::set<T>& s, std::size_t n = 12)
{
    std::ostringstream oss;
    oss << "{ ";
    std::size_t c = 0;
    for (const auto& v : s)
    {
        if (c++ >= n) break;
        if (c > 1) oss << ", ";
        oss << v;
    }
    if (s.size() > n) oss << ", ...";
    oss << " }";
    return oss.str();
}

} // namespace

TEST(CommonTree3DUnitTest, AdaptToRefInterpolatesLinearFieldCorrectly)
{
    boost::mpi::communicator world;
    if (world.size() < 2)
    {
        GTEST_SKIP() << "adapt_to_ref overlap 3D test requires at least 2 MPI ranks; got "
                     << world.size() << " ranks.";
    }

    Dictionary dict_ref("./configs/adapt_to_ref_driver.cfg", 0, nullptr);
    {
        Dictionary dict1("./configs/common_tree_1.cfg", 0, nullptr);
        Dictionary dict2("./configs/common_tree_2.cfg", 0, nullptr);
        Dictionary dict3("./configs/common_tree_3.cfg", 0, nullptr);

        auto domain1 = std::make_unique<CommonTree>(&dict1);
        auto domain2 = std::make_unique<CommonTree>(&dict2);
        auto domain3 = std::make_unique<CommonTree>(&dict3);

        domain1->run(201, false);
        domain2->run(202, false);
        domain3->run(203, false);
    }

    const int idx_start =
        dict_ref.get_dictionary("simulation_parameters")->get_or<int>("nStart", 100);
    const int n_total =
        dict_ref.get_dictionary("simulation_parameters")->get_or<int>("nTotal", 100);
    const int nskip =
        dict_ref.get_dictionary("simulation_parameters")->get_or<int>("nskip", 100);

    auto merger = MergeTrees<CommonTree>(&dict_ref);
    auto domain_ref = std::make_unique<CommonTree>(&dict_ref);
    domain_ref->run(299, false);

    const auto ref_keys_global = MergeTrees<CommonTree>::globalize_key_set(
        world, MergeTrees<CommonTree>::collect_physical_leaf_key_ids(*domain_ref));
    for (int i = 0; i < n_total; ++i)
    {
        auto domain_i = merger.adapt_to_ref(idx_start + i * nskip);
        const auto final_keys_global = MergeTrees<CommonTree>::globalize_key_set(
            world, MergeTrees<CommonTree>::collect_physical_leaf_key_ids(*domain_i));
        EXPECT_FALSE(final_keys_global.empty()) << "Adapted tree must contain leaf keys.";

        const auto missing = set_diff(ref_keys_global, final_keys_global);
        const auto extra = set_diff(final_keys_global, ref_keys_global);
        EXPECT_TRUE(missing.empty() && extra.empty())
            << "Adapted tree keys must match common-reference keys.\n"
            << "missing.size() = " << missing.size()
            << ", extra.size() = " << extra.size() << "\n"
            << "missing(first) = " << first_n(missing) << "\n"
            << "extra(first) = " << first_n(extra);

        const double local_max_err = MergeTrees<CommonTree>::max_linear_u_error(*domain_i);
        double global_max_err = 0.0;
        boost::mpi::all_reduce(
            world, local_max_err, global_max_err, boost::mpi::maximum<double>());

        domain_i->run(100 + idx_start + i * nskip, false);

        EXPECT_LT(global_max_err, 1e-10)
            << "Interpolated u field must preserve linear profile.";
    }
}

} // namespace iblgf

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    boost::mpi::environment env(argc, argv);
    return RUN_ALL_TESTS();
}
