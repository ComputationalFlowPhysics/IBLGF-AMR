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

#include <gtest/gtest.h>
#include <filesystem>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/types.hpp>
#include <iblgf/domain/octree/tree.hpp>

namespace iblgf
{
namespace octree
{
using namespace types;
/** @brief Test fixure for oct-tree */
class tree_tests : public ::testing::Test
{
  public:
    struct block
    {
        int dat = -1;
    };

  public:
    static constexpr int Dim = 3;
    using tree_type = Tree<Dim, block>;
    using coordinate_type = typename tree_type::coordinate_type;
    using key_type = typename tree_type::key_type;

  public:
    tree_tests()
    : ext(5)
    , t(ext)
    {
    }

  protected:
    coordinate_type ext;
    tree_type       t;
};

//TODO:  Clean iterators, clean blockDescriptor
//  tuple utils
TEST_F(tree_tests, Construction)
{

    coordinate_type p0(0), p1(6);
    ASSERT_EQ(t.base_level(), 3);
    ASSERT_EQ(t.depth(), 4);

    auto it = t.find(p0, t.base_level());
    ASSERT_NE(it, t.end());
    auto it2 = t.find(p1, t.base_level()+1);
    ASSERT_EQ(it2, t.end());


    ext += 1;
    rcIterator<Dim>::apply(coordinate_type(0), ext, [&](const auto& _p) {
        auto it = t.find(_p);
        if (it != t.end()) { EXPECT_TRUE(it->is_leaf_search()); }
        else
        {
            // Cannot be found
            const auto Linf = norm_inf(_p);
            EXPECT_GE(Linf, 4);
        }
    });
    for (auto it = t.begin(); it != t.end(); ++it)
    {
        if (it->level() == t.base_level()) { EXPECT_TRUE(it->is_leaf_search()); }
        else
        {
            EXPECT_FALSE(it->is_leaf());
        }
    }
}

} // namespace domain
} //namespace iblgf
