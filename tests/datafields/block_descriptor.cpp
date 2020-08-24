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
#include <iblgf/domain/dataFields/blockDescriptor.hpp>

namespace iblgf
{
namespace domain
{
namespace test
{
/**  @brief Test fixure for block_iterator unit test */
class block_descriptor_test : public ::testing::Test
{
  public:
    static constexpr int Dim = 3;
    using block_descriptor_type = BlockDescriptor<int, Dim>;
    using coordinate_type = block_descriptor_type::coordinate_type;

  public:
    block_descriptor_test()
    : b0(coordinate_type(-1), coordinate_type({3, 3, 4}), 1)
    {
    }

  protected:
    block_descriptor_type b0;
};

TEST_F(block_descriptor_test, transform)
{
    auto b_tmp = b0;

    //scale: Odd extent
    EXPECT_EQ(b0.level(), 1);
    EXPECT_EQ(b0.min(), coordinate_type(-1));
    EXPECT_EQ(b0.max(), coordinate_type({1, 1, 2}));
    EXPECT_EQ(b0.extent(), coordinate_type({3, 3, 4}));

    b0.level_scale(2);

    EXPECT_EQ(b0.level(), 2);
    EXPECT_EQ(b0.min(), coordinate_type(-2));
    EXPECT_EQ(b0.max(), coordinate_type({3, 3, 5}));
    EXPECT_EQ(b0.extent(), coordinate_type({6, 6, 8}));

    b0.level_scale(1);

    EXPECT_TRUE(b0 == b_tmp);

    //non-exact scaling:
    b0.level_scale(0);
    EXPECT_EQ(b0.level(), 0);
    EXPECT_EQ(b0.min(), coordinate_type(-1));
    EXPECT_EQ(b0.max(), coordinate_type({0, 0, 1}));
    EXPECT_EQ(b0.extent(), coordinate_type({2, 2, 3}));

    b0.level_scale(1);
    EXPECT_FALSE(b0 == b_tmp);

    //shift
    b0 = b_tmp;
    auto shift = coordinate_type({1, 2, 3});
    b0.shift(shift);
    EXPECT_EQ(b0.min(), coordinate_type({0, 1, 2}));
    EXPECT_EQ(b0.max(), coordinate_type({2, 3, 5}));
    EXPECT_EQ(b0.extent(), coordinate_type({3, 3, 4}));

    //enlarge to fit
    block_descriptor_type blarge(coordinate_type(-10), coordinate_type(1));
    b0 = b_tmp;
    b0.enlarge_to_fit(blarge);
    EXPECT_EQ(b0.min(), coordinate_type(-10));
    EXPECT_EQ(b0.max(), coordinate_type({1, 1, 2}));
    EXPECT_EQ(b0.extent(), coordinate_type({12, 12, 13}));

    b0 = b_tmp;
    b0.grow(coordinate_type(1), coordinate_type(2));
    EXPECT_EQ(b0.min(), coordinate_type(-2));
    EXPECT_EQ(b0.max(), coordinate_type({3, 3, 4}));
    EXPECT_EQ(b0.extent(), coordinate_type({6, 6, 7}));
}

TEST_F(block_descriptor_test, overlap)
{
    block_descriptor_type b1(coordinate_type(0), coordinate_type(4), 1);

    block_descriptor_type overlap = b1;

    //Same level
    if (b0.overlap(b1, overlap))
    {
        EXPECT_EQ(overlap.min(), coordinate_type(0));
        EXPECT_EQ(overlap.max(), coordinate_type({1, 1, 2}));
        EXPECT_EQ(overlap.extent(), coordinate_type({2, 2, 3}));
    }
    //upscaled level
    if (b0.overlap(b1, overlap, 2))
    {
        EXPECT_EQ(overlap.min(), coordinate_type(0));
        EXPECT_EQ(overlap.max(), coordinate_type({3, 3, 5}));
        EXPECT_EQ(overlap.extent(), coordinate_type({4, 4, 6}));
    }

    b0.shift(coordinate_type(100));
    EXPECT_FALSE(b0.overlap(b1, overlap));
}

TEST_F(block_descriptor_test, queries)
{
    auto b_tmp = b0;

    //is_inside
    coordinate_type p(0);
    EXPECT_TRUE(b0.is_inside(p));
    p = b0.max();
    EXPECT_TRUE(b0.is_inside(p));
    p[0] += 1;
    EXPECT_FALSE(b0.is_inside(p));

    //on_boundary
    p = b0.max();
    EXPECT_TRUE(b0.on_boundary(p));
    p[0] = p[0] - 1;
    EXPECT_TRUE(b0.on_boundary(p));
    p = b0.max() - 1;
    EXPECT_FALSE(b0.on_boundary(p));

    p = b0.max();
    EXPECT_TRUE(b0.on_max_boundary(p));
    EXPECT_FALSE(b0.on_min_boundary(p));

    p = b0.min();
    EXPECT_TRUE(b0.on_min_boundary(p));
    EXPECT_FALSE(b0.on_max_boundary(p));

    block_descriptor_type btt;
    EXPECT_TRUE(btt.is_empty());

    EXPECT_EQ(b0.size(), 36);
}

TEST_F(block_descriptor_test, indices)
{
    coordinate_type p(0);

    EXPECT_EQ(b0.index(b0.base()), 0);
    EXPECT_EQ(b0.index_zeroBase(p), 0);
    EXPECT_EQ(b0.index(p), 13);
}

} //namespace test
} //namespace domain
} //namespace iblgf
