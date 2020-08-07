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
#include <iblgf/domain/dataFields/datafield.hpp>
#include <iblgf/domain/dataFields/dataBlock.hpp>
#include <iblgf/domain/dataFields/node.hpp>

namespace iblgf
{
namespace domain
{
using namespace types;

TEST(view_test, views)
{
    //Create a datablock with a scalar and a vector field

    // clang-format off
    static constexpr int Dim = 3;
    REGISTER_FIELDS
    (Dim,
     (
        (p,    float_type,  1,  1,  1,  cell,true)
     )
    )
    // clang-format on
    using datablock_t = DataBlock<Dim, node, p_type>;

    using blockd_type = BlockDescriptor<int, Dim>;
    using coordinate_type = blockd_type::coordinate_type;

    blockd_type blockD(coordinate_type(0), coordinate_type(8));
    datablock_t db(blockD);

    //Set some values for the datafields
    db(p) = -1;

    /************************************************************************/
    //Getting a boundary view from a field:
    //A view is constructed from a block_descriptor to set the viewing
    //area and a stride
    //view0 is edge along z
    auto base = db.base();
    auto extent = coordinate_type(1);
    extent[2] = db.extent()[2];
    const auto  stride = coordinate_type(1);
    blockd_type view_area(db.base(), extent);

    auto view0 = db(p).view(view_area, stride);

    //x-slices (min&max)
    view_area.base() = db.min();
    view_area.extent() = db.extent();
    view_area.extent()[0] = 1;
    auto view1 = db(p).view(view_area, stride);

    view_area.base()[0] = db.max()[0];
    auto view2 = db(p).view(view_area, stride);

    /************************************************************************/
    //Iteration:

    //Ranged-based
    int count = 0;
    for (auto& e : view0) { e = count++; }
    EXPECT_EQ(count, 8);

    //old iterators
    coordinate_type c(0);
    for (auto it = view0.begin(); it != view0.end(); ++it)
    {
        EXPECT_EQ(it.coordinate(), c);
        *it = c[2];
        ++c[2];
    }

    //lambda function
    count = 0;
    c = coordinate_type(0);
    view0.iterate([&count](const auto& val) {
        EXPECT_EQ(val, count);
        ++count;
    });

    count = 0;
    for (auto& e : view1) { e = count++; }
    EXPECT_EQ(count, 64);
    count = 0;
    for (auto& e : view2) { e = count++; }
    EXPECT_EQ(count, 64);

    bool verbose = false;
    if (verbose)
    {
        std::cout << view1 << std::endl;
        for (auto it = view2.begin(); it != view2.end(); ++it)
        { std::cout << it.coordinate() << std::endl; }
    }
    /************************************************************************/
    //Assign view
    view0 = 10;
    for (auto& e : view0) { EXPECT_EQ(e, 10); }

    view0 += 1;
    for (auto& e : view0) { EXPECT_EQ(e, 11); }
    view0 -= 1;
    for (auto& e : view0) { EXPECT_EQ(e, 10); }
    view0 *= 2;
    for (auto& e : view0) { EXPECT_EQ(e, 20); }
    view0 /= 2;
    for (auto& e : view0) { EXPECT_EQ(e, 10); }

    view1 = 10;
    view2 = 20;

    view1 += view2;
    for (auto& e : view1) { EXPECT_EQ(e, 30); }
    view1 -= view2;
    for (auto& e : view1) { EXPECT_EQ(e, 10); }

    view2 = 2;
    view1 *= view2;
    for (auto& e : view1) { EXPECT_EQ(e, 20); }
    view1 /= view2;
    for (auto& e : view1) { EXPECT_EQ(e, 10); }
}

} // namespace domain
} //namespace iblgf
