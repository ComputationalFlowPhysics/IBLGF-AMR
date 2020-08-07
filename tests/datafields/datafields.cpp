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
#include <boost/filesystem.hpp>
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

TEST(datafield_test, ctors)
{
    std::cout << "Testing the datafields" << std::endl;
    static constexpr int Dim = 3;
    static constexpr int Buff = 1;

    using block_d_type = BlockDescriptor<int, Dim>;
    using coordinate_type = typename block_d_type::coordinate_type;

    //Manual way to generate fields
    static constexpr tuple_tag_h f0_tag{"f0"};
    using f0_traits = field_traits<tag_type<f0_tag>, float_type, 3, Buff, Buff,
        MeshObject::cell, Dim, true>;
    using f0_type = Field<f0_traits>;

    static constexpr tuple_tag_h f1_tag{"f1"};
    using f1_traits = field_traits<tag_type<f1_tag>, float_type, 1, Buff, Buff,
        MeshObject::cell, Dim, true>;
    using f1_type = Field<f1_traits>;

    //Using convience macro to generate fields:
    // clang-format off
    REGISTER_FIELDS
    (Dim,
     //Fields tuples
     (
        (p,    float_type,  1,  1,  1,  cell,true),
        (vel,  float_type,  3,  1,  1,  face,false)
     )

     )
    // clang-format on

    using datablock_t =
        DataBlock<Dim, node, f0_type, f1_type, p_type, vel_type>;

    BlockDescriptor<int, Dim> blockD(coordinate_type(0), coordinate_type(8));
    datablock_t               db(blockD);
    auto&                     vel_field = db[vel];
    std::cout << vel_field.name() << std::endl;
    db.for_fields(
        [](const auto& f) { std::cout << "Field: " << f.name() << std::endl; });

    auto& pfield1 = db(p);
    std::cout << pfield1.size() << std::endl;
    auto& pfield2 = db(p.tag());
    std::cout << pfield2.size() << std::endl;
    auto& pfield3 = db(p_type::tag());
    std::cout << pfield3.size() << std::endl;

    //Iteration over a fields, scalar or vector:
    //Scalar field:
    for (auto& el : db(p)) { el = -3; }
    //Vector field:
    for (std::size_t i = 0; i < vel.nFields(); ++i)
    {
        for (auto& el : db(vel, 0)) { el = 10; }
    }

    //Iteration over entire datablock:
    int count = 0;
    for (auto& node : db)
    {
        EXPECT_EQ(node(p), -3);
        EXPECT_EQ(node(vel, 0), 10);
        EXPECT_EQ(node(vel, 1), 0);
        EXPECT_EQ(node(vel, 2), 0);

        if (count++ == 10) std::cout << "Fields on node: " << node << std::endl;
    }

    //Operator overloads:

    //Scalar assignment:
    db(p) = -1;
    for (auto& node : db) EXPECT_EQ(node(p), -1);
    db(vel, 0) = -1;
    for (auto& node : db) EXPECT_EQ(node(vel, 0), -1);

    //Unary operators
    db(p) += 3;
    for (auto& node : db) EXPECT_EQ(node(p), 2);
    db(p) -= 1;
    for (auto& node : db) EXPECT_EQ(node(p), 1);
    db(p) /= 2;
    for (auto& node : db) EXPECT_EQ(node(p), 0.5);
    db(p) *= 2;
    for (auto& node : db) EXPECT_EQ(node(p), 1);

    //Binary operators
    db(p) = 1;
    db(vel, 0) = 2;
    auto p3 = db(vel, 0) + db(p);
    for (auto& node : p3) EXPECT_EQ(node, 3);
    auto p4 = db(vel, 0) - db(p);
    for (auto& node : p4) EXPECT_EQ(node, 1);
    auto p5 = db(vel, 0) * db(p);
    for (auto& node : p5) EXPECT_EQ(node, 2);
    auto p6 = db(vel, 0) / db(p);
    for (auto& node : p6) EXPECT_EQ(node, 2);
}

} // namespace domain
} //namespace iblgf
