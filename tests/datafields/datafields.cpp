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

TEST(datafield_test, ctors)
{
    std::cout<<"Testing the datafields"<<std::endl;
    static constexpr int Dim=3;
    static constexpr int Buff=1;

    //Manual way to generate fields
    static constexpr static_tag       f0_tag{"f0"};
    static constexpr tag_type<f0_tag> f0{};
    using f0_traits = field_traits<tag_type<f0_tag>, float_type, 3, Buff, Buff,
        MeshObject::cell, Dim, true>;
    using f0_field_type = Field<f0_traits>;

    static constexpr static_tag       f1_tag{"f1"};
    static constexpr tag_type<f1_tag> f1{};
    using f1_traits = field_traits<tag_type<f1_tag>, float_type, 1, Buff, Buff,
        MeshObject::cell, Dim, true>;
    using f1_field_type = Field<f1_traits>;

    ////Using convience macro to generate fields:
    REGISTER_FIELDS
    (Dim,
    //Fields tuples
    (
        (p,    float_type,  1,  1,  1,  cell,true),
        (vel,  float_type,  3,  1,  1,  face,false)
    )

    )

    using datablock_t = DataBlock<Dim, node,
                                  f0_field_type,
                                  f1_field_type,
                                  p, 
                                  vel
                                 >;
    datablock_t db;
    auto& vel_field=db(f1_traits{});
    //auto& pfield=db(p_field_tag);
    //auto& f1_field=db.template get<f1_field_type>();

}
} //namespace iblgf
} //namespace iblgf
