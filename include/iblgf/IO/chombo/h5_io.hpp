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

#ifndef INCLUDED_H5_IO_HPP
#define INCLUDED_H5_IO_HPP

#include <ostream>
#include <fstream>
#include <iostream>
#include <sstream>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>
#include <iblgf/domain/octree/tree.hpp>
#include <iblgf/domain/dataFields/datafield.hpp>
#include <iblgf/IO/chombo/chombo.hpp>
#include <iblgf/IO/chombo/h5_file.hpp>
#include <iblgf/IO/parallel_ostream.hpp>

namespace iblgf
{
namespace io
{
template<int Dim, class Domain>
class H5_io
{
  public:
    using value_type = double;
    using coordinate_type = typename Domain::coordinate_type;
    using datablock_t = typename Domain::datablock_t;
    using field_type_iterator_t = typename Domain::field_type_iterator_t;
    using field_tuple = typename datablock_t::fields_tuple_t;
    using octant_type = typename Domain::octant_t;

    template<typename T, std::size_t D>
    using vector_t = math::vector<T, D>;

    using base_t = vector_t<int, Dim>;
    using extent_t = vector_t<int, Dim>;
    using blockDescriptor_t = typename domain::BlockDescriptor<int, Dim>;

    using blockData_t = typename datablock_t::node_field_type;
    //    using write_info_t = typename std::pair<blockDescriptor_t,blockData_t*>;
    //    using block_info_t = typename std::tuple<int, blockDescriptor_t, blockData_t*>;

    using chombo_t = typename chombo_writer::Chombo<Dim, blockDescriptor_t,
        blockData_t, Domain, hdf5_file<Dim, base_t>>;

  public:
    struct BlockInfo
    {
        int               rank;
        blockDescriptor_t blockDescriptor;
        blockData_t*      blockData;

        // Ctors:
        BlockInfo() = default;
        ~BlockInfo() = default;

        BlockInfo(const BlockInfo& rhs) = default;
        BlockInfo& operator=(const BlockInfo& rhs) = default;

        BlockInfo(BlockInfo&& rhs) = default;
        BlockInfo& operator=(BlockInfo& rhs) = default;

        // Should probably be const
        BlockInfo(int _rank, blockDescriptor_t _blockDescriptor,
            blockData_t* _blockData)
        {
            rank = _rank;
            blockDescriptor = _blockDescriptor;
            blockData = _blockData;
        }

        BlockInfo(const octant_type* _leaf)
        {
            rank = _leaf->rank();
            blockDescriptor = _leaf->data().descriptor();
            blockData = &(_leaf->data().node_field());
        }
    };

  public:
    template<typename Field>
    void read_h5(std::string _filename, Domain* _lt)
    {
        pcout << "Start reading file -> " << _filename << std::endl;
        boost::mpi::communicator world;

        auto octant_blocks = blocks_list_build(_lt);

        hdf5_file<Dim> chombo_file(_filename, true);
        chombo_t       ch_writer(
            octant_blocks); // Initialize writer with vector of octants
        ch_writer.template read_u<Field>(&chombo_file, octant_blocks, _lt);
    }

    void write_h5(
        std::string _filename, Domain* _lt, bool include_correction = false)
    {
        auto octant_blocks = blocks_list_build(_lt, include_correction);

        hdf5_file<Dim> chombo_file(_filename);

        chombo_t ch_writer(
            octant_blocks); // Initialize writer with vector of octants

        ch_writer.write_global_metaData(&chombo_file, _lt->dx_base());
        ch_writer.write_level_info(&chombo_file);
    }

    std::vector<octant_type*> blocks_list_build(
        Domain* _lt, bool include_correction = false)
    {
        boost::mpi::communicator world;

        int nPoints = 0;
        for (auto it = _lt->begin(); it != _lt->end(); ++it)
        {
            if (!it->has_data() || it->refinement_level() < 0) continue;
            if (!include_correction && it->is_correction()) continue;
            auto b = it->data().descriptor();
            b.grow(0, 1); //grow by one to fill the gap
            nPoints += b.nPoints();
        }

        std::vector<octant_type*> octant_blocks;
        int                       _count = 0;
        // Collect block descriptor and data from each block
        for (auto it = _lt->begin(); it != _lt->end(); ++it)
        {
            if (!it->has_data() || it->refinement_level() < 0 ||
                (world.rank() > 0 && !it->locally_owned()))
                continue;
            if (!include_correction && it->is_correction() && !it->is_leaf())
                continue;
            int rank = it->rank();

            if (rank == world.rank() || world.rank() == 0)
            {
                blockDescriptor_t block = it->data().descriptor();
                octant_blocks.push_back(it.ptr());

                it->index(_count * block.nPoints());
            }
            ++_count;
        }

        return octant_blocks;
    }
    parallel_ostream::ParallelOstream pcout =
        parallel_ostream::ParallelOstream(1);
};

} // namespace io
} // namespace iblgf

#endif
