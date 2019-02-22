#ifndef INCLUDED_H5_IO_HPP
#define INCLUDED_H5_IO_HPP


#include <ostream>
#include <fstream>
#include <iostream>
#include <sstream>

// IBLGF-specific
#include <global.hpp>
//#include <setups/tests/decomposition/decomposition_test.hpp>
#include <domain/dataFields/blockDescriptor.hpp>
#include <domain/octree/tree.hpp>
#include <domain/dataFields/datafield.hpp>
#include <IO/chombo/chombo.hpp>
#include <IO/chombo/h5_file.hpp>


namespace io
{



template<int Dim, class Domain>
class H5_io
{

public:

    using value_type             = double;
    using coordinate_type        = typename Domain::coordinate_type;
    using datablock_t            = typename Domain::datablock_t;
    using field_type_iterator_t  = typename Domain::field_type_iterator_t;
    using field_tuple            = typename datablock_t::fields_tuple_t;
    //using tree_t                 = octree::Tree<Dim,datablock_t>;
    //using octant_type            = typename tree_t::octant_type;
    using octant_type            = typename Domain::octant_t;

    template<typename T, std::size_t D>
    using vector_t=math::vector<T, D>;

    using base_t = vector_t<int,Dim>;
    using extent_t = vector_t<int,Dim>;
    using blockDescriptor_t = typename domain::BlockDescriptor<int, Dim>;


    //using node_t = typename NodeType<DataBlock>;
    using blockData_t = typename datablock_t::node_field_type;
    using write_info_t = typename std::pair<blockDescriptor_t,blockData_t*>;
    using block_info_t = typename std::tuple<int, blockDescriptor_t, blockData_t*>;

    using chombo_t= typename chombo_writer::Chombo<Dim, blockDescriptor_t,
                            blockData_t, Domain, hdf5_file<Dim, base_t> >;

public:

    struct BlockInfo
    {
        int rank;
        blockDescriptor_t blockDescriptor;
        blockData_t* blockData;

        // Ctors:
        BlockInfo()=default;
        ~BlockInfo()=default;

        BlockInfo(const BlockInfo& rhs)=default;
        BlockInfo& operator=(const BlockInfo& rhs)=default;

        BlockInfo(BlockInfo&& rhs)=default;
        BlockInfo& operator=(BlockInfo& rhs)=default;

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
            blockDescriptor = _leaf->data()->descriptor();
            blockData = &(_leaf->data()->node_field());
        }
    };

public:

    void write_h5(std::string _filename, Domain& _lt )
    {
        boost::mpi::communicator world;

        // vector of pairs of descriptors and field data
        std::vector<write_info_t> data_info;
     //   std::vector<BlockInfo> block_distribution;
        std::vector<block_info_t> block_distribution;


        int nPoints=0;
        for(auto it=_lt.begin_leafs();it!=_lt.end_leafs();++it)
        {
            auto b= it->data()->descriptor();
            b.grow(0,1); //grow by one to fill the gap
            nPoints+= b.nPoints();
        }

        std::cout<<"Rank= "<<world.rank()<<".  ------------POINTS "<<nPoints<<" float"<<std::endl;
        int _count=0;
        // Collect block descriptor and data from each block
        for(auto it=_lt.begin_leafs();it!=_lt.end_leafs();++it)
        {
            int rank = it->rank();
            blockDescriptor_t block =it->data()->descriptor();
            blockData_t* node_data=&(it->data()->node_field());

       //     world.barrier();
            std::cout<<"Rank = "<<world.rank()<<". Count = "<<_count
                    <<". Add Block to level = "<<block.level()
                    <<". octant level = "<<it->refinement_level()<<std::endl;
           // data_info.push_back(std::make_pair(block,node_data));


            block_info_t blockInfo = std::make_tuple(rank, block, node_data);
            block_distribution.push_back(blockInfo);

            it->index(_count*block.nPoints());
            ++_count;
        }


        world.barrier();
        if (_lt.is_server()) {
            std::cout<<"\n=========World Barrier=========\n"<<std::endl;
        }

        std::cout<<"Create hdf5_file object with rank "<<world.rank()<<" of "<<world.size()<<std::endl;

        world.barrier();
        if (_lt.is_server()) {
            std::cout<<"\n=========World Barrier=========\n"<<std::endl;
        }

        hdf5_file<Dim> chombo_file(_filename);
     //   chombo_t ch_writer(data_info);  // Initialize writer with vector of
                                        // info: pair of descriptor and data
        chombo_t ch_writer(block_distribution);  // Initialize writer with vector of
                                         // info: tuple of rank, descriptor and data
        world.barrier();
        if (_lt.is_server()) {
            std::cout<<"\n=========World Barrier=========\n"<<std::endl;
        }

        ch_writer.write_global_metaData(&chombo_file);

        world.barrier();
        if (_lt.is_server()) {
            std::cout<<"\n=========World Barrier=========\n"<<std::endl;
        }
        std::cout<<"------>write_level_info: rank "<<world.rank()<<" of "<<world.size()<<std::endl;

        ch_writer.write_level_info(&chombo_file);

        std::cout<<"<-----write_h5: End with rank "<<world.rank()<<" of "<<world.size()<<std::endl;
    }

};

}

#endif
