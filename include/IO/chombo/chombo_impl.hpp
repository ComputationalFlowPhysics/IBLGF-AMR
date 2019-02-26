#ifndef CHOMBOIMPL_HPP
#define CHOMBOIMPL_HPP

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <array>
#include <map>
#include <tuple>
#include <set>
#include <cmath>
#include <vector>

#include <IO/chombo/h5_file.hpp>
#include <IO/chombo/h5_io.hpp>
#include <domain/dataFields/blockDescriptor.hpp>
#include <global.hpp>
#include <boost/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
//#include <setups/tests/decomposition/decomposition_test.hpp>

namespace chombo_writer
{


template<std::size_t Dim, class BlockDescriptor, class FieldData,
                                class Domain, class HDF5File>
class Chombo
{

public: //memeber type:
    using value_type = double;
    using hsize_type = typename HDF5File::hsize_type;
    using field_type_iterator_t  = typename Domain::field_type_iterator_t;

    using write_info_t = typename std::pair<BlockDescriptor,FieldData*>;
    using block_info_t = typename std::tuple<int,BlockDescriptor,FieldData*>;
    using index_list_t = typename HDF5File::index_list_t;
//    using h5_io_t = io::H5_io<Dim, Domain>;
//    using block_info_t = h5_io_t::BlockInfo;

    using key_type = typename Domain::key_t;
    using offset_type = int;
//    using offset_vector = std::vector<offset_type>;
    using offset_vector = std::vector<offset_type>;
//    using offset_map_per_rank = std::map<int,offset_map>;

    using map_type = typename boost::unordered_map<key_type,offset_type>;

public: //Ctors:
    Chombo()=default;
    ~Chombo()=default;

    Chombo(const Chombo& rhs)=default;
    Chombo& operator=(const Chombo& rhs)=default;

    Chombo(Chombo&& rhs)=default;
    Chombo& operator=(Chombo& rhs)=default;

    struct LevelInfo
    {
        BlockDescriptor probeDomain;
        int level;
        std::vector<int> ranks;
        std::vector<BlockDescriptor> blocks;
        std::vector<FieldData*> fields;

        bool operator<(const LevelInfo& _other) const
        {
            return level<_other.level;
        }
    };



    //Chombo(const std::vector<BlockDescriptor>& _blocks)
    Chombo(const std::vector<write_info_t>& _data_blocks)
    {
        init(_data_blocks);
    }

    Chombo(const std::vector<block_info_t>& _data_blocks)
    {
        init(_data_blocks);
    }

private:

    void init(const std::vector<block_info_t>& _data_blocks)
    {
        //Blocks per level:
        for(auto& p: _data_blocks)
        {

            const int rank = std::get<0>(p);            // first is rank
            const auto b = std::get<1>(p);              // second is descriptor
            const auto field_data = std::get<2>(p);     // third is data
            auto it=level_map_.find(b.level());     // returns iterator to element with key b.level()
            // FIXME: Some blocks have -1 level
            if (b.level() >= 0) {
            if(it!=level_map_.end())                // if found somewhere (does not return end())
            {
                it->second.ranks.push_back(rank);
                it->second.blocks.push_back(b);     // add block to corresponding level
                it->second.fields.push_back(field_data);

                //Adapt probeDomain if there is a block beyond it
                for(std::size_t d=0; d<Dim;++d)
                {
                    if(it->second.probeDomain.min()[d] > b.min()[d])
                    {
                        it->second.probeDomain.min()[d]=b.min()[d];
                    }

                    if(it->second.probeDomain.max()[d] < b.max()[d])
                    {
                        it->second.probeDomain.extent()[d]=
                            b.max()[d]-it->second.probeDomain.base()[d]+1;
                    }
                }
            }
            else        // if not found, simply create that level and insert the block
            {
                std::cout<<"Adding new level : level "<<b.level()<<std::endl;
                LevelInfo l;
                l.level=b.level();
                l.ranks.push_back(rank);
                l.blocks.push_back(b);
                l.fields.push_back(field_data);
                l.probeDomain=b;
                level_map_.insert(std::make_pair(b.level(),l ));

            }

            }

        }
    }


    //void init(const std::vector<BlockDescriptor>& _blocks)
    void init(const std::vector<write_info_t>& _data_blocks)
    {

        //Blocks per level:
        for(auto& p: _data_blocks)
        {

            const auto b = p.first;   // first is block
            //const auto b = &(p.first);
            const auto field_data = p.second;       // second is data
//            printf("Adding block \n");



            auto it=level_map_.find(b.level());     // returns iterator to element with key b.level()
            if(it!=level_map_.end())                // if found somewhere (does not return end())
            {
//                printf("Current min = %d, %d, %d \n",
//                    it->second.probeDomain.min()[0],it->second.probeDomain.min()[1],
//                    it->second.probeDomain.min()[2]);
//                printf("Current max = %d, %d, %d \n",
//                    it->second.probeDomain.max()[0],it->second.probeDomain.max()[1],
//                    it->second.probeDomain.max()[2]);

//                std::cout<<"Add to existing level : level "<<b.level()<<std::endl;
                it->second.blocks.push_back(b);     // add block to corresponding level
                it->second.fields.push_back(field_data);

                //Adapt probeDomain if there is a block beyond it
                for(std::size_t d=0; d<Dim;++d)
                {
                    if(it->second.probeDomain.min()[d] > b.min()[d])
                    {
                        it->second.probeDomain.min()[d]=b.min()[d];
                    }

                    if(it->second.probeDomain.max()[d] < b.max()[d])
                    {
                        it->second.probeDomain.extent()[d]=
                            b.max()[d]-it->second.probeDomain.base()[d]+1;
                    }
                }
//                printf("New min = %d, %d, %d \n",
//                    it->second.probeDomain.min()[0],it->second.probeDomain.min()[1],
//                    it->second.probeDomain.min()[2]);
//                printf("New max = %d, %d, %d \n",
//                    it->second.probeDomain.max()[0],it->second.probeDomain.max()[1],
//                    it->second.probeDomain.max()[2]);
            }
            else        // if not found, simply create that level and insert the block
            {
                std::cout<<"Adding new level : level "<<b.level()<<std::endl;
                LevelInfo l;
                l.level=b.level();
                l.blocks.push_back(b);
                l.fields.push_back(field_data);
                l.probeDomain=b;
                level_map_.insert(std::make_pair(b.level(),l ));

//                printf("Level base = %d, %d, %d \n",
//                        l.probeDomain.base()[0],l.probeDomain.base()[1],
//                        l.probeDomain.base()[2]);
//                printf("Level extent = %d, %d, %d \n",
//                        l.probeDomain.extent()[0],l.probeDomain.extent()[1],
//                        l.probeDomain.extent()[2]);
            }

//            printf("Done Adding block \n");

        }
    }

public:

    void write_global_metaData( HDF5File* _file ,
                     value_type _time=0.0,
                     int _dt=1,
                     int _ref_ratio=2)
    {
        boost::mpi::communicator world;

        std::cout<<"------Write global meta data with rank "
                 <<world.rank()<<" of "<<world.size()<<"------"<<std::endl;

        auto root = _file->get_root();

        // iterate through field to get field names
        std::vector<std::string> components;
        if (world.rank()==0) std::cout<<"Components are : "<<std::endl;
        field_type_iterator_t::for_types([&components, &world]<typename T>()
        {
            std::string name = std::string(T::name());
            if (world.rank()==0) std::cout<<name<<std::endl;
            components.push_back(name);
        });

        // Write components, number of components and number of levels
        for(std::size_t i=0; i<components.size(); ++i)
        {
            _file->template create_attribute<std::string>(root,
                "component_"+std::to_string(i),components[i]);
        }

        // Get number of levels from server and communicate to all ranks
        int num_levels = 0;
        if (world.rank()==0) {
            num_levels = level_map_.size();
        }
        std::cout<<"I am rank "<<world.rank()<<" and numer of levels is "<<num_levels<<std::endl;

        world.barrier();
        boost::mpi::broadcast(world, num_levels, 0); // send from server to all others

        std::cout<<"After broadcast: I am rank "<<world.rank()<<" and number of levels is "<<num_levels<<std::endl;

        _file->template create_attribute<int>(root, "num_levels",num_levels);

        // num_components
        const int num_components = components.size();
        _file->template create_attribute<int>(root, "num_components",static_cast<int>(num_components));

        // Ordering of the dataspaces
        _file->template create_attribute<std::string>(root,
                "filetype","VanillaAMRFileType");
        // Node-centering=7, zone-centering=6
        _file->template create_attribute<int>(root,"data_centering",6);

        auto global_id=_file->create_group(root,"Chombo_global");
        _file->template create_attribute<int>(global_id,
                "SpaceDim", static_cast<int>(Dim));
        _file->close_group(global_id);

        // *******************************************************************
        // Write level structure and collective calls (everything except data

        // Use # of levels to write each level_* group
        std::cout<<"Create level structure. num_levels = "<<num_levels<<std::endl;
        for (int lvl = 0; lvl < num_levels; ++lvl) {
            auto group_id_lvl = _file->create_group(root,"level_"+std::to_string(lvl));

            // Write Attributes -----------------------------------------------
            // dt
            _file->template create_attribute<int>(group_id_lvl,"dt",_dt);

            // dx
            value_type dx=1./(std::pow(2,lvl));   // dx = 1/(2^i)
            _file->template create_attribute<value_type>(group_id_lvl,"dx",dx);

            // ref_ratio
            _file->template create_attribute<int>(group_id_lvl,
                                                    "ref_ratio",_ref_ratio);

            // time
            _file->template create_attribute<int>(group_id_lvl,"time",_time);

            // data attributes
            auto group_id_dattr=_file->create_group(group_id_lvl,
                    "data_attributes");
            _file->template create_attribute<std::string>(group_id_dattr,
                    "objectType","CArrayBox");

            // prob_domain ****************************************************
            // 1: Server get prob_domain
            index_list_t min_cellCentered = 0;
            index_list_t max_cellCentered = 0;
            if (world.rank()==0) {
                auto it_lvl = level_map_.find(lvl);
                auto l = it_lvl->second;
                max_cellCentered=l.probeDomain.max();
                //Cell-centered data: ??? Seems fine without
                //max_cellCentered-=1;
            }
            // 2: Server broadcast prob_domain
            boost::mpi::broadcast(world, min_cellCentered[0], 0);
            boost::mpi::broadcast(world, min_cellCentered[1], 0);
            boost::mpi::broadcast(world, min_cellCentered[2], 0);
            boost::mpi::broadcast(world, max_cellCentered[0], 0);
            boost::mpi::broadcast(world, max_cellCentered[1], 0);
            boost::mpi::broadcast(world, max_cellCentered[2], 0);

            // 3: All write prob_domain
            //_file->template write_boxCompound<Dim>(group_id, "prob_domain",
            //  l.probeDomain.min(), max_cellCentered);      // write prob_domain as attribute
            _file->template write_boxCompound<Dim>(group_id_lvl, "prob_domain",
              min_cellCentered, max_cellCentered);      // write prob_domain as attribute

            // Create "boxes" dataset: ****************************************
            hsize_type boxes_size = 0;

            if (world.rank()==0) {
                auto it_lvl = level_map_.find(lvl);
                auto l = it_lvl->second;
                auto p= l.blocks.begin()->min();        // vector of ints size 3
                auto p2= l.blocks.begin()->max();
                std::vector<decltype(p)> mins;
                std::vector<decltype(p2)> maxs;

                for(std::size_t j=0;j<l.blocks.size();++j)
                {
                    mins.push_back(l.blocks[j].min());
                    maxs.push_back(l.blocks[j].max());
                }

                boxes_size = mins.size();
            }
            boost::mpi::broadcast(world, boxes_size, 0); // send from server to all others

            _file->template create_boxCompound<Dim>(group_id_lvl, "boxes", boxes_size, false);
            //_file->template create_boxCompound<Dim>(group_id, "boxes", mins, maxs, false);


            // Create dataset for data ****************************************
            // Server gets size:
            hsize_type offsets_size = 0;
            hsize_type dset_size = 0;   // Count size of dataset for this level
            if (world.rank()==0) {
                // Write DATA for different components-----------------------------
                auto it_lvl = level_map_.find(lvl);
                auto l = it_lvl->second;

                // Determine size of dataset for this level to create dataset
                for(std::size_t b=0; b<l.blocks.size(); ++b)
                {
                    hsize_type nElements_patch=1;
                    for(std::size_t d=0; d<Dim; ++d)        // for node-centered
                    {
                       nElements_patch *= l.blocks[b].extent()[d];
                    }
                    dset_size+=nElements_patch*num_components;
                }
                    std::cout<<"Final dataset size = "<<dset_size<<std::endl;

                // Get size of "offsets" (number of blocks +1)
                offsets_size = l.blocks.size()+1;
                std::cout<<"Offsets_size = "<<offsets_size<<std::endl;
            }

            // Scatter size of boxes
            boost::mpi::broadcast(world, offsets_size, 0); // send from server to all others

            // Create offsets dataset
            auto space_attr =_file->create_simple_space(1, &offsets_size, NULL);
            auto dset_Attr=_file->template create_dataset<int>(group_id_dattr,
                        "offsets",space_attr);
                // _file->template write<int>(dset_Attr,size, &patch_offsets_vec[0]);
            _file->close_dset(dset_Attr);

            // send size to all clients
            std::cout<<"I am rank "<<world.rank()<<" and dset_size is "<<dset_size<<std::endl;
            world.barrier();
            boost::mpi::broadcast(world, dset_size, 0); // send from server to all others
            std::cout<<"After broadcast: I am rank "<<world.rank()<<" and dest_size is "<<dset_size<<std::endl;


            // Create full empty dataset with all processes
            auto space =_file->create_simple_space(1, &dset_size, NULL);
            // ***TODO: Make collective
            auto dset_id=_file->template create_dataset<value_type>(group_id_lvl,
                "data:datatype=0", space);      //collective
            _file->close_space(space);

        }




        _file->close_group(root);
    }



    //TODO: write actual data in parallel
    void write_level_info(HDF5File* _file,
                     value_type _time=0.0,
                     int _dt=1,
                     int _ref_ratio=2)
    {
        boost::mpi::communicator world;

        std::cout<<"--> Write level info with rank "<<world.rank()<<" of "<<world.size()<<std::endl;

        //using hsize_type =H5size_type;
        auto root = _file->get_root();

        // get field names and number of components
        std::vector<std::string> components;
        field_type_iterator_t::for_types([&components]<typename T>()
        {
            std::string name = std::string(T::name());
            components.push_back(name);
        });
        const int num_components = components.size();

        std::cout<<"Create offset vector"<<std::endl;
        std::vector < std::vector < std::vector<offset_type> > > offset_vector;
        for(int i = 0; i < world.size(); i++)
        {
            std::vector < std::vector < offset_type > > w;
            offset_vector.push_back( w );
            for(int j = 0; j < 19; j++)
            {
                std::vector <int> v;
                offset_vector[i].push_back( v );
            }
        }

        if (world.rank()==0) {
            // Loop initially to get offsets from rank==0
            for(auto& lp : level_map_)
            {
                auto l = lp.second;     // LevelInfo
                int lvl = l.level;
             //   offset_vector block_offsets;
                offset_type offset = 0;             // offset in dataset

                // Loop through blocks and
                // Determine size of dataset for this level to create dataset
                for(std::size_t b=0; b<l.blocks.size(); ++b)
                {
                    int rank = l.ranks[b];

                    // Loop through components in block
                    hsize_type nElements_patch=1;
                    for(std::size_t d=0; d<Dim; ++d)        // for node-centered
                    {
                       nElements_patch *= l.blocks[b].extent()[d];
                    }

                    int data_block_size=nElements_patch*num_components;

                    offset_vector[rank][lvl].push_back(offset);

                 //   std::cout<<"==| Level is "<<lvl<<". Rank is "<<rank<<". Block is "<<b<<". Offset is "<<offset<<std::endl;
                    std::cout<<"Lvl "<<lvl<<". b = "<<b;
                    std::cout<<". Offset = "<<offset;
                    std::cout<<". Base = "<<l.blocks[b].base()
                              <<". Max = "<<l.blocks[b].max()<<std::endl;


                    offset+=data_block_size;
                }

            }

        }

        // CHECK: Print out offsets
        std::cout<<"Print offsets: "<<std::endl;
        for (int i=0; i<world.size(); i++) {
            world.barrier();
            if (world.rank()==0) {
                std::cout<<"========== Rank is "<<i<<" =========="<<std::endl;
                for (int j=0; j<offset_vector[i].size(); j++) {
                    std::cout<<"========== Level is "<<j<<" =========="<<std::endl;

                    std::cout<<"Offset is ";
                    for (int k=0; k<offset_vector[i][j].size(); k++) {
                        std::cout<<offset_vector[i][j][k]<<", ";
                    }
                    std::cout<<std::endl;
                }
            }
        }


        // MPI Scatter the vectors of offsets

      //  std::vector<offset_type> offsets_test;
      //  offset_type offsets_test;


        std::cout<<" _____ START MPI SCCATTER ______ "<<std::endl;

        std::vector<std::vector<std::vector<offset_type>>> hello;
        for(int i = 0; i < world.size(); i++)
        {
            std::vector<std::vector< offset_type >> w;
            hello.push_back( w );
            for(int j = 0; j < 8; j++)
            {
                std::vector<offset_type> y;
                hello[i].push_back(y);

                for(int k = 0; k < 3; k++)
                {
                    int v = 100*i + 10*j + k;
                    hello[i][j].push_back( v );
                }
            }
        }

        std::vector<std::vector<offset_type>> hello_test;
        std::vector<std::vector<offset_type>> offsets_for_rank;

        boost::mpi::scatter(world, offset_vector, offsets_for_rank, 0);
        boost::mpi::scatter(world, hello, hello_test, 0);

        for (int i=0; i<world.size(); i++) {
            world.barrier();
            if (world.rank()==i) {
                std::cout<<">>>RANK IS "<<world.rank()<<std::endl;
                for (int j = 0; j < offsets_for_rank.size(); j++)
                {
                    for (int k = 0; k < offsets_for_rank[j].size(); k++)
                    {
                        std::cout<<"Rank is "<<world.rank()<<". offsets_for_rank = "<<offsets_for_rank[j][k]<<std::endl;
                    }
                }
            }
        }

        world.barrier();


        // Parallel Write -----------------------------------------------------
        // Loop over levels and write patches
        // level_map_ is map made of level (int) and LevelInfo (per Level)
        // (probeDomain, level, vector of blocks)
        for(auto& lp : level_map_)
        {
            // lp iterates over each pair of <int, LevelInfo> in the full map
            auto l = lp.second;     // LevelInfo
            int lvl = l.level;

            // Write attributes for this lvl (e.g. dx, dt)
            // ***TODO: Make collective
            auto group_id=_file->create_group(root,"level_"+std::to_string(lvl));
            std::cout<<"Group name is: level_"<<std::to_string(lvl)<<std::endl;


            /*****************************************************************/
            // Write level data
     //       if (world.rank()==0) {

                std::vector<int> patch_offsets_vec;
                patch_offsets_vec.push_back(0);

                // Write DATA for different components-----------------------------
                hsize_type dset_size = 0;   // Count size of dataset for this level

                // Determine size of dataset for this level to create dataset
                for(std::size_t b=0; b<l.blocks.size(); ++b)
                {
                    hsize_type patch_offset = 0;
                    hsize_type nElements_patch=1;
                    for(std::size_t d=0; d<Dim; ++d)        // for node-centered
                    {
                       nElements_patch *= l.blocks[b].extent()[d];
                    }
                    patch_offset = nElements_patch*num_components;
                    patch_offsets_vec.push_back(patch_offset);

                    dset_size+=nElements_patch*num_components;
                }
                std::cout<<"------> Final dataset size = "
                         <<dset_size<<std::endl;
       //     }

            // Create full empty dataset
//            auto space =_file->create_simple_space(1, &dset_size, NULL);
//            // ***TODO: Make collective
//            auto dset_id=_file->template create_dataset<value_type>(group_id,
//                "data:datatype=0", space);      //collective
            auto dset_id = _file->open_dataset(group_id, "data:datatype=0");

            std::vector<value_type> single_block_data;
            offset_type offset = 0;     // offset in dataset
            // ----------------------------------------------------------------
            // BLOCK ITERATOR
            for(std::size_t b=0; b<l.blocks.size(); ++b)
            {
                int rank = l.ranks[b];      // only contained on server
                single_block_data.clear();

                auto block = l.blocks[b];
                FieldData* field = l.fields[b];

                auto base=block.base();
                auto max=block.max();

                // ------------------------------------------------------------
                // COMPONENT ITERATOR
                field_type_iterator_t::for_types([&l, &lvl, &offset, &b,
                        &max, &base, &field,
                        &single_block_data, &num_components, &world]<typename T>()
                {
                    double field_value = 0.0;
                    if (world.rank()!=0)
                    {
                        // Loop through cells in block
                        for(auto k = base[2]; k<=max[2]; ++k)
                        {
                            for(auto j = base[1]; j<=max[1]; ++j)
                            {
                                for(auto i = base[0]; i<=max[0]; ++i)
                                {
                                    auto n = field->get(i,j,k);

                                    field_value = 0.0;
                                    if (std::abs(n.template get<T>())>=1e-32)
                                    {
                                        field_value=static_cast<value_type>(n.template get<T>());
                                    }
                                    // std::cout<<field_value<<" ";
//                                    std::cout<<field_value<<" "; //<<std::endl;
                                    // field_value=10+c;

                                    // Add data for this cell and component
                                    single_block_data.push_back(field_value);
                              //      dset_size += 1;     // increase by one per node
                                }
                            }
                        }
                    }
                });     // COMPONENT ITERATOR____________________________

                // Write single block data
                hsize_type block_data_size = single_block_data.size();
                hsize_type start = -1;
              //  std::cout<<"World rank: "<<world.rank()<<". Block rank: "<<rank<<std::endl;

                if (world.rank()!=0) {
   //                 std::cout<<"Size of offsets = "<<offsets_for_rank.size()<<std::endl;
              //      std::cout<<"Size of offsets[lvl] = "<<offsets_for_rank[lvl].size()<<std::endl;

                    start = offsets_for_rank[lvl][b];

                    std::cout<<"Lvl "<<lvl<<". b = "<<b
                            <<". Offset = "<<start
                            <<". Base = "<<base<<". Max = "<<max<<std::endl;

                    //std::cout<<"Rank is "<<world.rank()<<". b is "<<b<<std::endl;
                    //std::cout<<"Correct offset is "<<offset<<
                    //        ". offset from vector is "<<offsets_for_rank[lvl][b]<<std::endl;

                    // Write single block data
                    _file->template write<value_type>(dset_id, block_data_size, start,
                                                        &single_block_data[0]);

             //       _file->template write<value_type>(dset_id, block_data_size, offset,
               //                                         &single_block_data[0]);
                }



                hsize_type nElements_patch=1;
                for(std::size_t d=0; d<Dim; ++d)        // for node-centered
                {
                   nElements_patch *= l.blocks[b].extent()[d];
                }
                offset+=nElements_patch*num_components;        // update after each block/patch

            }       // BLOCK ITERATOR__________________________________________




            // Write Attributes -----------------------------------------------
//            // dt
//            _file->template create_attribute<int>(group_id,"dt",_dt);
//
//            // dx
//            value_type dx=1./(std::pow(2,lvl));   // dx = 1/(2^i)
//            _file->template create_attribute<value_type>(group_id,"dx",dx);
//
//            // ref_ratio
//            _file->template create_attribute<int>(group_id,
//                                                    "ref_ratio",_ref_ratio);
//
//            // time
//            _file->template create_attribute<int>(group_id,"time",_time);

//            _file->template create_attribute<std::string>(group_id_dattr,
//                    "objectType","CArrayBox");

            // prob_domain
//            auto max_cellCentered=l.probeDomain.max();
//            //Cell-centered data:
//            max_cellCentered-=1;
//            _file->template write_boxCompound<Dim>(group_id, "prob_domain",
//              l.probeDomain.min(), max_cellCentered);      // write prob_domain as attribute

            auto p= l.blocks.begin()->min();        // vector of ints size 3
            auto p2= l.blocks.begin()->max();
            std::vector<decltype(p)> mins;
            std::vector<decltype(p2)> maxs;

            for(std::size_t j=0;j<l.blocks.size();++j)
            {
                mins.push_back(l.blocks[j].min());
                maxs.push_back(l.blocks[j].max());
            }
//
//            // boxes.  write as dataset.
//            // 1 Create boxes
            hsize_type boxes_size = 0;
//
            if (world.rank()==0) {
                boxes_size = mins.size();
            }
//            boost::mpi::broadcast(world, boxes_size, 0); // send from server to all others
//
//            _file->template create_boxCompound<Dim>(group_id, "boxes", boxes_size, false);
//            //_file->template create_boxCompound<Dim>(group_id, "boxes", mins, maxs, false);


            // 2 Write boxes with just rank 0
            if (world.rank()==0) {
                _file->template open_write_boxCompound<Dim>(group_id, "boxes", mins, maxs, false);
            }
//            _file->template write_boxCompound<Dim>(group_id, "boxes", mins, maxs, false);

            // data attributes (open because already created)
            auto group_id_dattr=_file->create_group(group_id,
                    "data_attributes");

            // offsets (Written only by server)
            auto dset_Attr = _file->open_dataset(group_id_dattr, "offsets");
            if (world.rank()==0) {
                hsize_type size = patch_offsets_vec.size();
                _file->template write<int>(dset_Attr, size,
                                                        &patch_offsets_vec[0]);
            }
            _file->close_dset(dset_Attr);

            // Close spaces:
            //_file->close_space(space);
            _file->close_group(group_id_dattr);
            //_file->close_space(space_attr);
            _file->close_dset(dset_id);
            _file->close_group(group_id);

        }   // level iterator

        _file->close_group(root);

        std::cout<<"<--------Done writing level info with rank "<<world.rank()<<" of "<<world.size()<<std::endl;
    }



    void write_mesh(HDF5File* _file )
    {

        this->write_global_metaData(_file);
        this->write_level_info(_file);
    }

    void write_mesh(std::string _filename)
    {
        HDF5File f(_filename);
        this->write_global_metaData(&f);
        this->write_level_info(&f);
    }

    void write_mesh(std::string _filename,
                    std::vector<BlockDescriptor> _blocks)
    {
        level_map_.clear();
        this->init(_blocks);
        HDF5File f(_filename);
        this->write_global_metaData(&f);
        this->write_level_info(&f);
    }


    std::map<int,LevelInfo> level_map_;

};
}



#endif   //Chombo
