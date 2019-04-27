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


    using octant_type = typename Domain::octant_t;
    using write_info_t = typename std::pair<BlockDescriptor,FieldData*>;
    using block_info_t = typename std::tuple<int,BlockDescriptor,FieldData*>;
    using index_list_t = typename HDF5File::index_list_t;

    using key_type = typename Domain::key_t;
    using offset_type = int;
    using offset_vector = std::vector<offset_type>;

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
//        std::vector<int> ranks;
//        std::vector<BlockDescriptor> blocks;
        std::vector<FieldData*> fields;
        std::vector<octant_type*> octants;
        std::vector< std::vector<octant_type*> > octant_groups;

        bool operator<(const LevelInfo& _other) const
        {
            return level<_other.level;
        }
    };



    //Chombo(const std::vector<BlockDescriptor>& _blocks)
//    Chombo(const std::vector<write_info_t>& _data_blocks)
//    {
//        init(_data_blocks);
//    }

    Chombo(const std::vector<octant_type*>& _octant_blocks)
    {
        init(_octant_blocks);
    }

private:

    void init(const std::vector<octant_type*>& _octant_blocks)
    {
        boost::mpi::communicator world;

        //Blocks per level:
        for(auto& p: _octant_blocks)
        {

//            const int rank = std::get<0>(p);            // first is rank
//            const auto b = std::get<1>(p);              // second is descriptor
//            const auto field_data = std::get<2>(p);     // third is data
//            auto it=level_map_.find(b.level());     // returns iterator to element with key b.level()
//
            //const int rank = p->rank();            // first is rank
            const auto b = p->data()->descriptor();              // second is descriptor
            //const auto field_data = &(p->data()->node_field());     // third is data
            auto it=level_map_.find(b.level());     // returns iterator to element with key b.level()

            // FIXME: Some blocks have -1 level
            if (b.level() >= 0)
            {
                if(it!=level_map_.end())                // if found somewhere (does not return end())
                {
                    //it->second.ranks.push_back(rank);
                    //it->second.blocks.push_back(b);     // add block to corresponding level
                    //it->second.fields.push_back(field_data);

                    it->second.octants.push_back(p);

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
                    LevelInfo l;
                    l.level=b.level();
                    //l.ranks.push_back(rank);
                    //l.blocks.push_back(b);
                    //l.fields.push_back(field_data);
                    l.octants.push_back(p);
                    l.probeDomain=b;
                    level_map_.insert(std::make_pair(b.level(),l ));

                }
            }
        } // octant blocks
//


        // Once level_map_ is initialized, loop through and group blocks
        for (auto it=level_map_.begin(); it!=level_map_.end(); it++)
        {
            std::cout<<"Rank is "<<world.rank()<<" | Level is "<<it->first<<std::endl;

            auto& l=it->second;

            unsigned int track = 0;         // beginning of new group
            int group_no = -1;              // keep track of group number
            std::vector<octant_type*> v;

            // Loop through blocks in level map.
            for (unsigned int i=0; i!=l.octants.size(); ++i)
            {
                auto& block = l.octants[i];

                if (i==track)  // Beginning of new group
                {
                    l.octant_groups.push_back(v);
                    ++group_no;
                    l.octant_groups[group_no].push_back(block);
                    int group_rank = block->rank();

                    // Check how big group is need to add subsequent blocks to the group (how
                    // Currently just grouping if all children exist (group cubes).
                    unsigned int recurse = 0;
                    unsigned int recurse_lim = block->level(); // don't go beyond tree structure
                    auto* p = block;
                    while (i==track && recurse<recurse_lim) // only enters for first block in group
                    {
                        unsigned int shift = pow(pow(2,Dim), recurse+1)-1;
                        unsigned int shift_final = pow(pow(2,Dim), recurse);

                        if ( (p->key().child_number() == 0) &&
                                (i+shift <= l.octants.size()-1) )  // contained?
                        {
                            auto& block_end = l.octants[i+shift];

                            if (block->key()+shift==block_end->key() &&
                                    group_rank==block_end->rank())
                            {
                                ++recurse;
                                // Get parent to check bigger group.
                                p=p->parent();
                            }
                            else
                            {   track+=shift_final; }
                        }
                        else
                        {   track+=shift_final; }

                        if (track!=i)
                        {
                            std::cout<<std::endl;
                            std::cout<<"GROUP "<<group_no<<", rank = "<<group_rank<<", blocks = "<<i<<"-"<<track-1<<std::endl;
                            std::cout<<"   block "<<i<<std::endl;
                        }
                    }   // recursion
                }
                else if (i<track)   // add to existing group
                {
                    l.octant_groups[group_no].push_back(block);
                    std::cout<<"   block "<<i<<std::endl;
                }
            }





            // Print out structure of grouped blocks:
            std::cout<<"-------------------------------"<<std::endl;
            std::cout<<"PRINT STRUCTURE OF BLOCK GROUPS"<<std::endl;
            for (auto it=level_map_.begin(); it!=level_map_.end(); it++)
            {
                std::cout<<"Rank is "<<world.rank()<<" | Level is "<<it->first<<std::endl;
                auto& l=it->second;

                std::cout<<"There are "<<l.octant_groups.size()<<" groups."<<std::endl;
                int count = 0;
                //Loop over groups
                for (unsigned int j = 0; j < l.octant_groups.size(); ++j)
                {
                    //Loop over blocks in groups
                //    std::cout<<"    Group "<<j<<" of "<<l.octant_groups.size()<<" has size: "<<l.octant_groups[j].size()<<std::endl;

              //      std::cout<<"        Blocks :";
                    for (unsigned int k = 0; k < l.octant_groups[j].size(); ++k)
                    {
            //            std::cout<<" "<<count;
                        ++count;
                    }
                    std::cout<<std::endl;
                }
            }
        }
    } // init



public:

    void write_global_metaData( HDF5File* _file ,
                     value_type _dx=1,
                     value_type _time=0.0,
                     int _dt=1,
                     int _ref_ratio=2)
    {
        boost::mpi::communicator world;
        auto root = _file->get_root();

        // iterate through field to get field names
        std::vector<std::string> components;
        field_type_iterator_t::for_types([&components, &world]<typename T>()
        {
            std::string name = std::string(T::name());
            if (world.rank()==0) std::cout<<"          "<<name<<std::endl;
            components.push_back(name);
        });

        // rite components, number of components and number of levels
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

        world.barrier();
        boost::mpi::broadcast(world, num_levels, 0); // send from server
        _file->template create_attribute<int>(root, "num_levels",num_levels);

        // num_components
        const int num_components = components.size();
        _file->template create_attribute<int>(root, "num_components",
                static_cast<int>(num_components));

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
        for (int lvl = 0; lvl < num_levels; ++lvl) {
            auto group_id_lvl = _file->create_group(root,"level_"+std::to_string(lvl));

            // Write Attributes -----------------------------------------------
            // dt
            _file->template create_attribute<int>(group_id_lvl,"dt",_dt);

            // dx
            value_type dx=_dx/(std::pow(2,lvl));   // dx = 1/(2^i)
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
                min_cellCentered=l.probeDomain.min();
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
            _file->template write_boxCompound<Dim>(group_id_lvl, "prob_domain",
              min_cellCentered, max_cellCentered);      // write prob_domain as attribute

            // Create "boxes" dataset: ****************************************
            // TODO Boxes size for groups
            hsize_type boxes_size = 0;

            // Get size of boxes on server
            if (world.rank()==0) {
                auto it_lvl = level_map_.find(lvl);
                auto l = it_lvl->second;

                boxes_size = l.octant_groups.size();    // group
                //boxes_size = l.octants.size();          // normal
            }
            boost::mpi::broadcast(world, boxes_size, 0); // send from server to all others

            _file->template create_boxCompound<Dim>(group_id_lvl, "boxes", boxes_size, false);


            // Create dataset for data and "offsets" **************************
            // Server gets size:
            hsize_type offsets_size = 0;
            hsize_type dset_size = 0;   // Count size of dataset for this level

            // Get dataset size with octant_groups
            if (world.rank()==0) {
                // Write DATA for different components-----------------------------
                auto it_lvl = level_map_.find(lvl);
                auto l = it_lvl->second;

                // Determine size of dataset for this level to create dataset
                for(std::size_t g=0; g<l.octant_groups.size(); ++g)
                {

                    for (std::size_t b=0; b<l.octant_groups[g].size(); ++b)
                    {
                        hsize_type nElements_patch=1;
                        const auto& p = l.octant_groups[g][b];
                        const auto& block_desc = p->data()->descriptor();
                        for(std::size_t d=0; d<Dim; ++d)        // for node-centered
                        {
                            nElements_patch *= block_desc.extent()[d];
                        }
                        dset_size+=nElements_patch*num_components;
                    }
                }
                offsets_size = l.octant_groups.size()+1;

                std::cout<<"1: dset_size = "<<dset_size
                     <<". offsets_size = "<<offsets_size<<std::endl;
            }

//            // Normal version:
//            offsets_size = 0;
//            dset_size = 0;   // Count size of dataset for this level
//
//            if (world.rank()==0) {
//                // Write DATA for different components-----------------------------
//                auto it_lvl = level_map_.find(lvl);
//                auto l = it_lvl->second;
//
//                // Determine size of dataset for this level to create dataset
//                for(std::size_t b=0; b<l.octants.size(); ++b)
//                {
//                    hsize_type nElements_patch=1;
//                    const auto& p = l.octants[b];
//                    const auto& block_desc = p->data()->descriptor();
//                    for(std::size_t d=0; d<Dim; ++d)        // for node-centered
//                    {
//                        nElements_patch *= block_desc.extent()[d];
//                    }
//                    dset_size+=nElements_patch*num_components;
//                }
//                offsets_size = l.octants.size()+1;
//
//                std::cout<<"2: dset_size = "<<dset_size
//                     <<". offsets_size = "<<offsets_size<<std::endl;
//            }

            // Scatter size of "offsets"
            boost::mpi::broadcast(world, offsets_size, 0); // send from server to all others

            // Create "offsets" dataset
            // TODO Use number of groups instead of offsets size
            auto space_attr =_file->create_simple_space(1, &offsets_size, NULL);
            auto dset_Attr=_file->template create_dataset<int>(group_id_dattr,
                        "offsets",space_attr);
            _file->close_dset(dset_Attr);
            _file->close_group(group_id_dattr);


            // send size to all clients
            world.barrier();
            boost::mpi::broadcast(world, dset_size, 0); // send from server to all others

            // Create full empty dataset with all processes
            auto space =_file->create_simple_space(1, &dset_size, NULL);
            /*auto dset_id=*/_file->template create_dataset<value_type>(group_id_lvl,
                "data:datatype=0", space);      //collective
            _file->close_space(space);

        }
        _file->close_group(root);
    } // write_global_metaData_________________________________________________



    void write_level_info(HDF5File* _file,
                     value_type _time=0.0,
                     int _dt=1,
                     int _ref_ratio=2)
    {
        boost::mpi::communicator world;

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

        // TODO: Calculate offsets differently for groups of blocks
        if (world.rank()==0) {
            // Loop initially to get offsets from rank==0
            for(auto& lp : level_map_)
            {
                auto l = lp.second;     // LevelInfo
                int lvl = l.level;
             //   offset_vector block_offsets;
                offset_type offset = 0;             // offset in dataset

                // Calculate offsets by group
                for(std::size_t g=0; g<l.octant_groups.size(); ++g)
                {
                    // int rank = l.ranks[b];

                    for (std::size_t b=0; b<l.octant_groups[g].size(); ++b)
                    {
                        const auto& octant = l.octant_groups[g][b];
                        int rank = octant->rank();
                        // Loop through components in block
                        hsize_type nElements_patch=1;
                        const auto& block_desc = octant->data()->descriptor();
                        for(std::size_t d=0; d<Dim; ++d)        // for node-centered
                        {
                            nElements_patch *= block_desc.extent()[d];
                        }

                        int data_block_size=nElements_patch*num_components;

                        if (b==0) {
                            offset_vector[rank][lvl].push_back(offset);
                        }
                        offset+=data_block_size;
                    }
                }

//                // NORMAL -----------------------------------------------------
//                // Loop through blocks and
//                // Determine size of dataset for this level to create dataset
//                for(std::size_t b=0; b<l.octants.size(); ++b)
//                {
//                    // int rank = l.ranks[b];
//                    const auto& p = l.octants[b];
//                    int rank = p->rank();
//
//                    // Loop through components in block
//                    hsize_type nElements_patch=1;
//                    const auto& block_desc = p->data()->descriptor();
//                    for(std::size_t d=0; d<Dim; ++d)        // for node-centered
//                    {
//                        nElements_patch *= block_desc.extent()[d];
//                    }
//
//                    int data_block_size=nElements_patch*num_components;
//
//                    offset_vector[rank][lvl].push_back(offset);
//                    offset+=data_block_size;
//                }
            }
        }


        std::vector<std::vector<offset_type>> offsets_for_rank;
        boost::mpi::scatter(world, offset_vector, offsets_for_rank, 0);

        world.barrier();


        // Parallel Write -----------------------------------------------------
        // Loop over levels and write patches
        for(auto& lp : level_map_)
        {
            // lp iterates over each pair of <int, LevelInfo> in the full map
            auto l = lp.second;     // LevelInfo
            int lvl = l.level;

            // Write attributes for this lvl (e.g. dx, dt)
            auto group_id=_file->create_group(root,"level_"+std::to_string(lvl));


            /*****************************************************************/
            // Write level data

            // Write "offsets" ------------------------------------------------

            // Group version:
            std::vector<int> patch_offsets_vec;
            patch_offsets_vec.push_back(0);

            // Gather patch_offsets for each group
            for(std::size_t g=0; g<l.octant_groups.size(); ++g)
            {
                hsize_type patch_offset = 0;
                for(std::size_t b=0; b<l.octant_groups[g].size(); ++b)
                {
                    hsize_type nElements_patch=1;
                    const auto& p = l.octants[b];
                    const auto& block_desc = p->data()->descriptor();

                    for(std::size_t d=0; d<Dim; ++d)        // for node-centered
                    {
                        nElements_patch *= block_desc.extent()[d];
                    }
                    patch_offset += nElements_patch*num_components;
                }
                patch_offsets_vec.push_back(patch_offset);
                std::cout<<"Group "<<g<<". size = "<<l.octant_groups[g].size()
                         <<". patch_offset = "<<patch_offset<<std::endl;
            }


//            // Normal version*************************
//            patch_offsets_vec.clear();
//            patch_offsets_vec.push_back(0);
//
//            // Gather patch_offsets for each group
//            for(std::size_t b=0; b<l.octants.size(); ++b)
//            {
//                hsize_type patch_offset = 0;
//                hsize_type nElements_patch=1;
//                const auto& p = l.octants[b];
//                const auto& block_desc = p->data()->descriptor();
//
//                for(std::size_t d=0; d<Dim; ++d)        // for node-centered
//                {
//                    nElements_patch *= block_desc.extent()[d];
//                }
//                patch_offset = nElements_patch*num_components;
//                patch_offsets_vec.push_back(patch_offset);
//            }
//
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
            _file->close_group(group_id_dattr);


            // ----------------------------------------------------------------
            // Write DATA for different components-----------------------------
            auto dset_id = _file->open_dataset(group_id, "data:datatype=0");

            std::vector<value_type> single_block_data;
           // offset_type offset = 0;     // offset in dataset

            // GROUP ITERATOR
            for(std::size_t g=0; g<l.octant_groups.size(); ++g)
            {
                single_block_data.clear();
                const auto& octant0 = l.octant_groups[g][0];
                const auto& block_desc0 = octant0->data()->descriptor();

                auto base0=block_desc0.base();
                //auto max0=block_desc0.max();
                auto block_extent=block_desc0.extent();
                // for cubic blocks and cubic extents
                int group_extent = std::cbrt(l.octant_groups[g].size());

                std::cout<<"--- Group"<<g<<std::endl;
             //   std::cout<<"Group Base = "<<base0<<". Max = "<<max0
             //               <<". Group extent = "<<group_extent<<std::endl;


                // Order octants:
                std::vector<octant_type*> ordered_group(l.octant_groups[g].size());
                // BLOCK ITERATOR
                for(std::size_t b=0; b<l.octant_groups[g].size(); ++b)
                {
                    const auto& p = l.octant_groups[g][b];
                    const auto& block_desc = p->data()->descriptor();

                    // Get group coordinate based on shift
                    auto base_shift0 = block_desc.base()-base0;

                    // TODO Order and Determine block indices
                    // Order: Get flattened index and insert
                    // coord = k*maxI*maxJ + j*maxI + I
                    int group_coord = base_shift0[2]/block_extent[2]*group_extent*group_extent + base_shift0[1]/block_extent[1]*group_extent + base_shift0[0]/block_extent[0];

          //          std::cout<<"       Base_coord = "<<base_shift0/2
            //            <<"-->    Group coord   = "<<group_coord<<std::endl;

                    // Insert into vector:
                    ordered_group[group_coord] = p;
                }


                // Check order:
               // std::cout<<"   Ordered:"<<std::endl;
                for (std::size_t b=0; b<ordered_group.size(); ++b)
                {
                    //const auto& p = ordered_group[b];
                    //const auto& block_desc = p->data()->descriptor();
                    //auto base_shift0 = block_desc.base()-base0;

                    //int group_coord = base_shift0[2]/block_extent[2]*group_extent*group_extent + base_shift0[1]/block_extent[1]*group_extent + base_shift0[0]/block_extent[0];
        //            std::cout<<"       Base_coord = "<<base_shift0/2
        //                <<"-->    Group coord   = "<<group_coord<<std::endl;
                }



                // Iterate and print
                // Assumes all blocks have same extents

                // ------------------------------------------------------------
                // COMPONENT ITERATOR
                field_type_iterator_t::for_types([&single_block_data, &world,
                            &block_extent, &group_extent, &ordered_group]<typename T>()
                {
//                    std::cout<<"New field: "<<std::endl;
                    double field_value = 0.0;
                    if (world.rank()!=0)
                    {

                        // TODO Double loop: cells in block and blocks in group
                        for(auto z=0; z<group_extent; ++z)
                        {
                            // base and max should be based on 0 (block_extent)
                            for(auto k=0; k<block_extent[2]; ++k)
                            {
                                for(auto y=0; y<group_extent; ++y)
                                {
                                    for(auto j=0; j<block_extent[1]; ++j)
                                    {
                                        for(auto x=0; x<group_extent; ++x)
                                        {

                                            int group_coord = z*group_extent*group_extent + y*group_extent + x;
                                            const auto& octant = ordered_group[group_coord];
                                            const auto& block_desc = octant->data()->descriptor();
                                            const auto& field = &(octant->data()->node_field());     // third is data
                                            auto base=block_desc.base();

                                            for(auto i=0; i<block_extent[0]; ++i)
                                            {
                                                auto n = field->get(i+base[0],j+base[1],k+base[2]);

                                                field_value = 0.0;
                                                if (std::abs(n.template get<T>())>=1e-32)
                                                {
                                                    field_value=static_cast<value_type>(n.template get<T>());
                                                }
                                          //      std::cout<<" "<<field_value;
                                                single_block_data.push_back(field_value);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    std::cout<<std::endl;
                });     // COMPONENT ITERATOR____________________________


               // std::cout<<" DATA WRITTEN for Group: "<<g<<"-------------"<<std::endl;
               // for (std::size_t f=0; f<single_block_data.size(); ++f)
               // {
               //     std::cout<<" "<<single_block_data[f];
               // }
               // std::cout<<std::endl;

                // Write single block data
                hsize_type block_data_size = single_block_data.size();
                hsize_type start = -1;

                if (world.rank()!=0) {
                    start = offsets_for_rank[lvl][g];
               //     std::cout<<"block_data_size = "<<block_data_size<<std::endl;
               //     std::cout<<"start = "<<start<<std::endl;

                    _file->template write<value_type>(dset_id, block_data_size, start,
                            &single_block_data[0]);

                }

                hsize_type nElements_patch=1;
                for(std::size_t d=0; d<Dim; ++d)        // for node-centered
                {
                    nElements_patch *= block_desc0.extent()[d];
                }
            //    offset+=nElements_patch*num_components*l.octant_groups[g].size();        // update after each block/patch
              //  std::cout<<"offset = "<<offset<<std::endl;

            } // GROUP ITERATOR






//            // NORMAL -----------------------------------------------------
//            // Write DATA for different components-----------------------------
//            //auto dset_id = _file->open_dataset(group_id, "data:datatype=0");
//
//            // std::vector<value_type> single_block_data;
//            offset = 0;     // offset in dataset
//            // ----------------------------------------------------------------
//            // BLOCK ITERATOR
//            for(std::size_t b=0; b<l.octants.size(); ++b)
//            {
//                //int rank = l.ranks[b];      // only contained on server
//                single_block_data.clear();
//
//                const auto& p = l.octants[b];
//                const auto& block_desc = p->data()->descriptor();
//                const auto& field = &(p->data()->node_field());     // third is data
//
//                auto base=block_desc.base();
//                auto max=block_desc.max();
//
//                // ------------------------------------------------------------
//                // COMPONENT ITERATOR
//                field_type_iterator_t::for_types([&l, &lvl, &offset, &b,
//                        &max, &base, &field,
//                        &single_block_data, &num_components, &world]<typename T>()
//                {
//                    double field_value = 0.0;
//                    if (world.rank()!=0)
//                    {
//                        // Loop through cells in block
//                        for(auto k = base[2]; k<=max[2]; ++k)
//                        {
//                            for(auto j = base[1]; j<=max[1]; ++j)
//                            {
//                                for(auto i = base[0]; i<=max[0]; ++i)
//                                {
//                                    auto n = field->get(i,j,k);
//
//                                    field_value = 0.0;
//                                    if (std::abs(n.template get<T>())>=1e-32)
//                                    {
//                                        field_value=static_cast<value_type>(n.template get<T>());
//                                    }
//                                    single_block_data.push_back(field_value);
//                                }
//                            }
//                        }
//                    }
//                });     // COMPONENT ITERATOR____________________________
//
//                // Write single block data
//                hsize_type block_data_size = single_block_data.size();
//                hsize_type start = -1;
//
//                if (world.rank()!=0) {
//                    start = offsets_for_rank[lvl][b];
//
//                    _file->template write<value_type>(dset_id, block_data_size, start,
//                                                        &single_block_data[0]);
//
//                }
//
//                hsize_type nElements_patch=1;
//                for(std::size_t d=0; d<Dim; ++d)        // for node-centered
//                {
//                   nElements_patch *= block_desc.extent()[d];
//                }
//                offset+=nElements_patch*num_components;        // update after each block/patch
//                std::cout<<" normal offset = "<<offset<<std::endl;
//
//            }       // BLOCK ITERATOR__________________________________________



            // Write "boxes" --------------------------------------------------
            if (world.rank() == 0)
            {
                // Determine and write "boxes"
                auto p = l.octant_groups[0][0];
                auto& block_desc = p->data()->descriptor();

                auto pmin= block_desc.min();        // vector of ints size 3
                auto pmax= block_desc.max();
                std::vector<decltype(pmin)> mins(l.octant_groups.size());
                std::vector<decltype(pmax)> maxs(l.octant_groups.size());
                decltype(pmin) mins_temp;
                decltype(pmax) maxs_temp;

                // TODO: calculate boxes for each group
                for(std::size_t g=0; g<l.octant_groups.size(); ++g)
                {
                //    std::cout<<"New group ------"<<std::endl;
                    for(std::size_t b=0; b<l.octant_groups[g].size(); ++b)
                    {
                        const auto p = l.octant_groups[g][b];
                        const auto& block_desc = p->data()->descriptor();
                        if (b==0)
                        {
                            mins_temp = block_desc.min();
                            maxs_temp = block_desc.max();
                        }
                        else
                        {
                            for (std::size_t d=0; d<Dim; ++d)
                            {
                                if (mins_temp[d] > block_desc.min()[d]) {
                                    mins_temp[d] = block_desc.min()[d];
                                }
                                if (maxs_temp[d] < block_desc.max()[d]) {
                                    maxs_temp[d] = block_desc.max()[d];
                                }
                            }
                        }
                //        std::cout<<"Min = "<<mins_temp
                //            <<", block_desc.min() = "<<block_desc.min()<<std::endl;
                //        std::cout<<"Max = "<<maxs_temp
                //            <<", block_desc.max() = "<<block_desc.max()<<std::endl;
                    }
                    mins[g] = mins_temp;
                    maxs[g] = maxs_temp;
                }

                //            // normal
                //            mins.clear();
                //            maxs.clear();
                //            for(std::size_t b=0; b<l.octants.size(); ++b)
                //            {
//                const auto& p = l.octants[b];
//                const auto& block_desc = p->data()->descriptor();
//                mins.push_back(block_desc.min());
//                maxs.push_back(block_desc.max());
//            }
//

            // 2 Write boxes with just rank 0
            //if (world.rank()==0) {
                _file->template open_write_boxCompound<Dim>(group_id, "boxes", mins, maxs, false);
            }



            // Close spaces:
            //_file->close_space(space);
            //_file->close_group(group_id_dattr);
            //_file->close_space(space_attr);
            _file->close_dset(dset_id);
            _file->close_group(group_id);

        }   // level iterator

        _file->close_group(root);

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

//    void write_mesh(std::string _filename,
//                    std::vector<BlockDescriptor> _blocks)
//    {
//        level_map_.clear();
//        this->init(_blocks);
//        HDF5File f(_filename);
//        this->write_global_metaData(&f);
//        this->write_level_info(&f);
//    }
//

    std::map<int,LevelInfo> level_map_;

};
}



#endif   //Chombo
