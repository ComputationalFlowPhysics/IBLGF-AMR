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
#include <domain/dataFields/blockDescriptor.hpp>
#include <global.hpp>
#include <setups/tests/decomposition/decomposition_test.hpp>

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

private:

    //void init(const std::vector<BlockDescriptor>& _blocks)
    void init(const std::vector<write_info_t>& _data_blocks)
    {

        //Blocks per level:
        for(auto& p: _data_blocks)
        {

            const auto b = p.first;
            //const auto b = &(p.first);
            const auto field_data = p.second;
            printf("Adding block \n");


            auto it=level_map_.find(b.level());     // returns iterator to element with key b.level()
            if(it!=level_map_.end())                // if found somewhere (does not return end())
            {
                printf("Current min = %d, %d, %d \n",
                    it->second.probeDomain.min()[0],it->second.probeDomain.min()[1],
                    it->second.probeDomain.min()[2]);
                printf("Current max = %d, %d, %d \n",
                    it->second.probeDomain.max()[0],it->second.probeDomain.max()[1],
                    it->second.probeDomain.max()[2]);

              //  printf("Add to existing level \n");
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
                printf("New min = %d, %d, %d \n",
                    it->second.probeDomain.min()[0],it->second.probeDomain.min()[1],
                    it->second.probeDomain.min()[2]);
                printf("New max = %d, %d, %d \n",
                    it->second.probeDomain.max()[0],it->second.probeDomain.max()[1],
                    it->second.probeDomain.max()[2]);
            }
            else        // if not found, simply create that level and insert the block
            {
                printf("Adding new level \n");
                LevelInfo l;
                l.level=b.level();
                l.blocks.push_back(b);
                l.fields.push_back(field_data);
                l.probeDomain=b;
                level_map_.insert(std::make_pair(b.level(),l ));

                printf("Level base = %d, %d, %d \n",
                        l.probeDomain.base()[0],l.probeDomain.base()[1],
                        l.probeDomain.base()[2]);
                printf("Level extent = %d, %d, %d \n",
                        l.probeDomain.extent()[0],l.probeDomain.extent()[1],
                        l.probeDomain.extent()[2]);
            }

            printf("Done Adding block \n");

        }
    }

public:

    void write_global_metaData( HDF5File* _file )
    {
        std::cout<<"Write global meta data"<<std::endl;
        auto root= _file->get_root();

        // get field names
        std::vector<std::string> components;
        field_type_iterator_t::for_types([&components]<typename T>()
        {
            std::string name = std::string(T::name());
            std::cout<<name<<std::endl;
            components.push_back(name);
        });

        const int num_components = components.size();
        auto num_levels=level_map_.size();
        std::cout<<"num_levels:"<<num_levels<<std::endl;
        std::cout<<"num_components:"<< num_components <<std::endl;

        for(std::size_t i=0; i<components.size(); ++i)  // write component names
        {
            _file->template create_attribute<std::string>(root,
                "component_"+std::to_string(i),components[i]);
        }

        _file->template create_attribute<int>(root,
                "num_levels",num_levels);
        _file->template create_attribute<int>(root,
                "num_components",static_cast<int>(num_components));

        // Ordering of the dataspaces
        _file->template create_attribute<std::string>(root,
                "filetype","VanillaAMRFileType");
        // Node-centering=7, zone-centering=6
        _file->template create_attribute<int>(root,"data_centering",6);

        auto global_id=_file->create_group(root,"Chombo_global");
        _file->template create_attribute<int>(global_id,
                "SpaceDim", static_cast<int>(Dim));
        _file->close_group(global_id);
        _file->close_group(root);
    }

    //TODO: write actual data if needed
    //This writes only level info without the data
    void write_level_info(HDF5File* _file,
                     value_type _time=0.,
                     int _dt=1,
                     int _ref_ratio=2)
    {
        boost::mpi::communicator world;
        int mpi_size = world.size();
        int mpi_rank = world.rank();

        std::cout<<"-------->Write level info with rank "<<mpi_rank<<" of "<<mpi_size<<std::endl;


        //using hsize_type =H5size_type;
        auto root = _file->get_root();

        // get field names
        std::vector<std::string> components;
        field_type_iterator_t::for_types([&components]<typename T>()
        {
            std::string name = std::string(T::name());
            std::cout<<name<<std::endl;
            components.push_back(name);
        });
        const int num_components = components.size();

        //Loop over levels and write patches
        // level_map_ is map of level (int) and LevelInfo (per Level)
        // (probeDomain, level, vector of blocks)
        for(auto& lp : level_map_)
        {
            // lp iterates over each pair of <int, LevelInfo> in the full map
            auto l = lp.second;     // LevelInfo
            int lvl = l.level;

            auto group_id=_file->create_group(root,"level_"+std::to_string(lvl));
            _file->template create_attribute<int>(group_id,"dt",_dt);

            value_type dx=1./(std::pow(2,lvl));   // dx = 1/(2^i)
            _file->template create_attribute<value_type>(group_id,"dx",dx);


            auto max_cellCentered=l.probeDomain.max();
            //Cell-centered data:
            //max_cellCentered-=1;

            _file->template write_boxCompound<Dim>(group_id, "prob_domain",
              l.probeDomain.min(), max_cellCentered);      // write prob_domain as attribute

            _file->template create_attribute<int>(group_id,
                "ref_ratio",_ref_ratio);
            _file->template create_attribute<int>(group_id,"time",_time);


            auto p= l.blocks.begin()->min();        // vector of ints size 3
            auto p2= l.blocks.begin()->max();
            std::vector<decltype(p)> mins;
            std::vector<decltype(p2)> maxs;

            for(std::size_t j=0;j<l.blocks.size();++j)
            {
                mins.push_back(l.blocks[j].min());
                maxs.push_back(l.blocks[j].max());
            }

            _file->template write_boxCompound<Dim>(group_id, "boxes", mins, maxs, false);


            /*****************************************************************/
            //Level-patches

            std::vector<int> patch_offsets;
            patch_offsets.push_back(0);
            hsize_type accumulated_patchCount=0;


            std::vector<value_type> component_data;
            std::vector<value_type> single_comp_block;


            // Write DATA for different components-----------------------------
            printf("Start level %d \n",lvl);

            hsize_type dset_size = 0;   // Count size of dataset for this level
            int offset = 0;             // offset in dataset


            // Loop through blocks in level
            for(std::size_t b=0; b<l.blocks.size(); ++b)
            {
                std::cout<<"-- | Write for block "<<b<<std::endl;
                int c=0;

                field_type_iterator_t::for_types([&l, &c, &lvl, &patch_offsets,
                        &dset_size, &offset, &b,
                        &accumulated_patchCount, &component_data,
                        &single_comp_block, &num_components, &world]<typename T>()
                {
                    std::cout<<"Write for field: "<<std::string(T::name())<<std::endl;

                    hsize_type offset_acc=0;
                    hsize_type nElements_patch=1;

                    auto block = l.blocks[b];
                    FieldData* field = l.fields[b];

//                    printf("Block %d of %d \n",b, l.blocks.size());
                    nElements_patch=1;
                    for(std::size_t d=0; d<Dim; ++d)        // for node-centered
                    {
                       nElements_patch *= l.blocks[b].extent()[d];
                    }

                    accumulated_patchCount+=nElements_patch;
                    if(c==0)
                    {
                        offset_acc += nElements_patch*num_components;
                        patch_offsets.push_back(offset_acc);
                    }

                    // loop through cells in block
                    auto base=block.base();
                    auto max=block.max();

                    std::cout<<" block number = "<<b<<", component number = "<<c<< std::endl;
                    // iterate over the points in the block
                    // subtract 1 from each max because blocks are grown.
                    double field_value = 0.0;
                    single_comp_block.clear();
                    for(auto k =base[2];k<=max[2];++k)
                    {
                        for(auto j =base[1];j<=max[1];++j)
                        {
                            for(auto i =base[0];i<=max[0];++i)
                            {
                                if (world.rank()==1)
                                {
                                    auto n = field->get(i,j,k);
                                //    std::cout<< n.global_coordinate()<<std::endl;

                                    field_value = 0.0;
                                    if (std::abs(n.template get<T>())>=1e-32)
                                    {
                                        field_value=static_cast<value_type>(n.template get<T>());
                                    }

                                   // field_value=10+c;

                                //    std::cout<<" Value = "<<field_value<<std::endl;
                                }

                                single_comp_block.push_back(field_value);
                                dset_size += 1;     // increase by one per node

                            }
                        }
                    }

                   // double field_value =  100*(i+1)+10*(b+1)+c+1;
                    //single_comp_block.assign(nElements_patch, field_value);
                  //  single_comp_block.push_back(field_value);

                    printf("Single vector is: \n");
                    for(int count = 0; count<single_comp_block.size(); ++count)
                    {
                        std::cout<<single_comp_block[count]<<" ";
                    }
                    std::cout<<std::endl;

                    // Add to end of long vector, maybe better way to do this
                    component_data.insert(component_data.end(),
                        std::begin(single_comp_block), std::end(single_comp_block));

                    // Write data for one component and one block
                    std::cout<<"Offset is "<<offset<<std::endl;
                    offset+=nElements_patch;

                    ++c;
                });
                // Write per block

            }

          //  std::cout << "CELL_DATA " << nCells << std::endl;

//            field_type_iterator_t::for_types([&l]<typename T>()
//            {
//                std::string name = "cell_data_"+std::string(T::name());
//                std::cout<<name<<std::endl;
//            });


            // accumulated_patchCount*=num_components;
            // create simple 1D dataspace
            hsize_type data_size = component_data.size();

            auto space =_file->create_simple_space(1, &data_size, NULL);

            auto dset_id=_file->template create_dataset<value_type>(group_id,
                "data:datatype=0", space);      //collective

            //auto hyperslab = H5Screate_simple(1, &data_size, NULL);
        //    auto status =
            hsize_t start  = 0;
            hsize_t stride = 1;
            hsize_t count  = data_size;
            hsize_t block  = 1;


            // Write dataset
            std::cout<<"Write component data to dataset called: data:datatype=0 with rank "<<world.rank()<<" of "<<world.size()<<std::endl;

        //    _file->template write<value_type>(dset_id, data_size,
          //                                              &component_data[0]);

            _file->template write<value_type>(dset_id, data_size, start,
                                                        &component_data[0]);

            // Close spaces:
            _file->close_space(space);

            auto group_id_dattr=_file->create_group(group_id,
                    "data_attributes");
            _file->template create_attribute<std::string>(group_id_dattr,
                    "objectType","CArrayBox");
            hsize_type size=patch_offsets.size();

            auto space_attr =_file->create_simple_space(1, &size, NULL);
            auto dset_Attr=_file->template create_dataset<int>(group_id_dattr,
                    "offsets",space_attr);
            _file->template write<int>(dset_Attr,size, &patch_offsets[0]);

            _file->close_group(group_id_dattr);
            _file->close_space(space_attr);
            _file->close_dset(dset_Attr);
            _file->close_dset(dset_id);
            _file->close_group(group_id);

        }

        _file->close_group(root);

        std::cout<<"<--------Done writing level info with rank "<<mpi_rank<<" of "<<mpi_size<<std::endl;
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
