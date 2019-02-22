#ifndef INCLUDED_H5_IO2_HPP
#define INCLUDED_H5_IO2_HPP


#include <ostream>
#include <fstream>
#include <iostream>
#include <sstream>

// IBLGF-specific
#include <global.hpp>
#include <setups/tests/decomposition/decomposition_test.hpp>
#include <domain/dataFields/blockDescriptor.hpp>
#include <IO/chombo/chombo.hpp>
#include <IO/chombo/h5_file.hpp>


namespace io
{



template<int Dim, class Domain>
class H5_io2
{

public:

    // static constexpr int Dim = Domain::dimension;
    using coordinate_type        = typename Domain::coordinate_type;
    using datablock_t            = typename Domain::datablock_t;
    using field_type_iterator_t  = typename Domain::field_type_iterator_t;
    using field_tuple            = typename datablock_t::fields_tuple_t;

    template<typename T, std::size_t D>
    using vector_t=math::vector<T, D>;

    //const int Dim = 3;
    using base_t = vector_t<int,Dim>;
    using extent_t = vector_t<int,Dim>;
    using block_t = typename domain::BlockDescriptor<int, Dim>;
    using chombo_t= typename chombo_writer::Chombo<Dim, block_t,
                                        hdf5_file<Dim, base_t> >;
public:

    void write_h5(std::string _filename, Domain& _lt )
    {
        boost::mpi::communicator world;

        std::cout<<std::endl<<"----->write_h5: Start, with rank "<<world.rank()<<" of "<<world.size()<<std::endl;

        //std::ofstream ofs(_filename );``:w
        //
//        std::cout<<"# h5 DataFile Version ***"<<std::endl;
//        std::cout<<"HDF5 output"<<std::endl;
//        std::cout<<"ASCII"<<std::endl;
//        std::cout<<"DATASET STRUCTURED_GRID"<<std::endl;
        int nPoints=0;

        std::vector<block_t> write_blocks;    // vector of BlockDescriptors

        for(auto it=_lt.begin_leafs();it!=_lt.end_leafs();++it)
        {
            auto b= it->data()->descriptor();
            b.grow(0,1); //grow by one to fill the gap
            nPoints+= b.nPoints();
        }

        std::cout<<"------------POINTS "<<nPoints<<" float"<<std::endl;
        int nCells=0;
        int _count=0;
        for(auto it=_lt.begin_leafs();it!=_lt.end_leafs();++it)
        {
            auto block =it->data()->descriptor();
            int e = block.nPoints();
            nCells+=e;
            block.grow(0,1);

            auto base=block.base();
            auto max=block.max();

            // Add block to vector
            write_blocks.push_back(block);

            for(auto k =base[2];k<=max[2];++k)
            {
                for(auto j =base[1];j<=max[1];++j)
                {
                    for(auto i =base[0];i<=max[0];++i)
                    {
                        if (_lt.is_client())
                        {
                            auto n=it->data()->node(i,j,k);
          //                  std::cout<<n.global_coordinate()<<std::endl;
                            auto it2 = it->data()->nodes_domain().begin();
                          //  auto source_val = it2->get<source>();
             //               std::cout<<"Print source :"<<source_val<<std::endl;
                        }
                    }
                }
            }
            it->index(_count*block.nPoints());
            ++_count;
        }


        hdf5_file<Dim> chombo_file(_filename);
        chombo_t ch_writer(write_blocks);     // Initialize writer with vector of blocks
        ch_writer.write_global_metaData(&chombo_file);
        ch_writer.write_level_info(&chombo_file);

        std::cout<<"<-----write_h5: End, with rank "<<world.rank()<<" of "<<world.size()<<std::endl;

        std::cout<<"\n CELLS "<<nCells<<" "<<nCells*9<<std::endl;
//
//        //connectivity
//        for (auto it = _lt.begin_leafs(); it != _lt.end_leafs(); ++it)
//        {
//
//            auto block =it->data()->descriptor();
//            auto base=block.base();
//            auto max=block.max();
//
//            auto index_block= block;
//            index_block.grow(0,1);
//            for(auto k =base[2];k<=max[2];++k)
//            {
//                for(auto j =base[1];j<=max[1];++j)
//                {
//                    for(auto i =base[0];i<=max[0];++i)
//                    {
//                        //if (!n.on_max_blockBorder() )
//                        //if (true)
//                        ofs << "8 ";
//                        for (int k2=0;k2<2;++k2)
//                        {
//                            for (int j2=0;j2<2;++j2)
//                            {
//                                for (int i2=0;i2<2;++i2)
//                                {
//                                    auto nn_idx=index_block.get_flat_index(
//                                            coordinate_type({i+i2,j+j2,k+k2}));
//
//                                        ofs<< nn_idx+it->index()<<" ";
//                                }
//                            }
//                        }
//                        ofs<<std::endl;
//                    }
//                }
//            }
//        }
//        ofs << "\nCELL_TYPES " << nCells << std::endl;
//        for (int i = 0; i < nCells; ++i) ofs << "11" << std::endl;
//
//        //ofs << "POINT_DATA " << nPoints << std::endl;
//        //field_type_iterator_t::for_types([&_lt, &ofs, &nCells]<typename T>()
//        //{
//        //    std::string name = "vertex_data_"+std::string(T::name());
//        //    ofs << "SCALARS " << name << " float " << std::endl;
//        //    ofs << "LOOKUP_TABLE default"          << std::endl;
//        //    for (auto it = _lt.begin_leafs(); it != _lt.end_leafs(); ++it)
//        //    {
//        //            for (auto& n : it->data()->nodes())
//        //            {
//        //                ofs << n.template get<T>() << std::endl;
//        //            }
//        //    }
//        //});
//        ofs << "CELL_DATA " << nCells << std::endl;

        // Iterate through field types
        field_type_iterator_t::for_types([&_lt, &nCells]<typename T>()
        {
            std::string name = "cell_data_"+std::string(T::name());
            std::cout<< "COMPONENT DATA : " << name << std::endl;
          //  ofs << "LOOKUP_TABLE default"          << std::endl;
            for (auto it = _lt.begin_leafs(); it != _lt.end_leafs(); ++it)
            {
                auto block =it->data()->descriptor();
                auto base=block.base();
                auto max=block.max();

                auto index_block= block;
                index_block.grow(0,1);

                // Iterate over points
                for(auto k =base[2];k<=max[2];++k)
                {
                    for(auto j =base[1];j<=max[1];++j)
                    {
                        for(auto i =base[0];i<=max[0];++i)
                        {
                            auto n=it->data()->node(i,j,k);
                            if(std::fabs(n.template get<T>())<1e-32)
                                std::cout<<0.0<<std::endl;
                            else
                                std::cout<<static_cast<float>(n.template get<T>())<<std::endl;
                        }
                    }
                }
            }
        });


    }

    void write_vtk_good(std::string _filename, Domain& _lt )
    {
        std::ofstream ofs(_filename );
        ofs<<"# vtk DataFile Version 3.0"<<std::endl;
        ofs<<"vtk output"<<std::endl;
        ofs<<"ASCII"<<std::endl;
        ofs<<"DATASET UNSTRUCTURED_GRID"<<std::endl;
        int nPoints=0;
        for(auto it=_lt.begin_leafs();it!=_lt.end_leafs();++it)
        {
            nPoints+= it->data()->nodes().size();
        }

        ofs<<"POINTS "<<nPoints<<" float"<<std::endl;
        int nCells=0;
        int _count=0;
        for(auto it=_lt.begin_leafs();it!=_lt.end_leafs();++it)
        {
            auto block =it->data()->descriptor();
            block.extent()-=1;
            int e = block.nPoints();
            nCells+=e;
            for(auto& n : it->data()->nodes())
            {
                ofs<<n.global_coordinate()<<std::endl;
            }
            it->index(_count*it->data()->nPoints());
            ++_count;
        }
        ofs<<"\nCELLS "<<nCells<<" "<<nCells*9<<std::endl;

        //connectivity
        for (auto it = _lt.begin_leafs(); it != _lt.end_leafs(); ++it)
        {
            for (auto& n : it->data()->nodes())
            {
                if (!n.on_max_blockBorder() )
                //if (true)
                {
                    ofs << "8 ";
                    for (int i=0;i<2;++i)
                    {
                        for (int j=0;j<2;++j)
                        {
                            for (int k=0;k<2;++k)
                            {
                                auto nn = n.neighbor_check(coordinate_type({i,j,k}));
                                if (nn.second)
                                {
                                    ofs<< nn.first.index()+it->index()<<" ";
                                }
                            }
                        }
                    }
                    ofs<<std::endl;
                }
            }
        }
        ofs << "\nCELL_TYPES " << nCells << std::endl;
        for (int i = 0; i < nCells; ++i) ofs << "11" << std::endl;

        ofs << "POINT_DATA " << nPoints << std::endl;
        field_type_iterator_t::for_types([&_lt, &ofs, &nCells]<typename T>()
        {
            std::string name = "vertex_data_"+std::string(T::name());
            ofs << "SCALARS " << name << " float " << std::endl;
            ofs << "LOOKUP_TABLE default"          << std::endl;
            for (auto it = _lt.begin_leafs(); it != _lt.end_leafs(); ++it)
            {
                    for (auto& n : it->data()->nodes())
                    {
                        ofs << n.template get<T>() << std::endl;
                    }
            }
        });
    }

};

}

#endif
