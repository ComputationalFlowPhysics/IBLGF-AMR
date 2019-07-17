#ifndef INCLUDED_VK_IO_HPP
#define INCLUDED_VK_IO_HPP


#include <ostream>
#include <fstream>
#include <iostream>
#include <sstream>

// IBLGF-specific
#include <global.hpp>

namespace io
{



template<class Domain>
class Vtk_io
{

public:

    static constexpr int Dim = Domain::dimension;
    using coordinate_type        = typename Domain::coordinate_type;
    using datablock_t            = typename Domain::datablock_t;
    using field_type_iterator_t  = typename Domain::field_type_iterator_t;
    using field_tuple            = typename datablock_t::fields_tuple_t;

public:

    void write_vtk(std::string _filename, Domain* _lt )
    {
        std::ofstream ofs(_filename );
        ofs<<"# vtk DataFile Version 3.0"<<std::endl;
        ofs<<"vtk output"<<std::endl;
        ofs<<"ASCII"<<std::endl;
        ofs<<"DATASET UNSTRUCTURED_GRID"<<std::endl;
        int nPoints=0;
        for(auto it=_lt->begin_leafs();it!=_lt->end_leafs();++it)
        {
            auto b= it->data()->descriptor();
            b.grow(0,1); //grow by one to fill the gap
            nPoints+= b.nPoints();
        }

        ofs<<"POINTS "<<nPoints<<" float"<<std::endl;
        int nCells=0;
        int _count=0;
        for(auto it=_lt->begin_leafs();it!=_lt->end_leafs();++it)
        {
            auto block =it->data()->descriptor();
            int e = block.nPoints();
            nCells+=e;
            block.grow(0,1); 

            auto base=block.base();
            auto max=block.max();

            for(auto k =base[2];k<=max[2];++k)
            {
                for(auto j =base[1];j<=max[1];++j)
                {
                    for(auto i =base[0];i<=max[0];++i)
                    {
                        auto n=it->data()->node(i,j,k);
                        ofs<<n.global_coordinate()<<std::endl;
                    }
                }
            }
            it->index(_count*block.nPoints());
            ++_count;
        }
        ofs<<"\nCELLS "<<nCells<<" "<<nCells*9<<std::endl;
        
        //connectivity
        for (auto it = _lt->begin_leafs(); it != _lt->end_leafs(); ++it)
        {

            auto block =it->data()->descriptor();
            auto base=block.base();
            auto max=block.max();

            auto index_block= block;
            index_block.grow(0,1);
            for(auto k =base[2];k<=max[2];++k)
            {
                for(auto j =base[1];j<=max[1];++j)
                {
                    for(auto i =base[0];i<=max[0];++i)
                    {
                        //if (!n.on_max_blockBorder() )
                        //if (true)
                        ofs << "8 ";
                        for (int k2=0;k2<2;++k2)
                        {
                            for (int j2=0;j2<2;++j2)
                            {
                                for (int i2=0;i2<2;++i2)
                                {
                                    auto nn_idx=index_block.index(
                                            coordinate_type({i+i2,j+j2,k+k2}));

                                        ofs<< nn_idx+it->index()<<" ";
                                }
                            }
                        }
                        ofs<<std::endl;
                    }
                }
            }
        }
        ofs << "\nCELL_TYPES " << nCells << std::endl;
        for (int i = 0; i < nCells; ++i) ofs << "11" << std::endl;

        //ofs << "POINT_DATA " << nPoints << std::endl;
        //field_type_iterator_t::for_types([&_lt, &ofs, &nCells]<typename T>()
        //{
        //    std::string name = "vertex_data_"+std::string(T::name());
        //    ofs << "SCALARS " << name << " float " << std::endl;
        //    ofs << "LOOKUP_TABLE default"          << std::endl;
        //    for (auto it = _lt->begin_leafs(); it != _lt->end_leafs(); ++it)
        //    {
        //            for (auto& n : it->data()->nodes())
        //            {
        //                ofs << n.template get<T>() << std::endl;
        //            }
        //    }
        //});
        ofs << "CELL_DATA " << nCells << std::endl;
        field_type_iterator_t::for_types([&_lt, &ofs, &nCells]<typename T>()
        {
            std::string name = "cell_data_"+std::string(T::name());
            ofs << "SCALARS " << name << " float " << std::endl;
            ofs << "LOOKUP_TABLE default"          << std::endl;
            for (auto it = _lt->begin_leafs(); it != _lt->end_leafs(); ++it)
            {
                auto block =it->data()->descriptor();
                auto base=block.base();
                auto max=block.max();

                auto index_block= block;
                index_block.grow(0,1);
                for(auto k =base[2];k<=max[2];++k)
                {
                    for(auto j =base[1];j<=max[1];++j)
                    {
                        for(auto i =base[0];i<=max[0];++i)
                        {
                            auto n=it->data()->node(i,j,k);
                            if(std::fabs(n.template get<T>())<1e-32)
                                ofs<<0.0<<std::endl;
                            else
                                ofs<<static_cast<float>(n.template get<T>())<<std::endl;
                        }
                    }
                }
            }
        });
    }
    void write_vtk_good(std::string _filename, Domain* _lt )
    {
        std::ofstream ofs(_filename );
        ofs<<"# vtk DataFile Version 3.0"<<std::endl;
        ofs<<"vtk output"<<std::endl;
        ofs<<"ASCII"<<std::endl;
        ofs<<"DATASET UNSTRUCTURED_GRID"<<std::endl;
        int nPoints=0;
        for(auto it=_lt->begin_leafs();it!=_lt->end_leafs();++it)
        {
            nPoints+= it->data()->nodes().size();
        }

        ofs<<"POINTS "<<nPoints<<" float"<<std::endl;
        int nCells=0;
        int _count=0;
        for(auto it=_lt->begin_leafs();it!=_lt->end_leafs();++it)
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
        for (auto it = _lt->begin_leafs(); it != _lt->end_leafs(); ++it)
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
            for (auto it = _lt->begin_leafs(); it != _lt->end_leafs(); ++it)
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
