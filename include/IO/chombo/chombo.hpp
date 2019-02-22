#ifndef CHOMBO_HPP
#define CHOMBO_HPP

#include <IO/chombo/chombo_impl.hpp>

#include <IO/chombo/h5_file.hpp>


namespace chombo_writer
{

    template< std::size_t Dim, class BlockDescriptor, class FieldData, class Domain>
    using chombo_t= typename chombo_writer::Chombo<Dim, BlockDescriptor,
                                    FieldData, Domain, hdf5_file<Dim> >;


}



#endif   //Chombo
