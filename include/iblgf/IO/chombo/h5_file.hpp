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

#ifndef xlb_framework_h5_file_hpp
#define xlb_framework_h5_file_hpp

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <array>
#include <boost/mpi/communicator.hpp>
#include <hdf5.h>
//#include "h5_simple.hpp"
//#include <IO/chombo/vector.hpp>
#include <typeinfo>

namespace iblgf
{
using float_type = double;

//Basic hdf5-types
template<typename T>
struct hdf_type
{
    static hid_t type()
    {
        throw std::runtime_error("no hdf5_type registered");
        return 0;
    }
};
template<typename T>
struct hdf_type<std::vector<T>>
{
    static hid_t type() { return hdf_type<T>::type(); }
};
template<>
struct hdf_type<double>
{
    static hid_t type() { return H5T_NATIVE_DOUBLE; }
};
template<>
struct hdf_type<float>
{
    static hid_t type() { return H5T_NATIVE_FLOAT; }
};
template<>
struct hdf_type<char>
{
    static hid_t type() { return H5T_FORTRAN_S1; }
};
template<>
struct hdf_type<const char>
{
    static hid_t type() { return H5T_NATIVE_INT; }
};
template<>
struct hdf_type<int>
{
    static hid_t type() { return H5T_NATIVE_UINT; }
};
#define HDF5_CHECK_ERROR(ID, MSG)                                              \
    if (ID < 0) throw std::runtime_error(MSG);

template<std::size_t NumDims>
struct hyperslab
{
    using hid_type = hid_t;
    using hsize_type = hsize_t;

    static constexpr std::size_t rank = NumDims;

    std::array<hsize_type, NumDims> chunk_dims_; // chunk dimensions dims;
    std::array<hsize_type, NumDims> offset_;     // start offset of first block
    std::array<hsize_type, NumDims> stride_;     // stride between blocks
    std::array<hsize_type, NumDims> count_;      // # of blocks in each dim
    std::array<hsize_type, NumDims> block_;      // size of block
};

//template<std::size_t NumDims, class Index_list_t=std::array<long,NumDims>>
template<std::size_t NumDims, class Index_list_t = math::vector<int, NumDims>>
class hdf5_file
{
    /*
    File            - a contiguous string of bytes in a computer store (memory, disk, etc.), and the bytes represent zero or more objects of the model
    Group           - a collection of objects (including groups)
    Dataset         - a multidimensional array of data elements with attributes and other metadata
    Dataspace       - a description of the dimensions of a multidimensional array
    Datatype        - a description of a specific class of data element including its storage layout as a pattern of bits
    Attribute       - a named data value associated with a group, dataset, or named datatype
    Property List   - a collection of parameters (some permanent and some transient) controlling options in the library
    Link            - the way objects are connected
    */

  public:
    static constexpr std::size_t dimension = NumDims;
    using hid_type = hid_t;
    using hsize_type = hsize_t;
    using index_list_t = Index_list_t; // math::vector<int,dimension>;

    //Ctor:
    hdf5_file() {}
    hdf5_file(std::string _filename, bool _file_exist = false)
    {
        if (_file_exist) open_file2(_filename);
        else
            create_file(_filename);
    }

    //Copy&Move
    hdf5_file(hdf5_file&&) = default;
    hdf5_file(const hdf5_file&) = default;

    //Assignment operator
    hdf5_file& operator=(hdf5_file&&) & = default;
    hdf5_file& operator=(const hdf5_file&) & = default;

    //box compounds for chombo
    struct box_compound
    {
        box_compound() = default;
        ~box_compound() = default;
        box_compound(const index_list_t& _min, const index_list_t& _max)
        {
            lo_i = static_cast<int>(_min[0]);
            lo_j = static_cast<int>(_min[1]);
            if (dimension == 3) lo_k = static_cast<int>(_min[2]);
            hi_i = static_cast<int>(_max[0]);
            hi_j = static_cast<int>(_max[1]);
            if (dimension == 3) hi_k = static_cast<int>(_max[2]);
        }

        int         lo_i;
        int         lo_j;
        int         lo_k;
        int         hi_i;
        int         hi_j;
        int         hi_k;
        std::string lo_i_str = "lo_i";
        std::string lo_j_str = "lo_j";
        std::string lo_k_str = "lo_k";
        std::string hi_i_str = "hi_i";
        std::string hi_j_str = "hi_j";
        std::string hi_k_str = "hi_k";
    };

    void open_file2(std::string _filename, bool default_open = true)
    {
        boost::mpi::communicator world;
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, world, mpiInfo);
        if (!default_open)
            file_id = H5Fopen(_filename.c_str(), H5F_ACC_RDONLY, plist_id);
        else
            file_id = H5Fopen(_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        HDF5_CHECK_ERROR(file_id, "hdf5: could not open file")
    }

    hid_type open_dataset(hid_t _loc_id, std::string _name)
    {
        // plist_id = H5Pcreate(H5P_FILE_ACCESS);
        //     auto dtplist_id = H5Pcreate(H5P_DATASET_XFER);
        //     H5Pset_dxpl_mpio(dtplist_id, H5FD_MPIO_INDEPENDENT);
        //hid_t dataset_id = H5Dopen2(_loc_id, _name.c_str(), dtplist_id);
        hid_t dataset_id = H5Dopen2(_loc_id, _name.c_str(), H5P_DEFAULT);
        HDF5_CHECK_ERROR(dataset_id, "hdf5: could not open dataset: " + _name)
        //     H5Pclose(dtplist_id);
        return dataset_id;
    }

    //TODO: somehow it doesn't work
    //~hdf5_file()=default;
    ~hdf5_file()
    {
        close_everything();
        //close_file(file_id);
    }

    void update_plist() {}

    void create_file(std::string _filename)
    {
        boost::mpi::communicator world;
        plist_id = H5Pcreate(H5P_FILE_ACCESS);

        // Set property list for parallel open if necessary
        if (world.size() == 1)
            std::cout << "Create file for serial write" << std::endl;
        else
            H5Pset_fapl_mpio(plist_id, world, MPI_INFO_NULL);

        file_id =
            H5Fcreate(_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
        HDF5_CHECK_ERROR(
            file_id, "HDF5-Error: Could not create file: " + _filename);
        H5Pclose(plist_id);
    }

    // create group using location and relative group name
    hid_type create_group(hid_type loc_id, std::string _group_identifier)
    {
        //Check if group already existent:
        /*auto status = */ H5Eset_auto1(NULL, NULL);
        //update_plist();
        auto non_existent =
            H5Lexists(loc_id, _group_identifier.c_str(), H5P_DEFAULT);
        if (!non_existent)
        {
            auto group_id = H5Gcreate(loc_id, _group_identifier.c_str(),
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            HDF5_CHECK_ERROR(group_id,
                "HDF5-Error: Could not create group: " + _group_identifier);
            return group_id;
        }
        else
        {
            //update_plist();
            return get_group(loc_id, _group_identifier);
        }
    }

    // create group just using name (absolute from root)
    hid_type create_group(std::string _group_identifier)
    {
        return create_group(file_id, _group_identifier);
    }

    hid_type create_simple_space(
        int rank = 1, const hsize_t* dims = -1, const hsize_t* max_dims = NULL)
    {
        return H5Screate_simple(rank, dims, max_dims);
    }

    hid_type get_dataspace(const hid_type& _dset_id)
    {
        auto space = H5Dget_space(_dset_id);
        HDF5_CHECK_ERROR(space, "HDF5-Error: Could not fetch space ");
        return space;
    }

    template<typename T>
    hid_type create_dataset(
        hid_type location_id, std::string _dset_name, hid_type dataspace_id)
    {
        auto hdf_t = hdf_type<T>::type();
        auto dcplist_id = H5Pcreate(H5P_DATASET_CREATE); // Must be collective
        auto dtplist_id = H5Pcreate(H5P_DATASET_XFER);   // Must be collective
        H5Pset_dxpl_mpio(dtplist_id, H5FD_MPIO_COLLECTIVE);
        auto dataset_id = H5Dcreate(location_id, _dset_name.c_str(), hdf_t,
            dataspace_id, H5P_DEFAULT, dcplist_id, H5P_DEFAULT);
        //auto dataset_id=H5Dcreate(location_id, _dset_name.c_str(), hdf_t, dataspace_id, H5P_DEFAULT, dcplist_id, dtplist_id);
        //auto dataset_id=H5Dcreate(location_id, _dset_name.c_str(), hdf_t, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        HDF5_CHECK_ERROR(
            dataset_id, "HDF5-Error: Could not create dset: " + _dset_name);
        H5Pclose(dcplist_id);
        H5Pclose(dtplist_id);
        return dataset_id;
    }

    std::array<hsize_type, dimension> dataspace_dimensions(
        const hid_type& _dset_id)
    {
        std::array<hsize_type, dimension> dims;
        const int                         dim_check =
            H5Sget_simple_extent_dims(get_dataspace(_dset_id), &dims[0], NULL);
        HDF5_CHECK_ERROR(
            dim_check, "HDF5-Error: Could not fetch space dimensions");
        std::reverse(dims.begin(), dims.end());
        return dims;
    }

    template<typename T>
    void write(hid_type& dset_id, hsize_type dimsf, T* buf)
    {
        auto filespace = H5Dget_space(dset_id);

        auto hdf_t = hdf_type<T>::type();
        auto memspace_id = H5Screate_simple(1, &dimsf, NULL);

        auto dtplist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(dtplist_id, H5FD_MPIO_COLLECTIVE);
        //H5Pset_dxpl_mpio(dtplist_id, H5FD_MPIO_INDEPENDENT);

        //H5Dwrite(dset_id, hdf_t, memspace_id, filespace, dtplist_id, buf);
        H5Dwrite(dset_id, hdf_t, memspace_id, filespace, H5P_DEFAULT, buf);
        H5Pclose(dtplist_id);
    }

    // parallel hyperslab write
    template<typename T>
    void write(hid_type& dset_id, hsize_type dimsf, hsize_type offset, T* buf)
    {
        auto hdf_t = hdf_type<T>::type();

        auto filespace = H5Dget_space(dset_id);
        auto memspace_id = H5Screate_simple(1, &dimsf, NULL);

        H5Sselect_hyperslab(
            filespace, H5S_SELECT_SET, &offset, NULL, &dimsf, NULL);

        auto dtplist_id = H5Pcreate(H5P_DATASET_XFER);
        // H5Pset_dxpl_mpio(dtplist_id, H5FD_MPIO_COLLECTIVE);
        H5Pset_dxpl_mpio(dtplist_id, H5FD_MPIO_INDEPENDENT);

        H5Dwrite(dset_id, hdf_t, memspace_id, filespace, dtplist_id, buf);
        H5Pclose(dtplist_id);
    }

    // paralle write ***** ? same as above?
    template<typename T>
    void write(hid_type& dset_id, hsize_type dimsf, hsize_type block_data_size,
        hsize_type offset, T* buf)
    //  void write(hid_type& dset_id, hsize_type dimsf, auto buf)
    {
        auto filespace = H5Dget_space(dset_id);

        auto hdf_t = hdf_type<T>::type();

        auto memspace_id = H5Screate_simple(1, &block_data_size, NULL);

        H5Sselect_hyperslab(
            filespace, H5S_SELECT_SET, &offset, NULL, &block_data_size, NULL);

        auto plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        //H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

        H5Dwrite(dset_id, hdf_t, memspace_id, filespace, plist_id, buf);
    }

    template<typename T>
    void read_dataset(
        hid_t& dataset_id, hid_t& memspace_id, hid_t& dataspace_id, T* data)
    {
        auto hdf_t = hdf_type<T>::type();
        auto status = H5Dread(
            dataset_id, hdf_t, memspace_id, dataspace_id, H5P_DEFAULT, data);
        HDF5_CHECK_ERROR(status, "HDF5-Error: Could not read dataset ");
    }

    template<std::size_t Dset_Dim = NumDims>
    std::vector<std::size_t> read_dataset_dimsion(hid_t dataset_id)
    {
        hid_t dataspace_id = H5Dget_space(dataset_id);
        HDF5_CHECK_ERROR(dataspace_id, "hdf5: could not open dataspace")
        std::vector<hsize_t> dims;
        auto dimension = H5Sget_simple_extent_ndims(dataspace_id);
        dims.resize(dimension);
        dimension = H5Sget_simple_extent_dims(dataspace_id, &dims[0], NULL);
        std::vector<std::size_t> res;
        for (unsigned int d = 0; d < dimension; ++d)
            res[d] = static_cast<std::size_t>(dims[d]);
        return res;
    }

    //Read the full dataset
    template<typename T, std::size_t Dset_Dim = NumDims>
    void read_dataset(hid_t& dataset_id, std::vector<T>& data)
    {
        hid_t dataspace_id = H5Dget_space(dataset_id);
        HDF5_CHECK_ERROR(dataspace_id, "hdf5: could not open dataspace")
        std::array<hsize_t, Dset_Dim> dims;
        const int                     dimension_check =
            H5Sget_simple_extent_dims(dataspace_id, &dims[0], NULL);
        HDF5_CHECK_ERROR(dimension_check, "hdf5: could query dimension")

        //if(std::is_same<storage_order,boost::fortran_storage_order>::value)
        //{
        //    std::reverse(dims.begin(), dims.end() );
        //}

        hsize_type n_elements = 1;
        for (int d = 0; d < Dset_Dim; ++d)
        {
            n_elements *= dims[d];
            std::cout << dims[d] << std::endl;
        }
        data.resize(n_elements);

        const hsize_t rank_out = 1; // dimensionality of output vector

        hsize_t dimsm[1]; // memory space dimensions
        dimsm[0] = data.size();
        auto memspace = H5Screate_simple(rank_out, dimsm, NULL);
        HDF5_CHECK_ERROR(memspace, "hdf5: Could not select create memory space")

        //Read the hyperslab
        read_dataset(dataset_id, memspace, dataspace_id, &data[0]);
    }

    template<class BlockDescriptor>
    std::vector<BlockDescriptor> read_box_descriptors(
        hid_t& dataset_id, int fake_level)
    {
        hid_t dataspace_id = H5Dget_space(dataset_id);

        std::vector<hsize_t> dims(1);
        H5Sget_simple_extent_dims(dataspace_id, &dims[0], NULL);

        box_compound* data =
            (box_compound*)malloc(dims[0] * sizeof(box_compound));

        hid_t memtype = H5Dget_type(dataset_id);
        H5Dread(dataset_id, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);
        std::vector<BlockDescriptor> vec_box(dims[0]);
        //TODO : somehow the direct conversion doesn't work
        for (int i = 0; i < static_cast<int>(dims[0]); ++i)
        {
            vec_box[i] = BlockDescriptor(std::array<int, NumDims>({data[i].lo_i,
                                             data[i].lo_j, data[i].lo_k}),
                std::array<int, NumDims>({data[i].hi_i - data[i].lo_i + 1,
                    data[i].hi_j - data[i].lo_j + 1,
                    data[i].hi_k - data[i].lo_k + 1}),
                fake_level);
        }
        return vec_box;
    }

    template<typename T, class Base, class Extent, class Stride,
        std::size_t Dset_Dim = NumDims>
    void read_hyperslab(hid_type& dataset_id, Base base, Extent extent,
        Stride stride, std::vector<T>& data)
    {
        hid_t dataspace_id = H5Dget_space(dataset_id);
        HDF5_CHECK_ERROR(dataspace_id, "hdf5: could not open dataspace")

        //Get NumDims of the file:
        int dimension = H5Sget_simple_extent_ndims(dataspace_id);
        std::vector<hsize_t> dims(dimension);
        dimension = H5Sget_simple_extent_dims(dataspace_id, &dims[0], NULL);
        HDF5_CHECK_ERROR(dimension, "hdf5: could query dimension")

        std::reverse(dims.begin(), dims.end());

        long int nElements = 1;
        for (unsigned int d = 0; d < Dset_Dim; ++d) nElements *= dims[d];

        std::vector<hsize_t> offset(dimension, 0);
        std::vector<hsize_t> count(dimension, 0);
        std::vector<hsize_t> step(dimension, 0);

        hsize_type n_elements_domain = 1;
        for (int d = 0; d < dimension; ++d)
        {
            offset[d] = static_cast<hsize_t>(base[d]);
            count[d] = static_cast<hsize_t>(extent[d]);
            step[d] = static_cast<hsize_t>(stride[d]);
            n_elements_domain *= extent[d];
        }
        //std::vector<float_type> data(n_elements_domain);
        data.resize(n_elements_domain);

        std::reverse(offset.begin(), offset.end());
        std::reverse(count.begin(), count.end());
        std::reverse(step.begin(), step.end());

        auto status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET,
            &offset[0], &step[0], &count[0], NULL);
        HDF5_CHECK_ERROR(status, "hdf5: Could not select hyperslab")

        const hsize_t rank_out = 1; // dimensionality of output vector
        hsize_t       dimsm[1];     // memory space dimensions
        dimsm[0] = data.size();
        auto memspace = H5Screate_simple(rank_out, dimsm, NULL);
        HDF5_CHECK_ERROR(memspace, "hdf5: Could not select create memory space")

        auto hdf_t = hdf_type<T>::type();
        status = H5Dread(
            dataset_id, hdf_t, memspace, dataspace_id, H5P_DEFAULT, &data[0]);
    }

    template<typename T>
    void write_hyperslab_cont1D(
        hid_type& dset_id, hyperslab<1> _hyperslab, T* buf)
    {
        auto filespace = H5Dget_space(dset_id);
        auto err = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
            &_hyperslab.offset_[0], NULL, &_hyperslab.count_[0], NULL);
        HDF5_CHECK_ERROR(err, "HDF5-Error: Could not select hyperslab ");

        auto hdf_t = hdf_type<T>::type();
        auto memspace_id = H5Screate_simple(1, &_hyperslab.count_[0], NULL);

        //plist_id = H5Pcreate(H5P_FILE_ACCESS);
        auto plist_id = H5Pcreate(H5P_DATASET_XFER);
        //            H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
        //H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        H5Dwrite(dset_id, hdf_t, memspace_id, filespace, plist_id, buf);

        close_space(filespace);
        close_space(memspace_id);
        H5Pclose(plist_id);
    }

    template<typename T, std::size_t Dset_Dim = NumDims>
    void write_hyperslab(
        hid_type& dset_id, hyperslab<Dset_Dim> _hyperslab, T* buf)
    {
        auto filespace = H5Dget_space(dset_id);
        auto err = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
            &_hyperslab.offset_[0], &_hyperslab.stride_[0],
            &_hyperslab.count_[0], &_hyperslab.block_[0]);
        HDF5_CHECK_ERROR(err, "HDF5-Error: Could not select hyperslab ");

        auto memspace_id =
            H5Screate_simple(Dset_Dim, &_hyperslab.chunk_dims_[0], NULL);

        auto hdf_t = hdf_type<T>::type();
        auto plist_id = H5Pcreate(H5P_DATASET_XFER);
        //           H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        H5Dwrite(dset_id, hdf_t, memspace_id, filespace, plist_id, buf);

        close_space(filespace);
        close_space(memspace_id);
        H5Pclose(plist_id);
    }

    hid_type get_group(hid_type& _loc_id, std::string _name)
    {
        auto g_id = H5Gopen(_loc_id, _name.c_str(), H5P_DEFAULT);
        HDF5_CHECK_ERROR(g_id, "HDF5-Error: Could not fetch group: " + _name);
        return g_id;
    }

    hid_type get_group(std::string _name)
    {
        auto g_id = H5Gopen(file_id, _name.c_str(), H5P_DEFAULT);
        HDF5_CHECK_ERROR(g_id, "HDF5-Error: Could not fetch group: " + _name);
        return g_id;
    }
    hid_type get_root() { return get_group("/"); }

    inline void close_space(hid_type _space_id)
    {
        auto err = H5Sclose(_space_id);
        HDF5_CHECK_ERROR(err, "HDF5-Error: Could not close space");
    }
    inline void close_group(hid_type _group_id)
    {
        auto err = H5Gclose(_group_id);
        HDF5_CHECK_ERROR(err, "HDF5-Error: Could not close group");
    }
    inline void close_dset(hid_type _dset_id)
    {
        auto err = H5Dclose(_dset_id);
        HDF5_CHECK_ERROR(err, "HDF5-Error: Could not close dataset");
    }
    inline void close_attribute(hid_type _attr_id)
    {
        auto err = H5Aclose(_attr_id);
        HDF5_CHECK_ERROR(err, "HDF5-Error: Could not close attribute");
    }
    inline void close_file(hid_type file_id)
    {
        auto err = H5Fclose(file_id);
        HDF5_CHECK_ERROR(err, "HDF5-Error: Could not close file");
    }
    inline void close_type(hid_type type_id)
    {
        auto err = H5Tclose(type_id);
        HDF5_CHECK_ERROR(err, "HDF5-Error: Could not close type ");
    }

    void close_everything() { close_everything(file_id); }
    void close_everything(hid_type file_id)
    {
        //update_plist();
        //H5Eset_auto(NULL,NULL, NULL);
        //H5Eset_auto(nullptr,nullptr, nullptr);
        auto numOpenObjs = H5Fget_obj_count(
            file_id, H5F_OBJ_DATASET | H5F_OBJ_GROUP | H5F_OBJ_DATATYPE);
        if (numOpenObjs > 0)
        {
            std::vector<hid_type> obj_id_list(numOpenObjs);
            auto                  numReturnedOpenObjs = H5Fget_obj_ids(file_id,
                H5F_OBJ_DATASET | H5F_OBJ_GROUP | H5F_OBJ_DATATYPE, -1,
                &obj_id_list[0]);
            for (hsize_type i = 0;
                 i < static_cast<hsize_type>(numReturnedOpenObjs); ++i)
                H5Oclose(obj_id_list[i]);
        }

        numOpenObjs = H5Fget_obj_count(file_id, H5F_OBJ_ATTR);
        if (numOpenObjs > 0)
        {
            std::vector<hid_type> obj_id_list(numOpenObjs);
            auto                  numReturnedOpenObjs =
                H5Fget_obj_ids(file_id, H5F_OBJ_ATTR, -1, &obj_id_list[0]);
            for (hsize_type i = 0;
                 i < static_cast<hsize_type>(numReturnedOpenObjs); ++i)
                H5Aclose(obj_id_list[i]);
        }

        numOpenObjs = H5Fget_obj_count(file_id, H5F_OBJ_FILE);
        if (numOpenObjs > 0)
        {
            std::vector<hid_type> obj_id_list(numOpenObjs);
            auto                  numReturnedOpenObjs =
                H5Fget_obj_ids(file_id, H5F_OBJ_FILE, -1, &obj_id_list[0]);
            for (hsize_type i = 0;
                 i < static_cast<hsize_type>(numReturnedOpenObjs); ++i)
                H5Fclose(obj_id_list[i]);
        }
    }

    //Attribute: Default impl for generic objects
    template<class T, class U = void>
    struct attribute_helper
    {
        inline static void create(hdf5_file* h5, hid_type& _group_id,
            std::string _attr_identifier, const T& _attr_value)
        {
            hsize_type dims_at = 1;
            hid_type   dataspace_at = H5Screate_simple(1, &dims_at, NULL);

            auto  hdf_t = hdf_type<T>::type();
            hid_t att_id = H5Acreate(_group_id, _attr_identifier.c_str(), hdf_t,
                dataspace_at, H5P_DEFAULT, H5P_DEFAULT);
            auto  ret = H5Awrite(att_id, hdf_t, &_attr_value);
            HDF5_CHECK_ERROR(
                ret, "HDF5-Error: Could not write attr: " + _attr_identifier);
            h5->close_space(dataspace_at);
            h5->close_attribute(att_id);
        }

        inline static T read(
            hdf5_file* h5, hid_type& _group_id, std::string _attr_identifier)
        {
            T    res{};
            auto attr =
                H5Aopen(_group_id, _attr_identifier.c_str(), H5P_DEFAULT);

            auto hdf_t = hdf_type<T>::type();
            auto ret = H5Aread(attr, hdf_t, &res);
            HDF5_CHECK_ERROR(ret, "hdf5: could not read Attribute")
            ret = H5Aclose(attr);
            HDF5_CHECK_ERROR(ret, "hdf5: could not close Attribute")
            return res;
        }
    };

    //Vectors Specilaization
    template<class T, class U>
    struct attribute_helper<std::vector<T>, U>
    {
        inline static void create(hdf5_file* h5, hid_type& _group_id,
            std::string _attr_identifier, std::vector<T> _attr_value)
        {
            hsize_type dims_at = _attr_value.size();
            hid_type   dataspace_at = H5Screate_simple(1, &dims_at, NULL);
            auto       hdf_t = hdf_type<std::vector<T>>::type();
            hid_t att_id = H5Acreate(_group_id, _attr_identifier.c_str(), hdf_t,
                dataspace_at, H5P_DEFAULT, H5P_DEFAULT);
            auto  ret = H5Awrite(att_id, hdf_t, &_attr_value[0]);
            HDF5_CHECK_ERROR(att_id,
                "HDF5-Error: Could not write attr: " + _attr_identifier);
            h5->close_space(dataspace_at);
            h5->close_attribute(att_id);
        }

        inline static std::vector<T> read(
            hdf5_file* h5, hid_type& _group_id, std::string _attr_identifier)
        {
            std::vector<T> res;
            hsize_t        dims;
            auto           hdf_t = hdf_type<std::vector<T>>::type();

            auto attr =
                H5Aopen(_group_id, _attr_identifier.c_str(), H5P_DEFAULT);
            auto   filetype = H5Aget_type(attr);
            size_t sdim = H5Tget_size(filetype);
            auto   space = H5Aget_space(attr);

            const int dimension = H5Sget_simple_extent_dims(space, &dims, NULL);
            res.resize(dims);

            auto status = H5Aread(attr, hdf_t, &res[0]);
            HDF5_CHECK_ERROR(status, "hdf5: could not read Attribute")
            return res;
        }
    };

    //String spezialization:
    template<class U>
    struct attribute_helper<std::string, U>
    {
        //Write string attr:
        inline static void create(hdf5_file* h5, hid_type& _group_id,
            std::string _attr_identifier, std::string _attr_value)
        {
            hsize_type leng = _attr_value.length() + 1;
            char*      string_ptr = new char[leng];
            std::strcpy(string_ptr, _attr_value.c_str());

            auto aid3 = H5Screate(H5S_SCALAR);
            auto atype = H5Tcopy(H5T_C_S1);
            H5Tset_size(atype, leng);
            H5Tset_strpad(atype, H5T_STR_NULLTERM);
            auto attr_id = H5Acreate2(_group_id, _attr_identifier.c_str(),
                atype, aid3, H5P_DEFAULT, H5P_DEFAULT);
            auto ret = H5Awrite(attr_id, atype, string_ptr);
            HDF5_CHECK_ERROR(
                ret, "HDF5-Error: Could not write string attr: " + _attr_value);
            h5->close_space(aid3);
            h5->close_attribute(attr_id);
            delete[] string_ptr;
        }

        //Read string attribute:
        inline static std::string read(
            hdf5_file* h5, hid_type& _group_id, std::string _attr_identifier)
        {
            auto attr =
                H5Aopen(_group_id, _attr_identifier.c_str(), H5P_DEFAULT);
            auto   filetype = H5Aget_type(attr);
            size_t sdim = H5Tget_size(filetype);

            char* rdata;
            rdata = (char*)malloc(sdim * sizeof(char*));

            auto memtype = H5Tcopy(H5T_C_S1);
            auto status = H5Tset_size(memtype, sdim);
            status = H5Aread(attr, memtype, rdata);
            HDF5_CHECK_ERROR(
                status, "HDF5-Error: Could not open string attr: ");

            std::string str(rdata);
            H5Aclose(attr);
            return str;
        }
    };

    //interface
    template<class T, class U = void>
    inline void create_attribute(
        hid_type& _group_id, std::string _attr_identifier, T _attr_value)
    {
        attribute_helper<T, U>::create(
            this, _group_id, _attr_identifier, _attr_value);
    }

    template<class T, class U = void>
    inline T read_attribute(hid_type& _group_id, std::string _attr_identifier)
    {
        return attribute_helper<T, U>::read(this, _group_id, _attr_identifier);
    }

    // CREATE + WRITE *******************************************************
    // write_boxCompound( group_id, name, vector of min, vector of max, bool=true)
    // write vector of boxes (boxes, patch_offsets)
    template<std::size_t ND = dimension>
    void write_boxCompound(hid_type& _group_id, std::string _cName,
        std::vector<index_list_t>& _min, std::vector<index_list_t>& _max,
        bool asAttr = true)
    {
        std::vector<box_compound> boxes;
        for (unsigned int i = 0; i < _min.size(); ++i)
        {
            box_compound _c(_min[i], _max[i]);
            boxes.push_back(_c);
        }
        using tag = std::integral_constant<std::size_t, ND>*;
        hsize_type size = static_cast<hsize_type>(_min.size());
        write_boxCompound(_group_id, _cName, &boxes[0], tag(0), size, asAttr);
    }

    // write_boxCompound( group_id, name, index_list_t min, index_list_t max, bool=true)
    // index_list_t is math::vector<int, dimension>
    // write one box (probDomain)
    template<std::size_t ND = dimension>
    void write_boxCompound(hid_type& _group_id, std::string _cName,
        index_list_t& _min, index_list_t& _max, bool asAttr = true)
    {
        box_compound _c(_min, _max);
        using tag = std::integral_constant<std::size_t, ND>*;
        write_boxCompound(_group_id, _cName, &_c, tag(0), 1, asAttr);
    }

    // CREATE + WRITE *******************************************************
    // for 2D
    void write_boxCompound(hid_type& _group_id, std::string _cName,
        box_compound* _c, std::integral_constant<std::size_t, 2>*,
        hsize_type _dim = 1, bool asAttr = true)
    {
        //create memory for the dataType:
        const hsize_type rank = 1;
        const hsize_type dim = _dim;
        auto             space = H5Screate_simple(rank, &dim, NULL);

        auto s1_tid = H5Tcreate(H5T_COMPOUND, sizeof(*_c));
        H5Tinsert(s1_tid, _c->lo_i_str.c_str(), HOFFSET(box_compound, lo_i),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->lo_j_str.c_str(), HOFFSET(box_compound, lo_j),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_i_str.c_str(), HOFFSET(box_compound, hi_i),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_j_str.c_str(), HOFFSET(box_compound, hi_j),
            H5T_NATIVE_INT);

        if (asAttr)
        {
            auto dataset = H5Acreate(_group_id, _cName.c_str(), s1_tid, space,
                H5P_DEFAULT, H5P_DEFAULT);
            auto status = H5Awrite(dataset, s1_tid, _c);
            close_attribute(dataset);
        }
        else
        {
            auto dataset = H5Dcreate(_group_id, _cName.c_str(), s1_tid, space,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            auto status =
                H5Dwrite(dataset, s1_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, _c);
            close_dset(dataset);
        }
        close_space(space);
        close_type(s1_tid);
    }

    // for 3D
    // write_boxCompound( group_id, name, box_compound, integral_const = 3, hsize_type _dim=1, bool=true)
    void write_boxCompound(hid_type& _group_id, std::string _cName,
        box_compound* _c, std::integral_constant<std::size_t, 3>*,
        hsize_type _dim = 1, bool asAttr = true)
    {
        //create memory for the dataType:
        const hsize_type rank = 1;
        const hsize_type dim = _dim;
        auto             space = H5Screate_simple(rank, &dim, NULL);

        auto s1_tid = H5Tcreate(H5T_COMPOUND, sizeof(*_c));
        H5Tinsert(s1_tid, _c->lo_i_str.c_str(), HOFFSET(box_compound, lo_i),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->lo_j_str.c_str(), HOFFSET(box_compound, lo_j),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->lo_k_str.c_str(), HOFFSET(box_compound, lo_k),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_i_str.c_str(), HOFFSET(box_compound, hi_i),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_j_str.c_str(), HOFFSET(box_compound, hi_j),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_k_str.c_str(), HOFFSET(box_compound, hi_k),
            H5T_NATIVE_INT);

        if (asAttr)
        {
            auto dataset = H5Acreate(_group_id, _cName.c_str(), s1_tid, space,
                H5P_DEFAULT, H5P_DEFAULT);
            /*auto status = */ H5Awrite(dataset, s1_tid, _c);
            close_attribute(dataset);
        }
        else
        {
            auto dataset = H5Dcreate(_group_id, _cName.c_str(), s1_tid, space,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            /*auto status = */ H5Dwrite(
                dataset, s1_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, _c);
            close_dset(dataset);
        }
        close_space(space);
        close_type(s1_tid);
    }

    // CREATE ONLY **********************************************************
    // create_boxCompound( group_id, name, vector of min, vector of max, bool=true)
    // write vector of boxes (boxes, patch_offsets)
    template<std::size_t ND = dimension>
    void create_boxCompound(hid_type& _group_id, std::string _cName,
        hsize_type _boxes_size = 1, bool asAttr = true)
    {
        index_list_t _min;
        _min[0] = 0;
        _min[1] = 0;
        _min[2] = 0;

        box_compound _c(_min, _min);
        using tag = std::integral_constant<std::size_t, ND>*;
        create_boxCompound(_group_id, _cName, &_c, tag(0), _boxes_size, asAttr);
    }

    // for 2D
    void create_boxCompound(hid_type& _group_id, std::string _cName,
        box_compound* _c, std::integral_constant<std::size_t, 2>*,
        hsize_type _dim = 1, bool asAttr = true)
    {
        //create memory for the dataType:
        const hsize_type rank = 1;
        const hsize_type dim = _dim;
        auto             space = H5Screate_simple(rank, &dim, NULL);

        int  ND = 2;
        int  sizeof_c = 2 * ND * 32 + 2 * ND * 4; // string size 32, int size 4
        auto s1_tid = H5Tcreate(H5T_COMPOUND, sizeof_c);
        //auto s1_tid = H5Tcreate (H5T_COMPOUND, sizeof(*_c));
        H5Tinsert(s1_tid, _c->lo_i_str.c_str(), HOFFSET(box_compound, lo_i),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->lo_j_str.c_str(), HOFFSET(box_compound, lo_j),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_i_str.c_str(), HOFFSET(box_compound, hi_i),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_j_str.c_str(), HOFFSET(box_compound, hi_j),
            H5T_NATIVE_INT);

        if (asAttr)
        {
            auto dataset = H5Acreate(_group_id, _cName.c_str(), s1_tid, space,
                H5P_DEFAULT, H5P_DEFAULT);
            //   auto status = H5Awrite(dataset, s1_tid, _c);
            close_attribute(dataset);
        }
        else
        {
            auto dataset = H5Dcreate(_group_id, _cName.c_str(), s1_tid, space,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            //   auto status = H5Dwrite(dataset, s1_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, _c);
            close_dset(dataset);
        }
        close_space(space);
        close_type(s1_tid);
    }

    // for 3D
    void create_boxCompound(hid_type& _group_id, std::string _cName,
        box_compound* _c, std::integral_constant<std::size_t, 3>*,
        hsize_type _dim = 1, bool asAttr = true)
    {
        //create memory for the dataType:
        const hsize_type rank = 1;
        const hsize_type dim = _dim;
        auto             space = H5Screate_simple(rank, &dim, NULL);

        int  ND = 3;
        int  sizeof_c = 2 * ND * 32 + 2 * ND * 4; // string size 32, int size 4
        auto s1_tid = H5Tcreate(H5T_COMPOUND, sizeof_c);
        //auto s1_tid = H5Tcreate (H5T_COMPOUND, sizeof(*_c));
        H5Tinsert(s1_tid, _c->lo_i_str.c_str(), HOFFSET(box_compound, lo_i),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->lo_j_str.c_str(), HOFFSET(box_compound, lo_j),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->lo_k_str.c_str(), HOFFSET(box_compound, lo_k),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_i_str.c_str(), HOFFSET(box_compound, hi_i),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_j_str.c_str(), HOFFSET(box_compound, hi_j),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_k_str.c_str(), HOFFSET(box_compound, hi_k),
            H5T_NATIVE_INT);

        if (asAttr)
        {
            auto dataset = H5Acreate(_group_id, _cName.c_str(), s1_tid, space,
                H5P_DEFAULT, H5P_DEFAULT);
            //        auto status = H5Awrite(dataset, s1_tid, _c);
            close_attribute(dataset);
        }
        else
        {
            auto dataset = H5Dcreate(_group_id, _cName.c_str(), s1_tid, space,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            //        auto status = H5Dwrite(dataset, s1_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, _c);
            close_dset(dataset);
        }
        close_space(space);
        close_type(s1_tid);
    }

    // OPEN + WRITE *******************************************************
    // open_write_boxCompound( group_id, name, vector of min, vector of max, bool=true)
    // write vector of boxes (boxes, patch_offsets)
    template<std::size_t ND = dimension>
    void open_write_boxCompound(hid_type& _group_id, std::string _cName,
        std::vector<index_list_t>& _min, std::vector<index_list_t>& _max,
        bool asAttr = true)
    {
        std::vector<box_compound> boxes;
        for (unsigned int i = 0; i < _min.size(); ++i)
        {
            box_compound _c(_min[i], _max[i]);
            boxes.push_back(_c);
        }
        using tag = std::integral_constant<std::size_t, ND>*;
        hsize_type size = static_cast<hsize_type>(_min.size());
        open_write_boxCompound(
            _group_id, _cName, &boxes[0], tag(0), size, asAttr);
    }

    // write_boxCompound( group_id, name, index_list_t min, index_list_t max, bool=true)
    // index_list_t is math::vector<int, dimension>
    // write one box (probDomain)
    template<std::size_t ND = dimension>
    void open_write_boxCompound(hid_type& _group_id, std::string _cName,
        index_list_t& _min, index_list_t& _max, bool asAttr = true)
    {
        box_compound _c(_min, _max);
        using tag = std::integral_constant<std::size_t, ND>*;
        open_write_boxCompound(_group_id, _cName, &_c, tag(0), 1, asAttr);
    }

    // for 2D
    void open_write_boxCompound(hid_type& _group_id, std::string _cName,
        box_compound* _c, std::integral_constant<std::size_t, 2>*,
        hsize_type _dim = 1, bool asAttr = true)
    {
        auto s1_tid = H5Tcreate(H5T_COMPOUND, sizeof(*_c));
        H5Tinsert(s1_tid, _c->lo_i_str.c_str(), HOFFSET(box_compound, lo_i),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->lo_j_str.c_str(), HOFFSET(box_compound, lo_j),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_i_str.c_str(), HOFFSET(box_compound, hi_i),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_j_str.c_str(), HOFFSET(box_compound, hi_j),
            H5T_NATIVE_INT);

        if (asAttr)
        {
            //    auto dataset = H5Acreate(_group_id, _cName.c_str(), s1_tid, space, H5P_DEFAULT, H5P_DEFAULT);
            auto dataset = H5Aopen(_group_id, _cName.c_str(), H5P_DEFAULT);
            auto status = H5Awrite(dataset, s1_tid, _c);
            close_attribute(dataset);
        }
        else
        {
            //    auto dataset = H5Dcreate(_group_id, _cName.c_str(), s1_tid, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            auto dataset = H5Dopen(_group_id, _cName.c_str(), H5P_DEFAULT);
            auto status =
                H5Dwrite(dataset, s1_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, _c);
            close_dset(dataset);
        }
        close_type(s1_tid);
    }

    // for 3D
    void open_write_boxCompound(hid_type& _group_id, std::string _cName,
        box_compound* _c, std::integral_constant<std::size_t, 3>*,
        hsize_type _dim = 1, bool asAttr = true)
    {
        auto s1_tid = H5Tcreate(H5T_COMPOUND, sizeof(*_c));
        H5Tinsert(s1_tid, _c->lo_i_str.c_str(), HOFFSET(box_compound, lo_i),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->lo_j_str.c_str(), HOFFSET(box_compound, lo_j),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->lo_k_str.c_str(), HOFFSET(box_compound, lo_k),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_i_str.c_str(), HOFFSET(box_compound, hi_i),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_j_str.c_str(), HOFFSET(box_compound, hi_j),
            H5T_NATIVE_INT);
        H5Tinsert(s1_tid, _c->hi_k_str.c_str(), HOFFSET(box_compound, hi_k),
            H5T_NATIVE_INT);

        if (asAttr)
        {
            //   auto dataset = H5Acreate(_group_id, _cName.c_str(), s1_tid, space, H5P_DEFAULT, H5P_DEFAULT);
            auto dataset = H5Aopen(_group_id, _cName.c_str(), H5P_DEFAULT);
            H5Awrite(dataset, s1_tid, _c);
            close_attribute(dataset);
        }
        else
        {
            //    auto dataset = H5Dcreate(_group_id, _cName.c_str(), s1_tid, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            auto dataset = H5Dopen(_group_id, _cName.c_str(), H5P_DEFAULT);
            H5Dwrite(dataset, s1_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, _c);
            close_dset(dataset);
        }
        close_type(s1_tid);
    }

    // *********************************************************************
    //Scan file, more or less taken from hdf5-site
    void info(std::ostream& os = std::cout)
    {
        //file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
        auto grp = H5Gopen(file_id, "/", H5P_DEFAULT);
        scan_group(grp, os);
    }

    //Some info functions
    std::vector<std::pair<std::string, hid_type>> get_all_datasets(
        hid_type _g_id)
    {
        std::vector<std::pair<std::string, hid_type>> res;
        char                                          group_name_c_str[maxChar];
        auto        len = H5Iget_name(_g_id, group_name_c_str, maxChar);
        std::string group_name(group_name_c_str);

        hsize_type nobj;
        auto       err = H5Gget_num_objs(_g_id, &nobj);
        HDF5_CHECK_ERROR(
            err, "HDF5-Error: Could get number of objects in group");
        for (auto i = 0; i < nobj; ++i)
        {
            char              memb_name_c_str[maxChar];
            const std::size_t maxChar = 1024;
            const auto        len2 = H5Gget_objname_by_idx(
                _g_id, static_cast<hsize_type>(i), memb_name_c_str, maxChar);
            std::string memb_name(memb_name_c_str);
            auto        otype =
                H5Gget_objtype_by_idx(_g_id, static_cast<std::size_t>(i));

            switch (otype)
            {
                case H5G_DATASET:
                {
                    auto dsid = H5Dopen(_g_id, memb_name_c_str, H5P_DEFAULT);
                    res.push_back(std::make_pair(memb_name, dsid));
                    break;
                }
                default:
                    break;
            }
        }
        return res;
    }

    std::vector<std::pair<std::string, hid_type>> get_all_groups(hid_type _g_id)
    {
        std::vector<std::pair<std::string, hid_type>> res;
        char                                          group_name_c_str[maxChar];
        auto        len = H5Iget_name(_g_id, group_name_c_str, maxChar);
        std::string group_name(group_name_c_str);

        hsize_type nobj;
        auto       err = H5Gget_num_objs(_g_id, &nobj);
        HDF5_CHECK_ERROR(
            err, "HDF5-Error: Could get number of objects in group");
        for (auto i = 0; i < nobj; ++i)
        {
            char              memb_name_c_str[maxChar];
            const std::size_t maxChar = 1024;
            const auto        len2 = H5Gget_objname_by_idx(
                _g_id, static_cast<hsize_type>(i), memb_name_c_str, maxChar);
            std::string memb_name(memb_name_c_str);
            auto        otype =
                H5Gget_objtype_by_idx(_g_id, static_cast<std::size_t>(i));

            switch (otype)
            {
                case H5G_GROUP:
                {
                    auto grpid = H5Gopen(_g_id, memb_name_c_str, H5P_DEFAULT);
                    res.push_back(std::make_pair(memb_name, grpid));
                    break;
                }
                default:
                    break;
            }
        }
        return res;
    }

    void scan_group(hid_type _g_id, std::ostream& os)
    {
        char        group_name_c_str[maxChar];
        auto        len = H5Iget_name(_g_id, group_name_c_str, maxChar);
        std::string group_name(group_name_c_str);

        scan_attrs(_g_id);
        hsize_type nobj;
        auto       err = H5Gget_num_objs(_g_id, &nobj);
        os << "Group Name: " << group_name << std::endl;
        HDF5_CHECK_ERROR(
            err, "HDF5-Error: Could get number of objects in group");
        os << "Member: ";
        for (auto i = 0; i < nobj; ++i)
        {
            char memb_name_c_str[maxChar];

            const std::size_t maxChar = 1024;
            const auto        len2 = H5Gget_objname_by_idx(
                _g_id, static_cast<hsize_type>(i), memb_name_c_str, maxChar);
            std::string memb_name(memb_name_c_str);
            auto        otype =
                H5Gget_objtype_by_idx(_g_id, static_cast<std::size_t>(i));

            switch (otype)
            {
                case H5G_LINK:
                {
                    printf(" SYM_LINK:\n");
                    scan_link(_g_id, memb_name_c_str);
                    break;
                }
                case H5G_GROUP:
                {
                    printf(" GROUP:\n");
                    auto grpid = H5Gopen(_g_id, memb_name_c_str, H5P_DEFAULT);
                    scan_group(grpid, os);
                    H5Gclose(grpid);
                    break;
                }
                case H5G_DATASET:
                {
                    printf(" DATASET:\n");
                    auto dsid = H5Dopen(_g_id, memb_name_c_str, H5P_DEFAULT);
                    scan_dataset(dsid);
                    H5Dclose(dsid);
                    break;
                }
                case H5G_TYPE:
                {
                    printf(" DATA TYPE:\n");
                    auto typeid_ = H5Topen(_g_id, memb_name_c_str, H5P_DEFAULT);
                    scan_datatype(typeid_);
                    H5Tclose(typeid_);
                    break;
                }
                default:
                    printf(" unknown?\n");
                    break;
            }
        }
    }

    void scan_attrs(hid_type obj_id) noexcept
    {
        const int na = H5Aget_num_attrs(obj_id);

        for (int i = 0; i < na; i++)
        {
            auto aid = H5Aopen_idx(obj_id, static_cast<unsigned int>(i));
            scan_attr(aid);
            H5Aclose(aid);
        }
    }
    void scan_attr(hid_type _attr_id) noexcept
    {
        char buf[maxChar];

        //Get the name of the attribute.
        auto len = H5Aget_name(_attr_id, maxChar, buf);
        printf("    Attribute Name : %s\n", buf);

        // Get attribute information: dataspace, data type
        auto aspace =
            H5Aget_space(_attr_id); /* the dimensions of the attribute data */
        auto atype = H5Aget_type(_attr_id);
        scan_datatype(atype);

        H5Tclose(atype);
        H5Sclose(aspace);
    }

    void scan_datatype(hid_type _dtype_id) noexcept
    {
        H5T_class_t t_class;
        t_class = H5Tget_class(_dtype_id);
        if (t_class < 0) { puts(" Invalid datatype.\n"); }
        else
        {
            /*
	        	 * Each class has specific properties that can be
	        	 * retrieved, e.g., size, byte order, exponent, etc.
	        	 */
            if (t_class == H5T_INTEGER)
            {
                puts(" Datatype is 'H5T_INTEGER'.\n");
                /* display size, signed, endianess, etc. */
            }
            else if (t_class == H5T_FLOAT)
            {
                puts(" Datatype is 'H5T_FLOAT'.\n");
                /* display size, endianess, exponennt, etc. */
            }
            else if (t_class == H5T_STRING)
            {
                puts(" Datatype is 'H5T_STRING'.\n");
                /* display size, padding, termination, etc. */
            }
            else if (t_class == H5T_BITFIELD)
            {
                puts(" Datatype is 'H5T_BITFIELD'.\n");
                /* display size, label, etc. */
            }
            else if (t_class == H5T_OPAQUE)
            {
                puts(" Datatype is 'H5T_OPAQUE'.\n");
                /* display size, etc. */
            }
            else if (t_class == H5T_COMPOUND)
            {
                puts(" Datatype is 'H5T_COMPOUND'.\n");
                /* recursively display each member: field name, type  */
            }
            else if (t_class == H5T_ARRAY)
            {
                puts(" Datatype is 'H5T_COMPOUND'.\n");
                /* display  dimensions, base type  */
            }
            else if (t_class == H5T_ENUM)
            {
                puts(" Datatype is 'H5T_ENUM'.\n");
                /* display elements: name, value   */
            }
            else
            {
                puts(" Datatype is 'Other'.\n");
                /* eg. Object Reference, ...and so on ... */
            }
        }
    }
    void scan_dataset(hid_type _dset_id)
    {
        hid_t   tid;
        hid_t   pid;
        hid_t   sid;
        hsize_t size;
        char    ds_name[maxChar];

        //Information about the group:
        //Name and attributes
        H5Iget_name(_dset_id, ds_name, maxChar);
        printf("Dataset Name : ");
        puts(ds_name);
        printf("\n");

        //process the attributes of the dataset, if any.
        scan_attrs(_dset_id);

        //Get dataset information: dataspace, data type
        sid = H5Dget_space(
            _dset_id); /* the dimensions of the dataset (not shown) */
        tid = H5Dget_type(_dset_id);
        printf(" DATA TYPE:\n");
        scan_datatype(tid);

        //Retrieve and analyse the dataset properties
        pid = H5Dget_create_plist(_dset_id); /* get creation property list */
        scan_plist(pid);
        size = H5Dget_storage_size(_dset_id);
        printf("Total space currently written in file: %d\n", (int)size);

        H5Pclose(pid);
        H5Tclose(tid);
        H5Sclose(sid);
    }

    void scan_link(hid_type _gid, char* name) noexcept
    {
        herr_t status;
        char   target[maxChar];

        status = H5Gget_linkval(_gid, name, maxChar, target);
        printf("Symlink: %s points to: %s\n", name, target);
    }

    void scan_plist(hid_type _pid) noexcept
    {
        hsize_t          chunk_dims_out[2];
        int              rank_chunk;
        int              nfilters;
        H5Z_filter_t     filtn;
        int              i;
        unsigned int     filt_flags, filt_conf;
        size_t           cd_nelmts;
        unsigned int     cd_values[32];
        char             f_name[maxChar];
        H5D_fill_time_t  ft;
        H5D_alloc_time_t at;
        H5D_fill_value_t fvstatus;
        unsigned int     szip_options_mask;
        unsigned int     szip_pixels_per_block;

        /* zillions of things might be on the plist */
        /*  here are a few... */

        /*
	         * get chunking information: rank and dimensions.
	         *
	         *  For other layouts, would get the relevant information.
	         */
        if (H5D_CHUNKED == H5Pget_layout(_pid))
        {
            rank_chunk = H5Pget_chunk(_pid, 2, chunk_dims_out);
            printf("chunk rank %d, dimensions %lu x %lu\n", rank_chunk,
                (unsigned long)(chunk_dims_out[0]),
                (unsigned long)(chunk_dims_out[1]));
        } /* else if contiguous, etc. */

        /*
	         *  Get optional filters, if any.
	         *
	         *  This include optional checksum and compression methods.
	         */

        nfilters = H5Pget_nfilters(_pid);
        for (i = 0; i < nfilters; i++)
        {
            /* For each filter, get
	        	 *   filter ID
	        	 *   filter specific parameters
	        	 */
            cd_nelmts = 32;
            filtn = H5Pget_filter(_pid, (unsigned)i, &filt_flags, &cd_nelmts,
                cd_values, (size_t)maxChar, f_name, &filt_conf);
            /*
	        	 *  These are the predefined filters
	        	 */
            switch (filtn)
            {
                case H5Z_FILTER_DEFLATE: /* AKA GZIP compression */
                    printf("DEFLATE level = %d\n", cd_values[0]);
                    break;
                case H5Z_FILTER_SHUFFLE:
                    printf("SHUFFLE\n"); /* no parms */
                    break;
                case H5Z_FILTER_FLETCHER32:
                    printf("FLETCHER32\n"); /* Error Detection Code */
                    break;
                case H5Z_FILTER_SZIP:
                    szip_options_mask = cd_values[0];
                    ;
                    szip_pixels_per_block = cd_values[1];

                    printf("SZIP COMPRESSION: ");
                    printf("PIXELS_PER_BLOCK %d\n", szip_pixels_per_block);
                    /* print SZIP options mask, etc. */
                    break;
                default:
                    printf("UNKNOWN_FILTER\n");
                    break;
            }
        }

        /*
	         *  Get the fill value information:
	         *    - when to allocate space on disk
	         *    - when to fill on disk
	         *    - value to fill, if any
	         */
        printf("ALLOC_TIME ");
        H5Pget_alloc_time(_pid, &at);

        switch (at)
        {
            case H5D_ALLOC_TIME_EARLY:
                printf("EARLY\n");
                break;
            case H5D_ALLOC_TIME_INCR:
                printf("INCR\n");
                break;
            case H5D_ALLOC_TIME_LATE:
                printf("LATE\n");
                break;
            default:
                printf("unknown allocation policy");
                break;
        }

        printf("FILL_TIME: ");
        H5Pget_fill_time(_pid, &ft);
        switch (ft)
        {
            case H5D_FILL_TIME_ALLOC:
                printf("ALLOC\n");
                break;
            case H5D_FILL_TIME_NEVER:
                printf("NEVER\n");
                break;
            case H5D_FILL_TIME_IFSET:
                printf("IFSET\n");
                break;
            default:
                printf("?\n");
                break;
        }

        H5Pfill_value_defined(_pid, &fvstatus);

        if (fvstatus == H5D_FILL_VALUE_UNDEFINED)
        { printf("No fill value defined, will use default\n"); }
        else
        {
            /* Read  the fill value with H5Pget_fill_value.
	        	 * Fill value is the same data type as the dataset.
	        	 * (details not shown)
	        	 **/
        }

        /* ... and so on for other dataset properties ... */
    }

    hid_type get_file_id() { return file_id; }

  private:
    hid_type file_id;   // file_id
    hid_type filespace; // file space id
    hid_type plist_id;  // property list identifier

    //Hyperslab properties:
    std::array<hsize_type, dimension> dimsf_; // dataset dimension
    MPI_Info                          mpiInfo = MPI_INFO_NULL;

    static constexpr std::size_t maxChar = 1024;
};
} // namespace iblgf

#endif //xlb_framework_io_chombo_hpp
