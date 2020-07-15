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

#ifndef INCLUDED_LGF_DOMAIN_DATABLOCK_HPP
#define INCLUDED_LGF_DOMAIN_DATABLOCK_HPP

#include <iostream>
#include <vector>
#include <tuple>

// IBLGF-specific
#include <iblgf/types.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>
#include <iblgf/utilities/tuple_utilities.hpp>
#include <iblgf/domain/dataFields/node.hpp>
#include <iblgf/domain/dataFields/datafield.hpp>

namespace iblgf
{
namespace domain
{
template<int Dim, template<class> class NodeType, class... DataFieldType>
class DataBlock : public BlockDescriptor<int, Dim>
{
    static constexpr auto dimension = Dim;

  public: //member types
    using node_t = NodeType<DataBlock>;
    friend node_t;

    using fields_tuple_t = std::tuple<DataFieldType...>;
    using field_type_iterator_t = tuple_utils::TypeIterator<DataFieldType...>;
    using node_field_type = DataField<node_t, Dim>;
    using buffer_type = typename node_field_type::buffer_d_t;

    using node_itertor = typename std::vector<node_t>::iterator;
    using node_const_iterator = typename std::vector<node_t>::const_iterator;
    using node_reverse_iterator =
        typename std::vector<node_t>::reverse_iterator;
    using node_const_reverse_iterator =
        typename std::vector<node_t>::const_reverse_iterator;
    using size_type = types::size_type;

    using block_descriptor_type = BlockDescriptor<int, dimension>;
    using super_type = block_descriptor_type;
    using coordinate_type = typename block_descriptor_type::coordinate_type;

    template<typename T>
    using vector_type = types::vector_type<T, dimension>;

    using real_coordinate_type = vector_type<types::float_type>;

  public: //Ctors:
    DataBlock() = default;
    ~DataBlock() = default;
    DataBlock(const DataBlock& rhs) = delete;
    DataBlock& operator=(const DataBlock&) & = default;

    DataBlock(DataBlock&& rhs) = default;
    DataBlock& operator=(DataBlock&&) & = default;

    DataBlock(coordinate_type _base, coordinate_type _extent, int _level = 0,
        bool _allocate = true)
    : super_type(_base, _extent, _level)
    {
        this->initialize(this->descriptor(), _allocate);
    }

    DataBlock(const block_descriptor_type& _b, bool _allocate = true)
    : super_type(_b)
    {
        this->initialize(_b, _allocate);
    }

    void initialize(const block_descriptor_type& _b, bool _allocate)
    {
        tuple_utils::for_each(fields, [&_b, _allocate](auto& field) {
            field.initialize(_b, _allocate, true, 0.0);
        });
        this->generate_nodes();
    }

  public: //Access and queries
    template<class Function>
    void for_fields(Function&& F)
    {
        tuple_utils::for_each(fields, F);
    }

    block_descriptor_type&       descriptor() noexcept { return *this; }
    const block_descriptor_type& descriptor() const noexcept { return *this; }

    block_descriptor_type bounding_box() const noexcept
    {
        return bounding_box_;
    }

    auto&       node_field() noexcept { return node_field_; }
    const auto& node_field() const noexcept { return node_field_; }

    auto nodes_domain_begin() const noexcept { return nodes_domain_.begin(); }
    auto nodes_domain_end() const noexcept { return nodes_domain_.end(); }
    const auto& nodes_domain() const { return nodes_domain_; }
    auto&       nodes_domain() { return nodes_domain_; }

    auto begin() noexcept { return nodes_domain_.begin(); }
    auto end() noexcept { return nodes_domain_.end(); }

    bool is_allocated() { return std::get<0>(fields)[0].data().size() > 0; }

  public: //Operators for tuple access
    template<class Tag,
        typename std::enable_if<
            std::tuple_element<tagged_tuple_index<typename Tag::tag_type,
                                   fields_tuple_t>::value,
                fields_tuple_t>::type::nFields() == 1,
            void>::type* = nullptr>
    auto& operator[](Tag _tag) noexcept
    {
        return std::get<
            tagged_tuple_index<typename Tag::tag_type, fields_tuple_t>::value>(
            fields)[0];
    }
    template<class Tag,
        typename std::enable_if<
            std::tuple_element<tagged_tuple_index<typename Tag::tag_type,
                                   fields_tuple_t>::value,
                fields_tuple_t>::type::nFields() == 1,
            void>::type* = nullptr>
    const auto& operator[](Tag _tag) const noexcept
    {
        return std::get<
            tagged_tuple_index<typename Tag::tag_type, fields_tuple_t>::value>(
            fields)[0];
    }
    template<class Tag,
        typename std::enable_if<
            std::tuple_element<tagged_tuple_index<typename Tag::tag_type,
                                   fields_tuple_t>::value,
                fields_tuple_t>::type::nFields() >= 2,
            void>::type* = nullptr>
    auto& operator[](Tag _tag) noexcept
    {
        return std::get<
            tagged_tuple_index<typename Tag::tag_type, fields_tuple_t>::value>(
            fields);
    }
    template<class Tag,
        typename std::enable_if<
            std::tuple_element<tagged_tuple_index<typename Tag::tag_type,
                                   fields_tuple_t>::value,
                fields_tuple_t>::type::nFields() >= 2,
            void>::type* = nullptr>
    const auto& operator[](Tag _tag) const noexcept
    {
        return std::get<
            tagged_tuple_index<typename Tag::tag_type, fields_tuple_t>::value>(
            fields);
    }

    template<class Tag>
    auto& operator()(Tag _tag, int _idx = 0) noexcept
    {
        return std::get<
            tagged_tuple_index<typename Tag::tag_type, fields_tuple_t>::value>(
            fields)[_idx];
    }
    template<class Tag>
    const auto& operator()(Tag _tag, int _idx = 0) const noexcept
    {
        return std::get<
            tagged_tuple_index<typename Tag::tag_type, fields_tuple_t>::value>(
            fields)[_idx];
    }

    template<class Tag>
    auto& operator()(Tag _tag, const coordinate_type& _c, int _idx = 0) noexcept
    {
        return std::get<
            tagged_tuple_index<typename Tag::tag_type, fields_tuple_t>::value>(
            fields)[_idx]
            .get(_c);
    }
    template<class Tag>
    const auto& operator()(
        Tag _tag, const coordinate_type& _c, int _idx = 0) const noexcept
    {
        return std::get<
            tagged_tuple_index<typename Tag::tag_type, fields_tuple_t>::value>(
            fields)[_idx]
            .get(_c);
    }

    friend std::ostream& operator<<(std::ostream& os, const DataBlock& c)
    {
        tuple_utils::for_each(c.fields, [&os](auto& field) {
            os << "container field: " << field.name() << std::endl;
        });
        return os;
    }

    /*************************************************************************/

  private: //private member helpers
    /** @brief Generate nodes from the field tuple, both domain and nodes incl
     * buffer
     **/
    void generate_nodes()
    {
        bounding_box_ = *this;
        buffer_type lbuff(0), rbuff(0);
        for_fields([&](auto& field) {
            for (int d = 0; d < Dim; ++d)
            {
                if (field[0].lbuffer()[d] > lbuff[d])
                    lbuff[d] = field[0].lbuffer()[d];
                if (field[0].hbuffer()[d] > rbuff[d])
                    rbuff[d] = field[0].hbuffer()[d];
            }
            bounding_box_.enlarge_to_fit(field[0].real_block());
        });

        node_field_.initialize(*this, lbuff, rbuff);
        for (std::size_t i = 0; i < node_field_.size(); ++i)
        { node_field_[i] = node_t(this, i); }

        //Store most common views in vector of nodes:
        nodes_domain_.clear();
        nodes_domain_.resize(this->size());
        auto dview = node_field_.domain_view();
        int  count = 0;
        for (auto it = dview.begin(); it != dview.end(); ++it)
        { nodes_domain_[count++] = *it; }
    }

  private: //Data members
    /** @brief Fields stored in datablock */
    fields_tuple_t fields;

    /** @brief nodes in physical domain */
    std::vector<node_t> nodes_domain_;

    /** @brief field of nodes, including buffers. */
    node_field_type node_field_;

    /** @brief bounding box of all fields in the block*/
    super_type bounding_box_;
};

} // namespace domain
} // namespace iblgf

#endif

