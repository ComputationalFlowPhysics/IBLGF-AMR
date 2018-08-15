#ifndef DOMAIN_INCLUDED_DOMAIN_HPP
#define DOMAIN_INCLUDED_DOMAIN_HPP

#include <vector>
#include <stdexcept>

// IBLGF-specific
#include <global.hpp>
#include <domain/octree/tree.hpp>
#include <dictionary/dictionary.hpp>
	

namespace domain
{

using namespace dictionary;

template<int Dim, class DataBlock>
class Domain
{

public: 

    using datablock_t = DataBlock;
    using block_descriptor_t = typename datablock_t::block_descriptor_type;
    using extent_t           = typename block_descriptor_t::extent_t;
    using base_t             = typename block_descriptor_t::base_t;
    using scalar_coord_type  = typename block_descriptor_t::data_type;

    using tree_t   = octree::Tree<Dim,datablock_t>;
    using key_t    = typename tree_t::key_type;
    using octant_t = typename tree_t::octant_type;


    // iterator types
    using dfs_iterator = typename tree_t::dfs_iterator;
    using bfs_iterator = typename tree_t::bfs_iterator;

    using coordinate_type      = typename tree_t::coordinate_type;
    using real_coordinate_type = typename tree_t::real_coordinate_type;

    using field_type_iterator_t = typename datablock_t::field_type_iterator_t;

    static constexpr int dimension(){return Dim;}

public:

    Domain(const Domain&  other) = delete;
    Domain(      Domain&& other) = default;
    Domain& operator=(const Domain&  other) & = delete;
    Domain& operator=(      Domain&& other) & = default;
    ~Domain() = default;



    template<class DictionaryPtr>
    Domain(DictionaryPtr _dictionary)
    :Domain(parse_blocks(_dictionary, "block"), 
            extent_t(_dictionary->template get_or<int>("max_extent", 4096)),
            extent_t(_dictionary->template get_or<int>("block_extent", 10))
            )
    {
    }

    Domain(const std::vector<block_descriptor_t>& _baseBlocks, 
           extent_t _maxExtent= extent_t(4096),
           extent_t _blockExtent=extent_t(10))
    : block_extent_ (_blockExtent)
    {
        extent_t e(_blockExtent);
        std::vector<base_t> bases;
        for (auto& b : _baseBlocks)
        {
            for (int d = 0; d < Dim; ++d)
            {
                if (b.extent()[d]%e[d])
                {
                    throw 
                    std::runtime_error(
                    "Domain: Extent of blocks are not evenly divisible");
                }
                if (std::abs(b.base()[d])%e[d]/*&& e[d]%std::abs(b.base()[d])*/)
                {
                    throw 
                    std::runtime_error(
                    "Domain: Base of blocks are not evenly divisible");
                }
            }
            auto blocks_tmp =b.divide_into(e);
            for(auto &bb: blocks_tmp)
            {
                auto base_normalized=bb.base()/e;
                bases.push_back(base_normalized);
            }
        }
        extent_t max(std::numeric_limits<scalar_coord_type>::lowest());
        extent_t min(std::numeric_limits<scalar_coord_type>::max());
        for(auto& b : bases)
        {
           for(std::size_t d=0;d<b.size();++d)
           {
               if(b[d]<min[d]) min[d]=b[d];
               if(b[d]>max[d]) max[d]=b[d];
           }
        }

        extent_t extent(max-min+1);
        auto base_=extent_t(min);
        bounding_box_=block_descriptor_t(base_*e, extent*e);
        for(auto& b: bases) b-=base_;
        auto base_level=key_t::minimum_level(_maxExtent/_blockExtent);
        t_ = std::make_shared<tree_t>(bases, base_level);

        //Assign octant to real coordinate transform:
        t_->get_octant_to_real_coordinate()=
            [ blockExtent=_blockExtent, base=base_]
            (real_coordinate_type _oct_coord)
            {
                return (_oct_coord + base)*blockExtent;
            };

        //instantiate blocks
        for(auto it=t_->begin_leafs();it!=t_->end_leafs();++it)
        {
            const int level=0;
            auto bbase=t_->octant_to_real_coordinate(it->coordinate());
            it->data()=std::make_shared<datablock_t>(bbase, _blockExtent,level);
        }
    }


public: // Iterators:
    auto begin_leafs()     noexcept{ return t_->begin_leafs(); }
    auto end_leafs()       noexcept{ return t_->end_leafs(); }
    auto num_leafs() const noexcept{ return t_->num_leafs(); }
    auto begin_df()                { return dfs_iterator(t_->root()); }
    auto end_df()                  { return dfs_iterator(); }
    auto begin_bf()                { return bfs_iterator(t_->root()); }
    auto end_bf()                  { return bfs_iterator(); }
    auto begin()                   { return begin_df(); }
    auto end()                     { return end_df(); }
    auto begin(int _level)         { return t_->begin(_level); }
    auto end(int _level)           { return t_->end(_level); }

    template<class Iterator>
    auto begin_octant_nodes(Iterator it) noexcept{return it->data().nodes_begin();}
    template<class Iterator>
    auto end_octant_nodes(Iterator it) noexcept{return it->data().nodes_end();}



public:
    std::shared_ptr<tree_t> tree()const {return t_;}

    block_descriptor_t bounding_box()const noexcept{return bounding_box_;}

    template<class Iterator>
    void refine(Iterator& octant_it)
    {
        tree()->refine(octant_it,[this](auto& child_it)
        {
            auto bbase=t_->octant_to_real_coordinate(child_it->coordinate());
            auto level = child_it->level()-this->tree()->base_level();
            child_it->data()=std::make_shared<datablock_t>(bbase, block_extent_,level);
        });
    }

    const auto& block_extent()const noexcept { return block_extent_; }
    auto& block_extent()noexcept { return block_extent_; }


public:
    
    friend std::ostream& operator<<(std::ostream& os, Domain& d) 
    {
        os<<"Number of octants: "<<d.num_leafs()<<std::endl;
        os<<"Block extent : "<<d.block_extent_<<std::endl;

        os<<"Domain Bounding Box: "<<d.bounding_box_<<std::endl;
        os<<"Fields:"<<std::endl;
        auto it=d.begin_leafs();
        it->data()->for_fields([&](auto& field)
                {
                    os<<"\t "<<field.name()<<std::endl;
                }
        );
        return os;
    }

private:

    template<class DictionaryPtr>
    std::vector<block_descriptor_t> 
    parse_blocks(DictionaryPtr _dict, 
                 std::string _dict_name="block")
    {
        std::vector<block_descriptor_t> res;
        auto dicts=_dict->get_all_dictionaries(_dict_name);
        for(auto& sd: dicts)
        {
            auto base=sd->template get<int,Dim>("base");
            auto extent=sd->template get<int, Dim>("extent");
            const int level=0;
            res.emplace_back(base, extent,level);
        }
        return res;
    }


private:
    std::shared_ptr<tree_t> t_; 
    extent_t block_extent_;
    block_descriptor_t bounding_box_;



};

}

#endif 
