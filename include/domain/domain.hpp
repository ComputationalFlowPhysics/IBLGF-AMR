#ifndef DOMAIN_INCLUDED_DOMAIN_HPP
#define DOMAIN_INCLUDED_DOMAIN_HPP

#include <vector>
#include <stdexcept>

// IBLGF-specific
#include <global.hpp>
#include <domain/octree/tree.hpp>
#include <dictionary/dictionary.hpp>
#include <domain/decomposition/decomposition.hpp>

namespace domain
{

using namespace dictionary;

/** @brief Spatial Domain
 *  @detail Given a datablock (and its corresponding dataFields)
 *  in Dim-dimensional space, the domain is constructed using an
 *  octree of blocks. Base blocks are read in from *  the config file.
 */
template<int Dim, class DataBlock>
class Domain
{

public:

    using datablock_t = DataBlock;
    using block_descriptor_t = typename datablock_t::block_descriptor_type;
    using extent_t           = typename block_descriptor_t::extent_t;
    using base_t             = typename block_descriptor_t::base_t;
    using scalar_coord_type  = typename block_descriptor_t::data_type;

    // tree related types
    using tree_t   = octree::Tree<Dim,datablock_t>;
    using key_t    = typename tree_t::key_type;
    using octant_t = typename tree_t::octant_type;

    // iterator types
    using dfs_iterator = typename tree_t::dfs_iterator;
    using bfs_iterator = typename tree_t::bfs_iterator;

    template<class Iterator=bfs_iterator>
    using conditional_iterator = typename tree_t::template conditional_iterator<Iterator>;

    using coordinate_type      = typename tree_t::coordinate_type;
    using real_coordinate_type = typename tree_t::real_coordinate_type;

    using field_type_iterator_t = typename datablock_t::field_type_iterator_t;

    using communicator_type = boost::mpi::communicator;
    using decompositon_type = Decomposition<Domain>;

    using refinement_condition_fct_t = std::function<bool(octant_t*, int diff_level)>;

    template<class DictionaryPtr>
    using block_initialze_fct = std::function<std::vector<extent_t>(DictionaryPtr,Domain*)>;

    static constexpr int dimension(){return Dim;}


public: //C/Dtors

    Domain(const Domain&  other) = delete;
    Domain(      Domain&& other) = default;
    Domain& operator=(const Domain&  other) & = delete;
    Domain& operator=(      Domain&& other) & = default;
    Domain()=default;
    ~Domain() = default;

    template<class DictionaryPtr>
    Domain(DictionaryPtr _dictionary,
           block_initialze_fct<DictionaryPtr> _init_fct=
                    block_initialze_fct<DictionaryPtr>())
    {
        this->initialize(_dictionary, _init_fct);
    }


    /** @brief Initialize domain, by reading in dictionary.
     *         Custom function may also be provided to generate all
     *         bases for the domain. Block extent_ is read in by the
     *         dictionary. */
    template<class DictionaryPtr>
    void initialize( DictionaryPtr _dictionary,
                     block_initialze_fct<DictionaryPtr> _init_fct=
                        block_initialze_fct<DictionaryPtr>() )
    {


        //Construct base mesh, vector of bases with a given block_extent_
        block_extent_=_dictionary->template get_or<int>("block_extent", 10);
        std::vector<extent_t> bases;
        if(_init_fct)
        {
            bases= _init_fct(_dictionary, this);
        }
        else
        {
            bases=construct_basemesh_blocks(_dictionary, block_extent_);
        }

        //Read scaling parameters from dictionary, bsaed on bounding box etc
        read_parameters(_dictionary);

        //Construct tree of base mesh
        extent_t max_extent=_dictionary->template get_or<int>("max_extent", 4096);
        construct_tree( bases,max_extent, block_extent_);
    }

    template<class DictionaryPtr>
    void read_parameters(DictionaryPtr _dictionary)
    {
        if(_dictionary->has_key("Lx"))
        {
            const float_type L= _dictionary->template get<float_type>("Lx");
            dx_base_=L/ (bounding_box_.extent()[0]-1);
        }
        else if(_dictionary->has_key("Ly"))
        {
            const float_type L= _dictionary->template get<float_type>("Ly");
            dx_base_=L/ (bounding_box_.extent()[1]-1);
        }
        else if(_dictionary->has_key("Lz"))
        {
            const float_type L= _dictionary->template get<float_type>("Lz");
            dx_base_=L/ (bounding_box_.extent()[2]-1);
        }
        else
        {
            throw std::runtime_error(
            "Domain: Please specify length scale Lx or Ly or Lz in dictionary"
            );
        }
    }

    /** @brief Initialize octree based on bases of the blocks.  **/
    void construct_tree( std::vector<extent_t>& bases,
                         extent_t _maxExtent, extent_t _blockExtent)
    {
        auto base_=bounding_box_.base()/_blockExtent;
        for(auto& b: bases) b-=base_;
        auto base_level=key_t::minimum_level(_maxExtent/_blockExtent);

        //Initialize tree only on the master process
        decomposition_ = decompositon_type(this);
        if(decomposition_.is_server())
        {
            t_ = std::make_shared<tree_t>(bases, base_level);

            //Assign octant to real coordinate transform:
            t_->get_octant_to_level_coordinate()=
                [ blockExtent=_blockExtent, base=base_]
                (real_coordinate_type _oct_coord, int _level)
                {
                    return (_oct_coord + base*std::pow(2,_level))*blockExtent;
                };

            //instantiate blocks
            for(auto it=begin_df();it!=end_df();++it)
            {
                const int level=0;
                auto bbase=t_->octant_to_level_coordinate(it->tree_coordinate());
                it->data()=std::make_shared<datablock_t>(bbase,
                        _blockExtent,level, false);
            }
        }
        else if(decomposition_.is_client())
        {
            t_=std::make_shared<tree_t>(base_level);

            //Instantiate blocks only after master has distributed tasks

            //Assign octant to real coordinate transform:
            t_->get_octant_to_level_coordinate()=
                [ blockExtent=_blockExtent, base=base_]
                (real_coordinate_type _oct_coord, int _level)
                {
                    return (_oct_coord + base*std::pow(2,_level))*blockExtent;
                };
        }
    }


    /** @brief Create base mesh out of blocks. Read in base blocks from configFile
     *         ad split up into blocks with given extent.
     **/
    template<class DictionaryPtr>
    auto construct_basemesh_blocks( DictionaryPtr _dictionary,
                                    extent_t _blockExtent)
    {
        auto _baseBlocks = parse_blocks(_dictionary);
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
        bounding_box_=block_descriptor_t(base_*e, extent*e+1);
        return bases;

    }


    void init_refine(int nRef=0)
    {
        if(is_server())
        {
            this->tree()->construct_leaf_maps();
            this->tree()->construct_level_maps();
            this->tree()->construct_neighbor_lists();

            for(int l=0;l<nRef;++l)
            {
                for (auto it = this->begin_leafs(); it != this->end_leafs(); ++it)
                {
                    if(!ref_cond_) return;
                    if(ref_cond_(*it, nRef-l) && it->refinement_level()==l)
                    {
                        this->refine(it);
                    }
                }
                this->tree()->construct_leaf_maps();
                this->tree()->construct_level_maps();
                this->tree()->construct_neighbor_lists();
            }

        }
     }

    void distribute()
    {
        decomposition_.distribute();
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

    /** @brief ConditionalIterator based on generic conditon lambda.
     *  Iterate through tree and skip octant if condition is not fullfilled.
     */
    template<class Func, class Iterator=bfs_iterator>
    auto begin_cond(const Func& _f){ return conditional_iterator<Iterator>(t_->root(), _f);}

    template<class Iterator=bfs_iterator>
    auto end_cond(){ return conditional_iterator<Iterator>();}

    template<class Iterator=bfs_iterator>
    auto begin_local(){
        return begin_cond<Iterator>( [](const auto& it){return it->locally_owned();});
    }
    template<class Iterator=bfs_iterator>
    auto end_local(){ return end_cond<Iterator>(); }

    template<class Iterator=bfs_iterator>
    auto begin_ghost(){
        return begin_cond<Iterator>( [](const auto& it){return !it->locally_owned();});
    }
    template<class Iterator=bfs_iterator>
    auto end_ghost(){ return end_cond<Iterator>(); }



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
        tree()->refine(*octant_it,[this](auto& child_it)
        {
            auto level = child_it->level()-this->tree()->base_level();
            auto bbase=t_->octant_to_level_coordinate(
                            child_it->tree_coordinate(), level);

            bool init_field=this->is_client();
            child_it->data()=
                std::make_shared<datablock_t>(bbase, block_extent_,level,init_field);
        });
    }


public:


    //template<class Field>
    void exchange_level_buffers(int level)
    {
        coordinate_type lbuff(1),hbuff(1);
        auto _begin=begin(level);
        auto _end=end(level);
        for(auto it=_begin ; it!=_end;++it )
        {
            //determine neighborhood
            //FIXME:  To be general this should include interlevel neighbors
            auto neighbors = it->get_level_neighborhood(lbuff, hbuff);

            //box-overlap per field
            it->data()->for_fields( [this,it, _begin, _end, &neighbors](auto& field)
            {
                for(auto& jt: neighbors)
                {
                    if(it->key()==jt->key())continue;

                    //Check for overlap with current
                    block_descriptor_t overlap;
                    if(field.buffer_overlap(jt->data()->descriptor(), overlap,
                                            jt->refinement_level()))
                    {
                        using field_type = std::remove_reference_t<decltype(field)>;
                        auto& src = jt->data()->template get<field_type>();
                        const auto overlap_src = overlap;

                        //it is target and jt is source
                        coordinate_type stride_tgt(1);
                        coordinate_type stride_src(1);

                        assign(src, overlap,stride_src,
                               field, overlap, stride_tgt);
                    }
                }
            });
        }
    }


    void exchange_buffers()
    {
        coordinate_type lbuff(1),hbuff(1);
        auto _begin=begin_leafs();
        auto _end=end_leafs();
        for(auto it=_begin ; it!=_end;++it )
        {
            //determine neighborhood
            //FIXME:  Only neighbors
            //FIXME:  periodicity needs to be accounted for
            auto neighbors = it->get_level_neighborhood(lbuff, hbuff);

            //box-overlap per field
            it->data()->for_fields( [this,it, _begin, _end, &neighbors](auto& field)
            {
                //for(auto& jt: neighbors)
                for(auto jt= _begin; jt!=_end;++jt )
                {
                    if(it->key()==jt->key())continue;

                    //Check for overlap with current
                    block_descriptor_t overlap;
                    if(field.buffer_overlap(jt->data()->descriptor(), overlap,
                                            jt->refinement_level()))
                    {
                        using field_type = std::remove_reference_t<decltype(field)>;
                        auto& src = jt->data()->template get<field_type>();
                        const auto overlap_src = overlap;

                        auto overlap_tgt = overlap_src;
                        overlap_tgt.level_scale(it->refinement_level());

                        //it is target and jt is source
                        int tgt_stride=1, src_stride=1;
                        if(it->refinement_level() > jt->refinement_level())
                        {
                            tgt_stride=2;
                            src_stride=1;
                        }
                        else if(it->refinement_level() < jt->refinement_level())
                        {
                            tgt_stride=1;
                            src_stride=2;
                        }
                        coordinate_type stride_tgt(tgt_stride);
                        coordinate_type stride_src(src_stride);

                        assign(src, overlap_src,stride_src,
                               field, overlap_tgt, stride_tgt);
                    }
                }
            });
        }
    }

    template<template<std::size_t> class Field>
    std::vector<std::pair<octant_t*,block_descriptor_t>>
    determine_fineToCoarse_interfaces(int _level)
    {
        std::vector<std::pair<octant_t*,block_descriptor_t>> res;
        auto _begin=begin_leafs();
        auto _end=end_leafs();
        for(auto it=_begin ; it!=_end;++it )
        {
            //determine neighborhood
            //TODO: This should only include the neighbors

            auto j_begin=begin_leafs();
            auto j_end=end_leafs();
                 for(auto jt= j_begin; jt!=j_end;++jt )
            {
                if(it->key()==jt->key())continue;
                if(it->refinement_level() != jt->refinement_level()+1) continue;

                //Check for overlap with current
                block_descriptor_t overlap;
                auto fBlock= it->data()->descriptor();
                auto cBlock= jt->data()->descriptor();

                auto crBlock= cBlock;
                cBlock.level_scale(it->refinement_level());
                cBlock.grow(1,1);

                if(fBlock.overlap(cBlock, overlap, fBlock.level()))
                {
                    bool viable =false;
                    int oc=0;
                    for(int d=0;d<3;++d)
                    {
                        if(overlap.extent()[d]==1)
                        {
                            //++oc;
                        }
                    }

                             viable=true;
                    //if(oc==1) //this avoids corners
                    {
                        //std::cout<<crBlock <<std::endl;
                        std::cout<<fBlock <<std::endl;
                        auto tt=fBlock;
                        std::cout<<overlap<<std::endl;
                        std::cout<<std::endl;
                        res.push_back(std::make_pair(*it, overlap));
                        //res.push_back(std::make_pair(it.ptr(), overlap));
                    }
                }
            }
        }
        return res;
    }

    block_descriptor_t get_level_bb(int _l)
    {
        auto _begin=begin_leafs();
        auto _end=end_leafs();
        block_descriptor_t bb(
                base_t(std::numeric_limits<scalar_coord_type>::max()),
                extent_t(0), _l
        );

        for(auto it=_begin ; it!=_end;++it )
        {
            if(it->level()==_l)
            {
               for(int d=0;d<Dim;++d)
               {
                   bb.base()[d]=
                       std::min(bb.base()[d], it->data()->descriptor().base()[d]);
               }
               bb.enlarge_to_fit(it->data()->descriptor()) ;
            }
        }
        return bb;
    }

public: //Access

    /**@brief Resolution on the base level */
    float_type dx_base()const noexcept{return dx_base_;}

    /**@brief Extent of each block */
    const extent_t& block_extent()const noexcept { return block_extent_; }
    /**@brief Extent of each block */
    extent_t& block_extent()noexcept { return block_extent_; }

    /**@brief Extent of each block */
    bool is_server()noexcept { return decomposition_.is_server(); }
    bool is_client()noexcept { return decomposition_.is_client(); }

    const decompositon_type& decomposition()const noexcept {
        return decomposition_;}
    decompositon_type& decomposition() noexcept{return decomposition_;}


    const refinement_condition_fct_t& register_refinement_condition() const noexcept
    {
        return ref_cond_;
    }
    refinement_condition_fct_t& register_refinement_condition() noexcept
    {
        return ref_cond_;
    }

public:

    friend std::ostream& operator<<(std::ostream& os, Domain& d)
    {
        boost::mpi::communicator w;
        os<<"Total number of processes: "<<w.size()<<std::endl;
        os<<"Number of octants: "<<d.num_leafs()<<std::endl;
        os<<"Block extent : "<<d.block_extent_<<std::endl;
        os<<"Base resolution "<<d.dx_base()<<std::endl;
        os<<"Base level "<<d.tree()->base_level()<<std::endl;
        os<<"Tree depth "<<d.tree()->depth()<<std::endl;

        os<<"Domain Bounding Box: "<<d.bounding_box_<<std::endl;
        //os<<"Fields:"<<std::endl;
        //auto it=d.begin_leafs();
        //it->data()->for_fields([&](auto& field)
        //        {
        //            os<<"\t "<<field.name()<<std::endl;
        //        }
        //);
        return os;
    }

private:

    template<class DictionaryPtr, class Fct >
    std::vector<block_descriptor_t>
    parse_blocks(DictionaryPtr _dict, Fct _fct)
    {
        if(_fct)
            return _fct(_dict);
        else
            return parse_blocks(_dict);
    }


    template<class DictionaryPtr>
    std::vector<block_descriptor_t>
    parse_blocks(DictionaryPtr _dict)
    {
        std::vector<block_descriptor_t> res;
        auto dicts=_dict->get_all_dictionaries("block");
        for(auto& sd: dicts)
        {
            auto base=sd->template get<int,Dim>("base");
            auto extent=sd->template get<int, Dim>("extent");
            const int level=0;
            res.emplace_back(base, extent,level);
        }
        return res;
    }


    /** @brief Default refinement condition */
    static bool refinement_cond_default( octant_t*, int ) { return false; }

private:
    std::shared_ptr<tree_t> t_;
    extent_t block_extent_;
    block_descriptor_t bounding_box_;
    float_type dx_base_;
    decompositon_type decomposition_;
    refinement_condition_fct_t ref_cond_ = &Domain::refinement_cond_default;

};

}

#endif
