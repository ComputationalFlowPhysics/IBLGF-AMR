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
    using adapt_condition_fct_t = std::function<void(std::vector<float_type>, std::vector<key_t>&, std::vector<int>& )>;

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
        boost::mpi::communicator w;
        if(w.rank()!=0)client_comm_=client_comm_.split(1);
        else client_comm_=client_comm_.split(0);


        //Construct base mesh, vector of bases with a given block_extent_
        block_extent_=_dictionary->template get_or<int>("block_extent", 14);

        bd_base_=_dictionary->template get<int,Dim>("bd_base");
        bd_extent_=_dictionary->template get<int, Dim>("bd_extent");

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
        construct_tree( bases,bd_extent_, block_extent_);
    }

    template<class DictionaryPtr>
    void read_parameters(DictionaryPtr _dictionary)
    {
        if(_dictionary->has_key("Lx"))
        {
            const float_type L= _dictionary->template get<float_type>("Lx");
            dx_base_=L/ (bounding_box_.extent()[0]);
        }
        else if(_dictionary->has_key("Ly"))
        {
            const float_type L= _dictionary->template get<float_type>("Ly");
            dx_base_=L/ (bounding_box_.extent()[1]);
        }
        else if(_dictionary->has_key("Lz"))
        {
            const float_type L= _dictionary->template get<float_type>("Lz");
            dx_base_=L/ (bounding_box_.extent()[2]);
        }
        else if(_dictionary->has_key("dx_base"))
        {
            dx_base_=_dictionary->template get<float_type>("dx_base");
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

    template<class DictionaryPtr>
    void initialize_with_keys( DictionaryPtr _dictionary, std::string restart_domain_dir)
    {
        boost::mpi::communicator w;
        if(w.rank()!=0)client_comm_=client_comm_.split(1);
        else client_comm_=client_comm_.split(0);

        //Construct base mesh, vector of bases with a given block_extent_
        block_extent_=_dictionary->template get_or<int>("block_extent", 14);

        bd_base_=_dictionary->template get<int,Dim>("bd_base");
        bd_extent_=_dictionary->template get<int, Dim>("bd_extent");
        bounding_box_=block_descriptor_t(bd_base_, bd_extent_);

        auto base_=bounding_box_.base()/block_extent_;

        //Read scaling parameters from dictionary, bsaed on bounding box etc
        read_parameters(_dictionary);

        auto base_level=key_t::minimum_level(bd_extent_/block_extent_);

        decomposition_ = decompositon_type(this);
        t_=std::make_shared<tree_t>(base_level);

        t_->get_octant_to_level_coordinate()=
            [ blockExtent=block_extent_, base=base_]
            (real_coordinate_type _oct_coord, int _level)
            {
                return (_oct_coord + base*std::pow(2,_level))*blockExtent;
            };

        if(decomposition_.is_server())
        {

            std::vector<key_t> keys;
            std::vector<bool> leafs;

            t_->read(restart_domain_dir, keys, leafs);

            t_->insert_keys(keys, [&](octant_t* _o){
                    auto level = _o->refinement_level();
                    level=level>=0?level:0;
                    auto bbase=this->tree()->octant_to_level_coordinate(
                            _o->tree_coordinate(),level);

                    _o->data()=std::make_shared<datablock_t>(bbase,
                            this->block_extent(),level, false);
                    }, false);

            //assuming the same order
            int c=0;
            for (auto it=this->begin(); it!=this->end(); ++it)
            {

                if (it->level()+1 > this->tree()->depth())
                    this->tree()->depth()=it->level()+1;
                it->flag_leaf(leafs[c]);
                c++;
            }

            std::cout<< "Server read restart from keys done " <<std::endl;
        }

        extent_t e(block_extent_);
        extent_t bd_key_base_(0);
        extent_t bd_key_interior_base_(baseBlockBufferNumber_);
        extent_t bd_key_extent_=bd_extent_;
        extent_t bd_key_interior_extent_=bd_extent_;

        for(std::size_t d=0;d<bd_key_interior_extent_.size();++d)
        {
            bd_key_interior_extent_[d]=bd_key_interior_extent_[d]/e[d]-2*baseBlockBufferNumber_;
            bd_key_extent_[d]=bd_key_extent_[d]/e[d];
        }

        key_bd_box_=block_descriptor_t(bd_key_base_, bd_key_extent_);
        key_bd_interior_box_=block_descriptor_t(bd_key_interior_base_, bd_key_interior_extent_);

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

        for(std::size_t d=0;d<min.size();++d)
        {
            if ((min[d])*e[d]<bd_base_[d])
            {
                std::cout<<"The bouding box provided might be smaller than a domain block"<<std::endl;
                bd_extent_[d]+=(bd_base_[d]-min[d]*e[d]);
                bd_base_[d]=min[d]*e[d];
            }
            if ( (max[d]+1)*e[d]> bd_base_[d]+bd_extent_[d])
            {
                std::cout<<"The bouding box provided might be smaller than a domain block"<<std::endl;
                bd_extent_[d]=(max[d]+1)*e[d] -  bd_base_[d];
            }
        }

        bounding_box_=block_descriptor_t(bd_base_, bd_extent_);

        extent_t bd_key_base_(0);
        extent_t bd_key_interior_base_(baseBlockBufferNumber_);
        extent_t bd_key_extent_=bd_extent_;
        extent_t bd_key_interior_extent_=bd_extent_;

        for(std::size_t d=0;d<min.size();++d)
        {
            bd_key_interior_extent_[d]=bd_key_interior_extent_[d]/e[d]-2*baseBlockBufferNumber_;
            bd_key_extent_[d]=bd_key_extent_[d]/e[d];
        }

        key_bd_box_=block_descriptor_t(bd_key_base_, bd_key_extent_);
        key_bd_interior_box_=block_descriptor_t(bd_key_interior_base_, bd_key_interior_extent_);

        return bases;
        std::cout<<"Initial base level blocks done "<<std::endl;

    }

    void output_level_test()
    {
        if(is_server())
        {
            const int base_level=this->tree()->base_level();
            for(int l= this->tree()->depth()-1; l>=base_level;--l)
            {
                for (auto it = this->begin(l);
                        it != this->end(l);
                        ++it)
                {
                    it->flag_leaf(it->refinement_level()==0);
                    //if (it->refinement_level()>0)
                    //    this->tree()->delete_oct(it.ptr());
                }
            }
            this->tree()->construct_lists();
            this->tree()->construct_level_maps();
            this->tree()->construct_leaf_maps(true);

        }

        this->decomposition().sync_decomposition();
    }

    void delete_all_children(std::vector<std::vector<key_t>>& deletion, bool only_unmark_deletion=true)
    {
        const int base_level=this->tree()->base_level();
        for(int l= this->tree()->depth()-2; l>=base_level;--l)
        {
            for (auto it = this->begin(l);
                    it != this->end(l);
                    ++it)
            {
                if (it->is_leaf_search(true)) continue;

                bool delete_all_children=true;

                for (int i = 0; i < it->num_children(); ++i)
                {

                    auto child = it->child(i);
                    if (!child || !child->data())
                        continue;

                    if ( !child->aim_deletion() || (!child->is_leaf() && !child->is_correction())  )
                    {
                        delete_all_children=false;
                        break;
                    }
                }

                if (delete_all_children)
                {
                    if (only_unmark_deletion)
                    {
                        for (int i = 0; i < it->num_children(); ++i)
                        {
                            auto child = it->child(i);
                            if (!child )
                                continue;
                            child->flag_leaf(false);
                            child->flag_correction(false);
                            child->leaf_boundary()=false;
                            child->aim_deletion(true);

                        }
                        it->flag_leaf(true);
                        it->flag_correction(false);
                        it->leaf_boundary()=false;
                        it->aim_deletion(false);

                    }
                    else
                    {
                    }
                }
                else
                {
                    for (int i = 0; i < it->num_children(); ++i)
                    {
                        auto child = it->child(i);

                        if (child && child->is_leaf())
                            child->aim_deletion(false);
                    }
                 }
            }
        }


    }

    void mark_leaf_boundary()
    {

        for (auto it=this->begin(); it!=this->end(); ++it)
            it->leaf_boundary()=false;

        for (auto it=this->begin(); it!=this->end(); ++it)
        {
            if (!it->data()) continue;

            if (it->is_leaf())
            {
                for(int i=0;i<it->nNeighbors();++i)
                {
                    auto neighbor_it=it->neighbor(i);
                    if (!neighbor_it || !neighbor_it->data()|| neighbor_it->is_correction() || neighbor_it->is_leaf()) continue;
                    it->leaf_boundary()=true;
                }
            } else
            {
                if (it->is_correction()) continue;
                for(int i=0;i<it->nNeighbors();++i)
                {
                    auto neighbor_it=it->neighbor(i);
                    if (!neighbor_it || !neighbor_it->data() || !neighbor_it->is_leaf()) continue;
                    it->leaf_boundary()=true;
                }
            }

            if (it->leaf_boundary())
            {
                for (int i = 0; i < it->num_children(); ++i)
                {
                    auto child = it->child(i);
                    if (child && child->data())
                        child->leaf_boundary()=true;
                }
            }
        }
     }

    void mark_correction()
    {
        const auto base_level=this->tree()->base_level();

        for (auto it=this->begin(); it!=this->end(); ++it)
        {
            it->physical(false);
        }


        for(int l=this->tree()->depth()-1; l>=base_level;--l)
        {
            for (auto it = this->begin(l);
                    it != this->end(l);
                    ++it)
            {
                if (!it->data()) continue;
                it->physical( it->is_leaf() && !it->aim_deletion());

                if (!it->physical())
                {
                    for (int i = 0; i < it->num_children(); ++i)
                    {
                        auto child = it->child(i);
                        if (child && child->data())
                            it->physical(it->physical() || child->physical());
                    }
                }
            }
        }


        // Add correction buffers
        if (use_correction_buffer_)
        {
            for(int l=base_level+1;l< this->tree()->depth();++l)
            {
                for (auto it = this->begin(l);
                        it != this->end(l);
                        ++it)
                {
                    if (!it->data()) continue;
                    if (!it->physical()) continue;

                    it->tree()->
                    insert_correction_neighbor(*it,
                            [this](auto neighbor_it)
                            {
                            auto level = neighbor_it->level()-this->tree()->base_level();
                            auto bbase=t_->octant_to_level_coordinate(
                                    neighbor_it->tree_coordinate(), level);

                            bool init_field=false;
                            neighbor_it->data()=
                            std::make_shared<datablock_t>(bbase, block_extent_,level,init_field);
                            } );
                }
            }

            this->tree()->construct_lists();
            this->tree()->construct_level_maps();
            this->tree()->construct_leaf_maps(true);

            for(int l=base_level+1;l< this->tree()->depth();++l)
            {
                for (auto it = this->begin(l);
                        it != this->end(l);
                        ++it)
                {
                    if (!it->physical()) continue;
                    //it->flag_correction(false);

                    for(int i=0;i<it->nNeighbors();++i)
                    {
                        auto neighbor_it=it->neighbor(i);
                        if (!neighbor_it || !neighbor_it->data()|| neighbor_it->physical()) continue;

                        neighbor_it->aim_deletion(false);
                        neighbor_it->flag_correction(true);

                    }
                }
            }

            this->tree()->construct_level_maps();
            this->tree()->construct_leaf_maps(true);
            this->tree()->construct_lists();

        }

        // flag base level boundary correction
        for (auto it = this->begin(base_level);
                it != this->end(base_level);
                ++it)
        {
            if (!it->data()) continue;
            if (!it->physical()) continue;
            bool _neighbors_exists=true;
            for(int i=0;i<it->nNeighbors();++i)
            {
                if(!it->neighbor(i) || !it->neighbor(i)->data())
                    _neighbors_exists=false;
            }

            if (!_neighbors_exists)
            {
                it->flag_correction(true);
                it->aim_deletion(false);
                it->flag_leaf(true);
            }
            else
            {
                it->flag_correction(false);
                it->aim_deletion(false);
            }
        }

        this->tree()->construct_level_maps();
        this->tree()->construct_lists();
        this->tree()->construct_leaf_maps(true);
    }

    void restart_list_construct()
    {
        if(is_server())
        {
            this->tree()->construct_level_maps();
            this->tree()->construct_lists();
            this->tree()->construct_leaf_maps(true);
            mark_correction();
            mark_leaf_boundary();
        }
    }

    void init_refine(int nRef, int level_up_max)
    {
        if(is_server())
        {
            this->tree()->construct_leaf_maps();
            this->tree()->construct_level_maps();
            this->tree()->construct_lists();

            std::unordered_map<key_t, bool> checklist;
            const auto base_level=this->tree()->base_level();
            for(int l=0;l<nRef;++l)
            {
                for (auto it = begin_df(); it != end_df(); ++it)
                {
                    if(!ref_cond_) return;
                    if(ref_cond_(it.ptr(), nRef-l) && it->refinement_level()==l)
                    {
                        if (this->tree()->try_2to1(it->key(), this->key_bounding_box(), checklist))
                            this->refine(it.ptr());
                    }
                }
            }

            this->tree()->construct_leaf_maps(true);
            this->tree()->construct_level_maps();
            this->tree()->construct_lists();

            for (int global_=0; global_<level_up_max; ++global_)
            {
                for (int l = this->tree()->depth()-1; l>=base_level; --l)
                {
                    for (auto it = begin(l); it != end(l); ++it)
                    {
                        if (it->is_leaf() )
                        {
                            this->refine(*it, false);
                        }
                    }
                }
                this->tree()->construct_leaf_maps(true);
                this->tree()->construct_level_maps();
                this->tree()->construct_lists();
            }

            mark_correction();
            mark_leaf_boundary();
        }

    }

    template<class LoadCalculator, class FmmMaskBuilder>
    void distribute()
    {
        decomposition_.template distribute<LoadCalculator, FmmMaskBuilder>();
    }

    auto adapt(std::vector<float_type> source_max, bool &base_mesh_update)
    {
        //communicating with server
        return decomposition_.adapt_decoposition(source_max, base_mesh_update);
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

    int num_corrections()
    {
        int c=0;
        for (auto it=this->begin(); it!=this->end(); ++it)
        {
            if (it->is_correction() && !it->is_leaf())
                ++c;
        }

        return c;
    }

    int num_allocations()
    {
        int c=0;
        for (auto it=this->begin(); it!=this->end(); ++it)
        {
            if (it->data() && it->data()->is_allocated())
                ++c;
        }

        return c;
    }


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
    block_descriptor_t key_bounding_box(bool interior=true)
    {
        if (interior)
            return key_bd_interior_box_;
        else
            return key_bd_box_;
    }
    int baseBlockBufferNumber() {return baseBlockBufferNumber_;}

    template<class Iterator>
    void refine(Iterator* octant_it, bool ratio_2to1=true)
    {
        tree()->
            refine(octant_it,
                [this](auto child_it)
                {
                    auto level = child_it->level() - this->tree()->base_level();
                    auto bbase=t_->octant_to_level_coordinate(
                        child_it->tree_coordinate(), level);

                    bool init_field=this->is_client();
                    child_it->data()=
                        std::make_shared<datablock_t>(bbase, block_extent_,level,init_field);
                },
                ratio_2to1
        );
    }

    template<class Iterator>
    void refine_with_exisitng_correction(Iterator* octant_it, std::vector<std::vector<key_t>>& deletion, bool ratio_2to1=true)
    {
        tree()->
            refine(octant_it,
                [this, &deletion](auto child_it)
                {
                    if (child_it && child_it->data() && child_it->is_correction() && child_it->rank()!=child_it->parent()->rank())
                    {
                        deletion[child_it->rank()].emplace_back(child_it->key());
                        child_it->rank()=-1;
                    }

                    auto level = child_it->level() - this->tree()->base_level();
                    auto bbase=t_->octant_to_level_coordinate(
                        child_it->tree_coordinate(), level);

                    bool init_field=this->is_client();
                    if (!child_it->data())
                        child_it->data()=
                            std::make_shared<datablock_t>(bbase, block_extent_,level,init_field);
                },
                ratio_2to1
        );

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
    bool is_server()const noexcept { return decomposition_.is_server(); }
    bool is_client()const noexcept { return decomposition_.is_client(); }

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

    const adapt_condition_fct_t& register_adapt_condition() const noexcept
    {
        return adapt_cond_;
    }
    adapt_condition_fct_t& register_adapt_condition() noexcept
    {
        return adapt_cond_;
    }


    std::vector<int> get_nPoints() noexcept
    {
        //auto client_comm_ = this->client_communicator();
        int nPts=0;
        int nPts_global=0;
        int nLevels=this->tree()->depth()-this->tree()->base_level();
        std::vector<int> nPoints_perLevel(nLevels,0);
        std::vector<int> nPoints_perLevel_global(nLevels,0);

        if(this->is_client())
        {
            for (auto it  = this->begin_leafs();
                    it != this->end_leafs(); ++it)
            {
                if(!it->locally_owned() || !it->data())continue;

                nPts+=it->data()->node_field().size();
                nPoints_perLevel[it->refinement_level()]+=
                    it->data()->node_field().size();
            }
        }
        boost::mpi::all_reduce(client_comm_,nPts, nPts_global, std::plus<int>());

        for(std::size_t i=0;i<nPoints_perLevel.size();++i)
        {
            boost::mpi::all_reduce(client_comm_,nPoints_perLevel[i],
                                   nPoints_perLevel_global[i],
                                   std::plus<int>());
        }
        nPoints_perLevel_global.push_back(nPts_global);
        return nPoints_perLevel_global;
    }
    int nLevels()const noexcept
    {
        return this->tree()->depth()-this->tree()->base_level();
    }
public:

    friend std::ostream& operator<<(std::ostream& os, Domain& d)
    {
        boost::mpi::communicator w;
        os<<"Total number of processes: "<<w.size()<<std::endl;
        os<<"Total number of leaf octants: "<<d.num_leafs()<<std::endl;
        os<<"Block extent : "<<d.block_extent_<<std::endl;
        os<<"Base resolution "<<d.dx_base()<<std::endl;
        os<<"Base level "<<d.tree()->base_level()<<std::endl;
        os<<"Tree depth "<<d.tree()->depth()<<std::endl;

        os<<"Domain Bounding Box: "<<d.bounding_box_<<std::endl;
        os<<"Interior Key Bounding Box: "<<d.key_bd_box_<<std::endl;
        //os<<"Fields:"<<std::endl;
        //auto it=d.begin_leafs();
        //it->data()->for_fields([&](auto& field)
        //        {
        //            os<<"\t "<<field.name()<<std::endl;
        //        }
        //);
        return os;
    }

    const auto& client_communicator() const noexcept { return client_comm_; }
    auto& client_communicator() noexcept { return client_comm_; }

   const bool& correction_buffer()const noexcept{return use_correction_buffer_;}
   bool& correction_buffer()noexcept{return use_correction_buffer_;}

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

    static void adapt_cond_default(std::vector<float_type> source_max,
            std::vector<key_t>& octs,
            std::vector<int>& level_change )
    {}

private:
    std::shared_ptr<tree_t> t_;
    extent_t block_extent_;
    extent_t bd_base_, bd_extent_;

    block_descriptor_t bounding_box_;
    block_descriptor_t key_bd_box_;
    block_descriptor_t key_bd_interior_box_;
    float_type dx_base_;
    decompositon_type decomposition_;

    refinement_condition_fct_t ref_cond_ = &Domain::refinement_cond_default;
    adapt_condition_fct_t adapt_cond_ = &Domain::adapt_cond_default;

    boost::mpi::communicator client_comm_;
    int baseBlockBufferNumber_=2;

    bool use_correction_buffer_=true;

};

//class DomainOperators
//{
//
//
//};

}

#endif
