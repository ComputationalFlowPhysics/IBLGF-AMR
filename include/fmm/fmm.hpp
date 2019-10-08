#ifndef IBLGF_INCLUDED_FMM
#define IBLGF_INCLUDED_FMM

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

// IBLGF-specific
#include <types.hpp>
#include <algorithm>
#include <list>
#include <vector>
#include <iostream>
#include <cmath>
#include <functional>
#include <cstring>
#include <fftw3.h>

#include <global.hpp>
#include <simulation.hpp>
#include <linalg/linalg.hpp>
#include <fmm/fmm_nli.hpp>
#include <domain/domain.hpp>
#include <domain/octree/tree.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <domain/octree/tree.hpp>
#include <IO/parallel_ostream.hpp>
#include "../utilities/convolution.hpp"
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

namespace fmm
{

template<class Domain>
struct FmmMaskBuilder
{

    using octant_t = typename Domain::octant_t;
    using MASK_LIST = typename octant_t::MASK_LIST;

public:
    static void fmm_if_load_build(Domain* domain_)
    {
        int base_level=domain_->tree()->base_level();
        // During 1 timestep
        // IF called 6X3=18 times while fmm called 3 times
        // effective factor 6

        fmm_dry_init_base_level_masks(domain_, base_level, true,
                6, true);
        fmm_dry_init_base_level_masks(domain_, base_level, false,
                6, true);
    }

    static void fmm_lgf_mask_build(Domain* domain_)
    {
        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {
            fmm_dry(domain_, l, false);
            fmm_dry(domain_, l, true);
        }
    }
    static void fmm_dry(Domain* domain_, int base_level, bool non_leaf_as_source)
    {

        fmm_dry_init_base_level_masks(domain_, base_level, non_leaf_as_source);
        fmm_upward_pass_masks(
                domain_,
                base_level,
                MASK_LIST::Mask_FMM_Source,
                MASK_LIST::Mask_FMM_Target,
                non_leaf_as_source);
    }

    static void fmm_upward_pass_masks(
            Domain* domain_,
            int base_level,
            int mask_source_id,
            int mask_target_id,
            bool non_leaf_as_source,
            float_type _base_factor=1.0)
    {

        int refinement_level = base_level-domain_->tree()->base_level();
        int fmm_mask_idx_ = refinement_level*2+non_leaf_as_source;

        // for all levels
        for (int level=base_level-1; level>=0; --level)
        {
            //_base_factor *=0.6;
            // parent's mask is true if any of its child's mask is true
            for (auto it = domain_->begin(level);
                    it != domain_->end(level);
                    ++it)
            {
                it->fmm_mask(fmm_mask_idx_,mask_source_id,false);
                for ( int c = 0; c < it->num_children(); ++c )
                {
                    if ( it->child(c) && it->child(c)->fmm_mask(fmm_mask_idx_,mask_source_id) )
                    {
                        it->fmm_mask(fmm_mask_idx_,mask_source_id, true);
                        break;
                    }
                }
            }

            for (auto it = domain_->begin(level);
                    it != domain_->end(level);
                    ++it)
            {
                // including ghost parents
                it->fmm_mask(fmm_mask_idx_,mask_target_id,false);
                for ( int c = 0; c < it->num_children(); ++c )
                {
                    if ( it->child(c) && it->child(c)->fmm_mask(fmm_mask_idx_,mask_target_id) )
                    {
                        it->fmm_mask(fmm_mask_idx_,mask_target_id, true);
                        break;
                    }
                }

                // calculate load
                if  (it->fmm_mask(fmm_mask_idx_,mask_target_id))
                    for(std::size_t i = 0; i< it->influence_number(); ++i)
                    {
                        const auto inf=it->influence(i);
                        if (inf && inf->fmm_mask(fmm_mask_idx_,mask_source_id))
                            it->add_load(1.0 * _base_factor);
                    }
            }

        }
    }

    static void fmm_dry_init_base_level_masks(
            Domain* domain_,
            int base_level,
            bool non_leaf_as_source,
            int _load_factor=1,
            bool _neighbor_only=false)
    {
        int refinement_level = base_level-domain_->tree()->base_level();
        int fmm_mask_idx_ = refinement_level*2+non_leaf_as_source;

        if (non_leaf_as_source)
        {
            for (auto it = domain_->begin(base_level);
                    it != domain_->end(base_level); ++it)
            {
                if ( it->is_leaf() )
                {
                    it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Source, false);
                    it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Target, false);
                } else
                {
                    it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Source, true);
                    if (!it->is_correction())
                        it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Target, true);

                    if (!_neighbor_only)
                        it->add_load(it->influence_number() * _load_factor);

                    it->add_load(it->neighbor_number() * _load_factor);
                }
            }
        } else
        {
            for (auto it = domain_->begin(base_level);
                    it != domain_->end(base_level); ++it)
            {
                it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Source, true);

                if (!it->is_correction())
                    it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Target, true);

                if (!_neighbor_only)
                    it->add_load(it->influence_number() * _load_factor);

                it->add_load(it->neighbor_number() * _load_factor);
            }
        }

    }
};




using namespace domain;

template<class Setup>
class Fmm
{
public: //Ctor:
    static constexpr std::size_t Dim=Setup::Dim;

    using dims_t = types::vector_type<int,Dim>;
    using datablock_t  = typename Setup::datablock_t;
    using block_dsrp_t = typename datablock_t::block_descriptor_type;
    using domain_t = typename Setup::domain_t;
    using octant_t = typename domain_t::octant_t;
    using MASK_LIST = typename octant_t::MASK_LIST;

    //Fields:
    using fmm_s = typename Setup::fmm_s;
    using fmm_t = typename Setup::fmm_t;

    using convolution_t = fft::Convolution;


public:
    Fmm(int Nb)
    :lagrange_intrp(Nb),
    conv_( dims_t{{Nb,Nb,Nb}}, dims_t{{Nb,Nb, Nb}} )
    {
    }

    template< class Source, class Target, class Kernel >
    void apply(domain_t* domain_,
               Kernel* _kernel,
               int level,
               bool non_leaf_as_source=false,
               float_type add_with_scale = 1.0)
    {

        const float_type dx_base=domain_->dx_base();
        auto refinement_level = level-domain_->tree()->base_level();
        auto dx_level =  dx_base/std::pow(2, refinement_level);

        base_level_ = level;
        fmm_mask_idx_ = refinement_level*2+non_leaf_as_source;

        if (_kernel->neighbor_only())
        {
            //pcout<<"Integrating factor for level: "<< level << std::endl;
            fmm_init_zero<fmm_s>(domain_, MASK_LIST::Mask_FMM_Source);
            fmm_init_zero<fmm_t>(domain_, MASK_LIST::Mask_FMM_Target);
            fmm_init_copy<Source, fmm_s>(domain_);
            fmm_IF(domain_, _kernel);

            fmm_add_equal<Target, fmm_t>(domain_,add_with_scale);

            return;
        }
        pcout<<"FMM For Level "<< level << " Start ---------------------------"<<std::endl;
        // New logic using masks
        // 1. Give masks to the base level

        //// The straight forward way:
            // 1.1 if non_leaf_as_source is true then
            // non-leaf's mask_fmm_source is 1 and non-leaf's mask_fmm_target is 0,
            // while leaf's mask_fmm_source is 1 and leaf's mask_fmm_target is 1

            // 1.2 if non_leaf_as_source is false then
            // non-leaf's mask_fmm_source is 0 and non-leaf's mask_fmm_target is 1,
            // while leaf's mask_fmm_source is 1 and leaf's mask_fmm_target is 1

        //// !However! we do it a bit differently !
            // 1.1 if non_leaf_as_source is true then
            // non-leaf's mask_fmm_source is 1 and non-leaf's mask_fmm_target is 1,
            // while leaf's mask_fmm_source is 0 and leaf's mask_fmm_target is 0
            // BUT WITH MINUS SIGN!!!!!

            // 1.2 if non_leaf_as_source is false then
            // non-leaf's mask_fmm_source is 1 and non-leaf's mask_fmm_target is 1,
            // while leaf's mask_fmm_source is 1 and leaf's mask_fmm_target is 1
            // BUT WITH PLUS SIGN!!!!!

        // This way one has one time more calculaion for non-leaf's source but
        // one time fewer for leaf's leaf_source
        // and non-leaves can be much fewer

        //// Initialize Masks
        // done at master node


        ////Initialize for each fmm//zero ing all tree
        fmm_init_zero<fmm_s>(domain_, MASK_LIST::Mask_FMM_Source);
        fmm_init_zero<fmm_t>(domain_, MASK_LIST::Mask_FMM_Target);

        //// Copy to temporary variables // only the base level
        fmm_init_copy<Source, fmm_s>(domain_);

        timings_=Timings() ;

        domain_->client_communicator().barrier();
        //// Anterpolation
        pcout<<"FMM Antrp start" << std::endl;
        auto t0_anterp=clock_type::now();
        fmm_antrp(domain_);
        domain_->client_communicator().barrier();
        auto t1_anterp=clock_type::now();
        timings_.anterp=t1_anterp-t0_anterp;

        domain_->client_communicator().barrier();

        //// FMM influence list
        pcout<<"FMM Bx start" << std::endl;
        //fmm_Bx_itr_build(domain_, level);
        auto t0_bx=clock_type::now();
        fmm_Bx(domain_, _kernel, dx_level);
        domain_->client_communicator().barrier();
        auto t1_bx=clock_type::now();
        timings_.bx=t1_bx-t0_bx;

        domain_->client_communicator().barrier();

        //// Interpolation
        pcout<<"FMM INTRP start" << std::endl;
        auto t0_interp=clock_type::now();
        fmm_intrp(domain_);
        domain_->client_communicator().barrier();
        auto t1_interp=clock_type::now();

        timings_.interp=t1_interp-t0_interp;
        timings_.global=t1_interp-t0_anterp;

        //std::cout<<"FMM INTRP done" << std::endl;

        //// Copy back
        //if (!non_leaf_as_source)
        fmm_add_equal<Target, fmm_t>(domain_,add_with_scale);
        //else
        //fmm_minus_equal<Target, fmm_t>(domain_);

        //std::cout<<"Rank "<<world.rank() << " FFTW_count = ";
        //std::cout<<conv_.fft_count << std::endl;
        pcout<<"FMM For Level "<< level << " End -------------------------"<<std::endl;
    }

    auto initialize_upward_iterator(int level, domain_t* domain_,bool _upward)
    {
        std::vector<std::pair<octant_t*, int>> octants;
        for (auto it = domain_->begin(level); it != domain_->end(level); ++it)
        {
            int recv_m_send_count=domain_-> decomposition().client()->
                updownward_pass_mcount(*it,_upward, fmm_mask_idx_);

            octants.emplace_back(std::make_pair(*it,recv_m_send_count));
        }
        //Sends=10000, recv1-10000, no_communication=0
        //descending order
        std::sort(octants.begin(), octants.end(),[&](const auto& e0, const auto& e1)
                {return e0.second> e1.second;  });
        return octants;
    }

    template<class Kernel>
    void fmm_IF(domain_t* domain_,
                Kernel* _kernel)
    {
        std::vector<std::pair<octant_t*, int>> octants;

        for (auto it = domain_->begin(base_level_); it != domain_->end(base_level_); ++it)
        {
            bool _neighbor = true;
            if (!(it->data()) || !it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Target) )
                continue;

            int recv_m_send_count =
            domain_->decomposition().client()->template
            communicate_induced_fields_recv_m_send_count<fmm_t, fmm_t>(it, _neighbor, fmm_mask_idx_);

            octants.emplace_back(std::make_pair(*it,recv_m_send_count));
        }
        //Sends=10000, recv1-10000, no_communication=0
        std::sort(octants.begin(), octants.end(),[&](const auto& e0, const auto& e1)
                {return e0.second> e1.second;  });

        const bool start_communication = true;

        for (auto B_it=octants.begin(); B_it!=octants.end(); ++B_it)
        {
            auto it =B_it->first;
            bool _neighbor = true;
            if (!(it->data()) || !it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Target) )
                continue;

            if(it->locally_owned())
            {
                // no need to use dx and so dx_level=1.0
                compute_influence_field(&(*it), _kernel,
                                       0, 1.0, _neighbor);
            }

            //setup the tasks
            //no need to use dx and so dx_level=1.0
            domain_->decomposition().client()->template
                communicate_induced_fields<fmm_t, fmm_t>(&(*it),
                    this, _kernel,0,1.0,
                    _neighbor,
                    start_communication, fmm_mask_idx_);
        }

        mDuration_type time_communication_Bx;

        TIME_CODE(time_communication_Bx, SINGLE_ARG(
        domain_->decomposition().client()->template
            finish_induced_field_communication();
        ))
    }

    template<class Kernel>
    void fmm_Bx(domain_t* domain_,
                Kernel* _kernel,
                float_type dx_level)
    {
        std::vector<std::pair<octant_t*, int>> octants;
        for (int level=base_level_; level>=0; --level)
        {
            for (auto it = domain_->begin(level); it != domain_->end(level); ++it)
            {
                bool _neighbor = (level==base_level_)? true:false;
                if (!(it->data()) || !it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Target) )
                    continue;

                int recv_m_send_count =
                    domain_->decomposition().client()->template
                    communicate_induced_fields_recv_m_send_count<fmm_t, fmm_t>(it, _neighbor, fmm_mask_idx_);

                octants.emplace_back(std::make_pair(*it,recv_m_send_count));
            }
        }
        std::sort(octants.begin(), octants.end(),[&](const auto& e0, const auto& e1)
                {return e0.second> e1.second;  });

        const bool start_communication = false;
        bool combined_messages=false;

        for (auto B_it=octants.begin(); B_it!=octants.end(); ++B_it)
        {
            auto it =B_it->first;
            int level = it->level();

            bool _neighbor = (level==base_level_)? true:false;
            if (!(it->data()) ||
                    !it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Target) )
                continue;

            if(it->locally_owned())
            {
                compute_influence_field(&(*it), _kernel,
                                       base_level_-level, dx_level, _neighbor);
            }

            //setup the tasks
            
            domain_->decomposition().client()->template
                communicate_induced_fields<fmm_t, fmm_t>(&(*it),
                    this, _kernel,base_level_-level,dx_level,
                    _neighbor,
                    start_communication, fmm_mask_idx_);

            if(!combined_messages && B_it->second==0)
            {
                    domain_->decomposition().client()->template
                        combine_induced_field_messages<fmm_t, fmm_t>();
                    combined_messages=true;
            }
            if(combined_messages)
            {
                domain_->decomposition().client()->template
                    check_combined_induced_field_communication<fmm_t,fmm_t>(false);
            }
        
        }

        //Finish the communication
        if(combined_messages)
        {
            domain_->decomposition().client()->template
                check_combined_induced_field_communication<fmm_t,fmm_t>(true);
        }
    }


    template<class Kernel>
    void compute_influence_field(octant_t* it, Kernel* _kernel,
                                 int level_diff, float_type dx_level,
                                 bool neighbor)  noexcept
    {

        if (!(it->data()) ||
            !it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Target)) return;

        conv_.fft_backward_field_clean();

        if(neighbor)
        {
            for (int i=0; i<it->nNeighbors(); ++i)
            {
                auto n_s = it->neighbor(i);
                if (n_s && n_s->locally_owned()
                        && n_s->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Source))
                {
                    fmm_tt(n_s, it, _kernel, 0);
                }
            }
        }

        if (!_kernel->neighbor_only())
        {
            for (std::size_t i=0; i< it->influence_number(); ++i)
            {
                auto n_s = it->influence(i);
                if (n_s && n_s->locally_owned()
                        && n_s->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Source))
                {
                    fmm_tt(n_s, it, _kernel,level_diff);
                }
            }
        }

        const auto t_extent = it->data()->template get<fmm_t>().
                                real_block().extent();
        block_dsrp_t extractor(dims_t(0), t_extent);

        float_type _scale = (_kernel->neighbor_only()) ? 1.0:dx_level*dx_level;
        conv_.apply_backward(extractor,
                it->data()->template get<fmm_t>(),
                _scale);
    }

    template< class f1, class f2 >
    void fmm_add_equal(domain_t* domain_, float_type scale)
    {

        for (auto it = domain_->begin(base_level_);
                it != domain_->end(base_level_);
                ++it)
        {
            if(it->data() && it->locally_owned() &&
               it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Target))
            {
                it->data()->template get_linalg<f1>().get()->
                    cube_noalias_view() +=
                                it->data()->template get_linalg_data<f2>() * scale;
            }
        }
    }

    template< class f1, class f2 >
    void fmm_minus_equal(domain_t* domain_)
    {
        for (auto it = domain_->begin(base_level_);
                it != domain_->end(base_level_);
                ++it)
        {
            if(it->data() && it->locally_owned() &&
               it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Target))
            {

                it->data()->template get_linalg<f1>().get()->
                    cube_noalias_view() -=
                            it->data()->template get_linalg_data<f2>();
            }
        }

    }

    template< class field >
    void fmm_init_zero(domain_t* domain_, int mask_id)
    {
        for (int level=base_level_; level>=0; --level)
        {
            for (auto it = domain_->begin(level);
                    it != domain_->end(level);
                    ++it)
            {
                    if(it->data() && it->fmm_mask(fmm_mask_idx_,mask_id))
                    {
                        for(auto& e: it->data()->template get_data<field>())
                            e=0.0;
                    }
            }
        }
    }


    template< class from, class to >
    void fmm_init_copy(domain_t* domain_)
    {
        // Neglecting the data in the buffer

        for (auto it = domain_->begin(base_level_);
                it != domain_->end(base_level_);
                ++it)
        {
            if(it->data() && it->locally_owned() &&
               it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Source))
            {
                auto lin_data_1 = it->data()->template get_linalg_data<from>();
                auto lin_data_2 = it->data()->template get_linalg_data<to>();

                xt::noalias( view(lin_data_2,
                                    xt::range(1,-1),  xt::range(1,-1), xt::range(1,-1)) ) =
                 view(lin_data_1, xt::range(1,-1),  xt::range(1,-1), xt::range(1,-1));

                //it->data()->template get_linalg<to>().get()->
                //    cube_noalias_view() =
                //     it->data()->template get_linalg_data<from>() * 1.0;
            }
        }

    }

    void fmm_intrp(domain_t* domain_)
    {
        const int mask_id = MASK_LIST::Mask_FMM_Target;
        for (int level=1; level<base_level_; ++level)
        {
            domain_->decomposition().client()-> template
                    communicate_updownward_assign<fmm_t, fmm_t>(level,false,true,fmm_mask_idx_);

            for (auto it = domain_->begin(level);
                    it != domain_->end(level);
                    ++it)
            {
                if(it->data() && it->fmm_mask(fmm_mask_idx_,mask_id) )
                    lagrange_intrp.nli_intrp_node<fmm_t>(it, mask_id, fmm_mask_idx_);
            }
        }
    }

    void fmm_antrp(domain_t* domain_)
    {
        const int mask_id = MASK_LIST::Mask_FMM_Source;
        for (int level=base_level_-1; level>=0; --level)
        {
            for (auto it = domain_->begin(level);
                    it != domain_->end(level);
                    ++it)
            {
                if(it->data() && it->fmm_mask(fmm_mask_idx_,mask_id) )
                    lagrange_intrp.nli_antrp_node<fmm_s>(it, mask_id, fmm_mask_idx_);
            }

            domain_->decomposition().client()->
                template communicate_updownward_add<fmm_s, fmm_s>(level, true, true, fmm_mask_idx_);

            for (auto it = domain_->begin(level);
                    it != domain_->end(level);
                    ++it)
            {

                if (!it->locally_owned() && it->data() && it->data()->is_allocated())
                {
                    auto& cp2 = it ->data()->template get_linalg_data<fmm_s>();
                    cp2*=0.0;
                }
            }
        }
    }

    template<class Kernel>
    void fmm_tt(octant_t* o_s, octant_t* o_t, Kernel* _kernel,
                int level_diff)
    {
        const auto t0_fft=clock_type::now();

        const auto t_base = o_t->data()->template get<fmm_t>().
                                        real_block().base();
        const auto s_base = o_s->data()->template get<fmm_s>().
                                        real_block().base();

        // Get extent of Source region
        const auto s_extent = o_s->data()->template get<fmm_s>().
                                real_block().extent();
        const auto shift    = t_base - s_base;

        // Calculate the dimensions of the LGF to be allocated
        const auto base_lgf   = shift - (s_extent - 1);
        const auto extent_lgf = 2 * (s_extent) - 1;

        block_dsrp_t lgf_block(base_lgf, extent_lgf);

        conv_.apply_forward_add(lgf_block, _kernel, level_diff,
                o_s->data()->template get<fmm_s>());

        const auto t1_fft=clock_type::now();
        timings_.fftw+=t1_fft-t0_fft;
        timings_.fftw_count_max+=1;
    }

    struct Timings{

        mDuration_type                    global=mDuration_type(0);
        mDuration_type                    anterp=mDuration_type(0);
        mDuration_type                    bx=mDuration_type(0);
        mDuration_type                    interp=mDuration_type(0);
        mDuration_type                    fftw=mDuration_type(0);
        std::size_t                       fftw_count_max=0;
        std::size_t                       fftw_count_min=0;

        void accumulate(boost::mpi::communicator _comm) noexcept
        {
            Timings tlocal=*this;

            decltype(tlocal.global.count()) cglobal,canterp,cbx,cinterp,cfftw;
            std::size_t  cfftw_count_max,cfftw_count_min;
            auto comp=[&](const auto& v0, const auto& v1){return v0>v1? v0  :v1;};
            auto min_comp=[&](const auto& v0, const auto& v1){return v0>v1? v1  :v0;};
            boost::mpi::all_reduce(_comm,tlocal.global.count(), cglobal,comp);
            boost::mpi::all_reduce(_comm,tlocal.anterp.count(), canterp,comp);
            boost::mpi::all_reduce(_comm,tlocal.bx.count(), cbx,comp);
            boost::mpi::all_reduce(_comm,tlocal.interp.count(), cinterp,comp);
            boost::mpi::all_reduce(_comm,tlocal.fftw.count(), cfftw,comp);

            boost::mpi::all_reduce(_comm,tlocal.fftw_count_max, cfftw_count_min,min_comp);
            boost::mpi::all_reduce(_comm,tlocal.fftw_count_max, cfftw_count_max,comp);

            this->global=mDuration_type(cglobal);
            this->anterp=mDuration_type(canterp);
            this->interp=mDuration_type(cinterp);
            this->bx=mDuration_type(cbx);
            this->fftw=mDuration_type(cfftw);
            this->fftw_count_min=cfftw_count_min;
            this->fftw_count_max=cfftw_count_max;
        }
    };

    const auto& timings() const noexcept{return timings_;}
    auto& timings()noexcept{return timings_;}



public:
    Nli lagrange_intrp;
private:
    int fmm_mask_idx_;
    int base_level_;
    convolution_t            conv_;      ///< fft convolution
    parallel_ostream::ParallelOstream pcout=
        parallel_ostream::ParallelOstream(1);

private: //timings
    Timings timings_;

};

}

#endif //IBLGF_INCLUDED_FMM_HPP
