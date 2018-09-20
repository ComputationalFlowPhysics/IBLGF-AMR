#ifndef IBLGF_INCLUDED_POISSON_HPP
#define IBLGF_INCLUDED_POISSON_HPP

#include <iostream>
#include <vector>
#include <tuple>

#include <algorithm>
#include <list>
#include <vector>
#include <iostream>
#include <numeric>
#include <random>
#include <cmath>
#include <functional>
#include <cstring>
#include <fftw3.h>

// IBLGF-specific
#include <global.hpp>
#include <simulation.hpp>
#include <domain/domain.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <domain/octree/tree.hpp>
#include <post-processing/parallel_ostream.hpp>
#include <lgf/lgf.hpp>
#include <fmm/fmm.hpp>

#include<utilities/convolution.hpp>
#include<utilities/interpolation.hpp>

const int Dim = 3;

using namespace domain;
using namespace octree;
using namespace types;
using namespace dictionary;
using namespace fft;

struct PoissonProblem
{
    using vel_type    = vector_type<float_type, Dim>;
    using size_v_type = vector_type<int       , Dim>;

    //              name                type     lBuffer.  hBuffer
    make_field_type(phi_num         , float_type, 1,       1)
    make_field_type(phi_interp      , float_type, 1,       1)
    make_field_type(source          , float_type, 1,       1)
    make_field_type(source_orig     , float_type, 1,       1)
    make_field_type(lgf_field_lookup, float_type, 1,       1)
    make_field_type(phi_exact       , float_type, 1,       1)
    make_field_type(lgf             , float_type, 1,       1)
    make_field_type(error           , float_type, 1,       1)
    make_field_type(error2          , float_type, 1,       1)
    make_field_type(lapace_field    , float_type, 1,       1)
    make_field_type(lapace_error    , float_type, 1,       1)
    make_field_type(dummy_field     , float_type, 1,       1)
    make_field_type(bla_field,        float_type, 1,       1)
    make_field_type(fmm_field,        float_type, 0,       1)

    using datablock_t = DataBlock<
        Dim, node,
        phi_num,
        phi_interp,
        source,
        source_orig,
        lgf_field_lookup,
        phi_exact,
        error,
        error2,
        lapace_field,
        lapace_error,
        dummy_field, 
        fmm_field, 
        bla_field     >;

    using datablock_t_2 = DataBlock<Dim, node, lgf>;

    using block_descriptor_t = typename datablock_t::block_descriptor_type;
    using tree_t             = Tree<Dim,datablock_t>;
    using octant_t           = typename tree_t::octant_type;
    using coordinate_t       = typename datablock_t::coordinate_type;
    using domain_t           = domain::Domain<Dim,datablock_t>;
    using b_descriptor       = BlockDescriptor<int, Dim>;
    using base_t             = typename b_descriptor::base_t;;
    using extent_t           = typename b_descriptor::extent_t;;


    PoissonProblem(Dictionary* _d)
    :simulation_(_d->get_dictionary("simulation_parameters")),
     //FIXME: exntent+1
     conv(simulation_.domain_.block_extent()+2,simulation_.domain_.block_extent()+2),
     //conv(simulation_.domain_.block_extent(),simulation_.domain_.block_extent()),
     fmm_(simulation_.domain_.block_extent()[0]+1)
    {
        pcout << "\n Setup:  LGF PoissonProblem \n" << std::endl;
        pcout << "Simulation: \n" << simulation_    << std::endl;

        const float_type L = simulation_.dictionary_->
            template get_or<float_type>("L", 1);

        auto tmp = L / (simulation_.domain_.bounding_box().extent()-1);
        dx = tmp[0];
        this->initialize();
        ///inteprolation_test();
        simulation_.write("solution_init.vtk");
    }


    /*
     * It initializes the Poisson problem using a manufactured solutions.
     */
    void initialize()
    {
        auto center = (simulation_.domain_.bounding_box().max() -
                       simulation_.domain_.bounding_box().min()) / 2.0 +
                       simulation_.domain_.bounding_box().min();

        int nRef=4;
        for(int l=0;l<nRef;++l)
        {
            int count=0;
            for (auto it  = simulation_.domain_.begin_leafs();
                    it != simulation_.domain_.end_leafs(); ++it)
            {
                //if (count++ ==0)simulation_.domain_.refine(it);
                auto b=it->data()->descriptor();
                coordinate_t lower(2), upper(2);
                b.grow(lower, upper);
                if(b.is_inside( std::pow(2.0,l)*center ) 
                   && it->refinement_level()==l
                  )
                {
                    simulation_.domain_.refine(it);
                }
            }
        }


        for (int lt = simulation_.domain_.tree()->base_level(); 
                 lt < simulation_.domain_.tree()->depth(); ++lt)
        {
            for (auto it  = simulation_.domain_.begin(lt);
                      it != simulation_.domain_.end(lt); ++it)
            {
                if(it->data())
                {
                    for(auto& e: it->data()->get_data<source>())
                        e=0.0;
                    for(auto& e: it->data()->get_data<phi_num>())
                        e=0.0;
                }
            }
        }


        center+=0.5/std::pow(2,nRef);
        //center[0]+=0.5;
        std::cout<<"center: " <<center   <<std::endl;

        const float_type a  = 10.;
        const float_type a2 = a*a;

        for (auto it  = simulation_.domain_.begin_leafs();
                it != simulation_.domain_.end_leafs(); ++it)
        {



            auto dx_level =  dx/std::pow(2,it->refinement_level());
            auto scaling =  std::pow(2,it->refinement_level());

            // ijk-way of initializing
            auto base = it->data()->base();
            auto max  = it->data()->max();
            for (auto k = base[2]; k <= max[2]; ++k)
            {
                for (auto j = base[1]; j <= max[1]; ++j)
                {
                    for (auto i = base[0]; i <= max[0]; ++i)
                    {
                        it->data()->get<source>(i,j,k)  = 0.0;
                        it->data()->get<phi_num>(i,j,k) = 0.0;

                        // manufactured solution:
                        float_type x = static_cast<float_type>
                            (i-center[0]*scaling+0.5)*dx_level;
                        float_type y = static_cast<float_type>
                            (j-center[1]*scaling+0.5)*dx_level;
                        float_type z = static_cast<float_type>
                            (k-center[2]*scaling+0.5)*dx_level;
                        const auto x2 = x*x;
                        const auto y2 = y*y;
                        const auto z2 = z*z;


                        it->data()->get<source>(i,j,k) = 
                            a*std::exp(-a*(x2)-a*(y2)-a*(z2))*(-6.0)+
                            (a2)*(x2)*std::exp(-a*(x2)-a*(y2)-a*(z2))*4.0 +
                            (a2)*(y2)*std::exp(-a*(x2)-a*(y2)-a*(z2))*4.0+
                            (a2)*(z2)*std::exp(-a*(x2)-a*(y2)-a*(z2))*4.0;

                        it->data()->get<source_orig>(i,j,k) = it->data()->get<source>(i,j,k);

                        it->data()->get<phi_exact>(i,j,k) =
                            std::exp((-a*x2 - a*y2 - a*z2));
                    }
                }
            }
        }
    }


    //Sipe check before 


    void neighbor_test()
    {

        int count=0;
        for (auto it  = simulation_.domain_.begin_leafs();    
                it != simulation_.domain_.end_leafs(); ++it)
        {
            coordinate_t direction(0);
            direction[0]=+1;
            auto nn=it->vertex_neighbor(direction);
            if(nn!=nullptr)
            {

                std::ofstream ofs("point_"+std::to_string(count)+".txt");
                std::ofstream ofs1("nn_"+std::to_string(count)+".txt");
                ofs<<it->global_coordinate()<<std::endl;
                ofs1<<nn->global_coordinate()<<std::endl;
                ++count;
            }
            else
            {
                std::cout<<"could not find neighbors"<<std::endl;
            }
        }
    }

  
    void neighborhood_test()
    {

        int count=0;
        for (auto it  = simulation_.domain_.begin_leafs();
                it != simulation_.domain_.end_leafs(); ++it)
        {
            coordinate_t lowBuffer(1);
            coordinate_t highBuffer(2);
            auto neighborhood = it->get_level_neighborhood(lowBuffer, highBuffer); 


            if(neighborhood.size()!=0)
            {
                std::ofstream ofs("point_nh__"+std::to_string(count)+".txt");
                ofs<<it->global_coordinate()<<std::endl;
                std::ofstream ofs1("nh_"+std::to_string(count)+".txt");
                for(auto& e:neighborhood)
                {
                    ofs1<<e->global_coordinate()<<std::endl;
                }
                count++;
            }
            else
            {
                std::cout<<"empty neighborhood"<<std::endl;
            }
        }
    }

    void simple_lapace_fd()
    {
        //Only in interior for simplicity:
        for (auto it  = simulation_.domain_.begin_leafs();
                  it != simulation_.domain_.end_leafs(); ++it)
        {

            auto base = it->data()->base();
            auto max  = it->data()->max();
            for (auto k = base[2]+1; k < max[2]; ++k)
            {
                for (auto j = base[1]+1; j < max[1]; ++j)
                {
                    for (auto i = base[0]+1; i < max[0]; ++i)
                    {
                        it->data()->get<lapace_field>(i,j,k) =
                            -6.0*it->data()->get<phi_num>(i,j,k)+
                                 it->data()->get<phi_num>(i+1,j,k)+
                                 it->data()->get<phi_num>(i-1,j,k)+
                                 it->data()->get<phi_num>(i,j+1,k)+
                                 it->data()->get<phi_num>(i,j-1,k)+
                                 it->data()->get<phi_num>(i,j,k+1)+
                                 it->data()->get<phi_num>(i,j,k-1);
                        it->data()->get<lapace_field>(i,j,k)/=dx*dx;
                        it->data()->get<lapace_error>(i,j,k)=
                            std::fabs(it->data()->get<lapace_field>(i,j,k)-
                                      it->data()->get<source>(i,j,k));
                    }
                }
            }
        }
    }

    void solve_singleLevel()
    {
        // allocate lgf
        std::vector<float_type> lgf;
        for (auto it_i  = simulation_.domain_.begin_leafs();
             it_i != simulation_.domain_.end_leafs(); ++it_i)
        {
            const auto ibase= it_i->data()->base();

            for (auto it_j  = simulation_.domain_.begin_leafs();
                 it_j != simulation_.domain_.end_leafs(); ++it_j)
            {

                const auto jbase   = it_j->data()->base();
                const auto jextent = it_j->data()->extent();
                const auto shift   = ibase - jbase;

                const auto base_lgf   = shift - (jextent - 1);
                const auto extent_lgf = 2 * (jextent) - 1;

                lgf_.get_subblock(block_descriptor_t(base_lgf,
                                                     extent_lgf), lgf);

                conv.execute(lgf, it_j->data()->get_data<source>());
                block_descriptor_t extractor(jbase, jextent);
                conv.add_solution(extractor,
                                  it_i->data()->get<phi_num>(), dx*dx);
            }
        }

        //simple_lapace_fd();
        compute_errors();
        pcout << "Writing solution " << std::endl;
        simulation_.write("solution.vtk");
    }

   
    void solve()
    {
        // allocate lgf
        std::vector<float_type> lgf;

        //Coarsification:
        pcout<<"coarsification "<<std::endl;
        for (int ls = simulation_.domain_.tree()->depth()-2;
                 ls >= simulation_.domain_.tree()->base_level(); --ls)
        {
            for (auto it_s  = simulation_.domain_.begin(ls);
                      it_s != simulation_.domain_.end(ls); ++it_s)
            {
                //this->coarsify(it_s);
                this->coarsify_avg(it_s);
            }
        }
        //this->scale_bcPlane();

        //Level-Interactions
        pcout<<"Level interactions "<<std::endl;
        for (int l  = simulation_.domain_.tree()->base_level();
                 l < simulation_.domain_.tree()->depth(); ++l)
        {
            auto target = 0;
            for (auto it_t  = simulation_.domain_.begin(l);
                      it_t != simulation_.domain_.end(l); ++it_t)
            {

                auto refinement_level = it_t->refinement_level();
                auto dx_level =  dx/std::pow(2,refinement_level);
                //std::cout<<"target :"<<it_t->data()->descriptor()<<std::endl;

                for (auto it_s  = simulation_.domain_.begin(l);
                          it_s != simulation_.domain_.end(l); ++it_s)
                {
                    if( !(it_s->is_leaf()) && !(it_t->is_leaf()) )
                    {
                         continue;
                    }
                    //std::cout<<"source :"<<it_s->data()->descriptor()<<std::endl;

                    // FIXME: extent+1
                    //const auto t_base = it_t->data()->base()-1;
                    const auto t_base = it_t->data()->get<phi_num>().
                                            real_block().base();
                    const auto s_base = it_s->data()->get<source>().
                                            real_block().base();
                    
                    // Get extent of source region
                    // FIXME: extent+1
                    //const auto s_extent = it_s->data()->extent()+2;
                    const auto s_extent = it_s->data()->get<source>().
                                            real_block().extent();
                    const auto shift    = t_base - s_base;
                    
                    // Calculate the dimensions of the LGF to be allocated
                    const auto base_lgf   = shift - (s_extent - 1);
                    const auto extent_lgf = 2 * (s_extent) - 1;
                    
                    // Calculate the LGF
                    lgf_.get_subblock(block_descriptor_t(base_lgf,
                                                         extent_lgf), lgf);
                    // Perform convolution
                    conv.execute_field(lgf, it_s->data()->get<source>());
                    
                    // Extract the solution
                    block_descriptor_t extractor(s_base, s_extent);
                    conv.add_solution(extractor,
                                      it_t->data()->get<phi_num>(),
                                      dx_level*dx_level);
                }
                //std::cout<<std::endl;
                target++;
            }
        }


        // Interpolation
        std::cout<<"Interpolation"<<std::endl;
        for (int lt = simulation_.domain_.tree()->base_level(); 
                 lt < simulation_.domain_.tree()->depth(); ++lt)
        {
            //simulation_.domain_.exchange_level_buffers(lt);
            for (auto it_t  = simulation_.domain_.begin(lt);
                      it_t != simulation_.domain_.end(lt); ++it_t)
            {
                if(it_t->is_leaf()) continue;
                this->template interpolate_cc<phi_num>(*it_t);
                //this->template interpolate_avg<phi_num>(*it_t);
            }
        }
                
        //simple_lapace_fd();
        compute_errors();
        pcout << "Writing solution " << std::endl;
        simulation_.write("solution.vtk");
    }

    template<template<std::size_t>class Field >
    void interpolate_avg(const octant_t* _b_parent)
    {
        
        //std::cout<<"parent: "<<_b_parent->data()->descriptor()<<std::endl;
        //interpolation 
        for (int i = 0; i < _b_parent->num_children(); ++i)
        {
            auto child = _b_parent->child(i);
            if(child==nullptr) continue;
            block_descriptor_t child_view =  
                child->data()->template get<phi_num>();
            block_descriptor_t parent_view = child->data()->template get<phi_num>();

            for (auto kc  = child_view.base()[2];
                      kc <= child_view.max()[2]; ++kc)
            {
                for (auto jc  = child_view.base()[1];
                          jc  <= child_view.max()[1]; ++jc)
                {
                    for (auto ic = child_view.base()[0];
                              ic <= child_view.max()[0]; ++ic)
                    {
                        int min_x= ic/2; int min_y= jc/2; int min_z= kc/2;

                        float_type interp= 
                            _b_parent->data()->template get<Field>(min_x, min_y,min_z);

                        child->data()->template get<Field>(ic,jc,kc) += interp;
                        child->data()->template get<phi_interp>(ic,jc,kc) = interp;
                    }
                }
            }
        }
    }

    template<class Block_it>
    void coarsify_avg(const Block_it& _parent)
    {
        //Note: only works for an even number of nodes 
        //      Base of all children is even, so now buffer is needed
        auto parent = _parent;
        if(parent->is_leaf())return;

        for (int i = 0; i < parent->num_children(); ++i)
        {
            auto child = parent->child(i);
            auto child_view= child->data()->descriptor();
            if(child==nullptr) continue;

            for (auto kc  = child_view.base()[2];
                    kc <= child_view.max()[2]; kc+=2)
            {
                for (auto jc  = child_view.base()[1];
                        jc  <= child_view.max()[1]; jc+=2)
                {
                    for (auto ic = child_view.base()[0];
                            ic <= child_view.max()[0]; ic+=2)
                    {
                        const float_type avg=1./8*(
                            child->data()->template get<source>(ic,  jc,  kc)+
                            child->data()->template get<source>(ic+1,jc,  kc)+
                            child->data()->template get<source>(ic,  jc+1,kc)+
                            child->data()->template get<source>(ic+1,jc+1,kc)+
                            child->data()->template get<source>(ic,  jc,  kc+1)+
                            child->data()->template get<source>(ic+1,jc,  kc+1)+
                            child->data()->template get<source>(ic,  jc+1,kc+1)+
                            child->data()->template get<source>(ic+1,jc+1,kc+1));
                        int ip=ic/2;
                        int jp=jc/2;
                        int kp=kc/2;
                        parent->data()->template get<source>(ip,jp,kp) =avg;
                    }
                }
            }
        }
    }

    void inteprolation_test()
    {
        auto center = (simulation_.domain_.bounding_box().max() -
                simulation_.domain_.bounding_box().min()) / 2.0 +
            simulation_.domain_.bounding_box().min();


        const float_type a  = 10.;
        const float_type a2 = a*a;
        //cooarsify exact solution 
        for (int ls = simulation_.domain_.tree()->depth()-2;
                ls >= simulation_.domain_.tree()->base_level(); --ls)
        {
            for (auto it  = simulation_.domain_.begin(ls);
                      it != simulation_.domain_.end(ls); ++it)
            {

                auto dx_level =  dx/std::pow(2,it->refinement_level());
                auto scaling =  std::pow(2,it->refinement_level());

                // ijk-way of initializing
                auto base = it->data()->base();
                auto max  = it->data()->max();
                for (auto k = base[2]-1; k <= max[2]+1; ++k)
                {
                    for (auto j = base[1]-1; j <= max[1]+1; ++j)
                    {
                        for (auto i = base[0]-1; i <= max[0]+1; ++i)
                        {
                            it->data()->get<source>(i,j,k)  = 0.0;
                            it->data()->get<phi_num>(i,j,k) = 0.0;

                            // manufactured solution:
                            float_type x = static_cast<float_type>
                                (i-center[0]*scaling+0.5)*dx_level;
                            float_type y = static_cast<float_type>
                                (j-center[1]*scaling+0.5)*dx_level;
                            float_type z = static_cast<float_type>
                                (k-center[2]*scaling+0.5)*dx_level;
                            const auto x2 = x*x;
                            const auto y2 = y*y;
                            const auto z2 = z*z;


                            it->data()->get<bla_field>(i,j,k) =
                                std::exp((-a*x2 - a*y2 - a*z2));
                        }
                    }
                }
            }
        }

        //interpolation back again
        
        // Interpolation
        std::cout<<"Interpolation"<<std::endl;
        for (int lt = simulation_.domain_.tree()->base_level(); 
                 lt < simulation_.domain_.tree()->depth(); ++lt)
        {
            //simulation_.domain_.exchange_level_buffers(lt);
            for (auto it_t  = simulation_.domain_.begin(lt);
                      it_t != simulation_.domain_.end(lt); ++it_t)
            {
                if(it_t->is_leaf()) continue;
                this->template interpolate_cc<bla_field>(*it_t);
            }
        }
    }



    template<template<std::size_t>class Field >
    void interpolate_cc(const octant_t* _b_parent)
    {
        
        static int c=0;
        
        //std::cout<<"parent: "<<_b_parent->data()->descriptor()<<std::endl;
        //interpolation 
        for (int i = 0; i < _b_parent->num_children(); ++i)
        {
            auto child = _b_parent->child(i);
            if(child==nullptr) continue;
            block_descriptor_t child_view =  
                child->data()->template get<Field>().real_block();
                //child->data()->descriptor();

            block_descriptor_t parent_view = _b_parent->data()->template get<Field>().real_block();
            auto parent_base= parent_view.base();
            auto parent_max= parent_view.max();
            //std::cout<<"child:  "<<child_view<<std::endl;

            
            int count=0;
            for (auto kc  = child_view.base()[2];
                      kc <= child_view.max()[2]; ++kc)
            {
                for (auto jc  = child_view.base()[1];
                          jc  <= child_view.max()[1]; ++jc)
                {
                    for (auto ic = child_view.base()[0];
                              ic <= child_view.max()[0]; ++ic)
                    {
                        //int min_x= ic/2; int min_y= jc/2; int min_z= kc/2;
                        int min_x= (ic+1)/2-1;
                        int min_y= (jc+1)/2-1;
                        int min_z= (kc+1)/2-1;
                        //if(min_x+2>=parent_max[0]) min_x-=1;
                        //if(min_y+2>=parent_max[1]) min_y-=1;
                        //if(min_z+2>=parent_max[2]) min_z-=1;
                        const float_type x= (ic-0.5)/2.0; 
                        const float_type y= (jc-0.5)/2.0; 
                        const float_type z= (kc-0.5)/2.0;


                        //std::ofstream ofsi("icube_"+std::to_string(c)+"_"+std::to_string(count)+".txt");
                        //std::ofstream ofs_pti("ipt_"+std::to_string(c)+"_"+std::to_string(count)+".txt");
                        //ofs_pti<<x<<" "<<y<<" "<<z<<std::endl;

                        //std::ofstream ofs("cube_"+std::to_string(c)+"_"+std::to_string(count)+".txt");
                        //std::ofstream ofs_pt("pt_"+std::to_string(c)+"_"+std::to_string(count)+".txt");
                        //ofs_pt<<ic/2.+0.25<<" "<<jc/2.+0.25<<" "<<kc/2.+0.25<<std::endl;


                        //unsigned int N=3;
                        //for (unsigned int i=0; i<N; ++i)
                        //{
                        //    for (unsigned int j=0; j<N; ++j)
                        //    {
                        //        for (unsigned int k=0; k<N; ++k)
                        //        {
                        //            ofs
                        //                <<min_x+0.5+i<<" "
                        //                <<min_y+0.5+j<<" "
                        //                <<min_z+0.5+k<<" "
                        //                <<std::endl;

                        //            ofsi<<min_x+i<<" "<<min_y+j<<" "<<min_z+k<<std::endl;
                        //        }
                        //    }
                        //}

                        //float_type interp= 
                        //interpolation::interpolate(min_x, min_y, min_z, x, y, z, 
                        //_b_parent->data()->template get<Field>());
                        //child->data()->template get<Field>(ic,jc,kc) +=interp;
                        //std::cout
                        //    <<min_x-parent_view.base()[0]<<" "
                        //    <<min_y-parent_view.base()[1]<<" "
                        //    <<min_z-parent_view.base()[2]<<" "
                        //    <<std::endl;
                            
                        
                        float_type interp= 
                            interpolation::interpolate(min_x, min_y, min_z, x, y, z, 
                                    _b_parent->data()->template get<Field>(), count++);

                        child->data()->template get<Field>(ic,jc,kc) += interp;
                        child->data()->template get<phi_interp>(ic,jc,kc) = interp;
                    }
                }
            }
            //throw std::runtime_error("bla");
        }
            //std::cout<<std::endl;
            ++c;
    }

    /*
     * Interpolate a given field from corser to finer level.
     * Note: maximum jump allowed is one level.
     */
    template<template<std::size_t>class Field >
    void interpolate(const octant_t* _b_parent)
    {
        
        //std::cout<<"parent: "<<_b_parent->data()->descriptor()<<std::endl;
        //interpolation 
        for (int i = 0; i < _b_parent->num_children(); ++i)
        {
            auto child = _b_parent->child(i);
            if(child==nullptr) continue;
            block_descriptor_t child_view =  
                child->data()->template get<phi_num>().real_block();

            //std::cout<<"child:  "<<child_view<<std::endl;

            
            for (auto kc  = child_view.base()[2];
                      kc <= child_view.max()[2]; ++kc)
            {
                for (auto jc  = child_view.base()[1];
                          jc  <= child_view.max()[1]; ++jc)
                {
                    for (auto ic = child_view.base()[0];
                              ic <= child_view.max()[0]; ++ic)
                    {
                        int min_x= ic/2; int min_y= jc/2; int min_z= kc/2;
                        //if(ic==child_view.max()[0]) min_x-=1; 
                        //if(jc==child_view.max()[1]) min_y-=1;
                        //if(kc==child_view.max()[2]) min_z-=1;
                        const float_type x= ic/2.0; 
                        const float_type y= jc/2.0; 
                        const float_type z= kc/2.0;

                        //float_type interp= 
                        //interpolation::interpolate(min_x, min_y, min_z, x, y, z, 
                        //_b_parent->data()->template get<Field>());
                        //child->data()->template get<Field>(ic,jc,kc) +=interp;
                        
                        float_type interp= 
                            interpolation::interpolate(min_x, min_y, min_z, x, y, z, 
                                    _b_parent->data()->template get<Field>());

                        child->data()->template get<Field>(ic,jc,kc) += interp;
                        child->data()->template get<phi_interp>(ic,jc,kc) = interp;

                    }
                }
            }
        }
            //std::cout<<std::endl;
    }

    template<class Block_it>
    void interpolate_level_interface(const Block_it* _b, 
                                     block_descriptor_t view)
    {
        for(auto kc  = view.base()[2];
                 kc <= view.max()[2]; ++kc)
        {
            for (auto jc  = view.base()[1];
                      jc  <= view.max()[1]; ++jc)
            {
                for (auto ic = view.base()[0];
                          ic <= view.max()[0]; ++ic)
                {
                    //Snap into parent grid:
                    const int min_x = 2*((ic)/2); 
                    const int min_y = 2*((jc)/2); 
                    const int min_z = 2*((kc)/2);
                    const float_type x= ic;
                    const float_type y= jc;
                    const float_type z= kc;
                    auto interp= 
                        interpolation::interpolate(min_x, min_y, min_z, x, y, z, 
                                _b->data()->template get<phi_num>(),2);
                    _b->data()->template get<phi_num>(ic,jc,kc)=interp;
                }
            }
        }
    } 
  
    /*
     * Coarsify given field from finer to coarser level.
     * Note: maximum jump allowed is one level.
     */
    template<class Block_it>
    void coarsify(const Block_it& _child)
    {
        auto child  = _child;
        auto parent = child->parent();

        auto child_view= child->data()->descriptor();
        auto parent_view= child_view;
        parent_view.level_scale(parent->refinement_level());

        auto kp = parent_view.base()[2];
        for (auto kc  = child_view.base()[2];
                  kc <= child_view.max()[2]; kc+=2)
        {
            auto jp = parent_view.base()[1];
            for (auto jc  = child_view.base()[1];
                      jc  <= child_view.max()[1]; jc+=2)
            {
                auto ip = parent_view.base()[0];
                for (auto ic = child_view.base()[0];
                          ic <= child_view.max()[0]; ic+=2)
                {
                    parent->data()->template get<source>(ip,jp,kp) =
                        child->data()->template get<source>(ic,jc,kc);
                    ip++;
                }
                jp++;
            }
            kp++;
        }
    }


    void scale_bcPlane()
    {
        auto lb=simulation_.domain_.tree()->base_level();
        std::cout<<"Base level : "<<lb-1<<std::endl;
        auto ld=simulation_.domain_.tree()->depth();

        for(int ll=lb; ll<ld;++ll)
        {
            auto bb= simulation_.domain_.get_level_bb(ll  ) ;

            std::cout<<"bounding box for level: "<<ll<<" "<<bb<<std::endl;
            //int dir=1;
            for(int dir=0; dir<=1; ++dir)
            {
                for(int d=0; d<Dim; ++d)
                {
                    auto bc=bb.bcPlane(d,dir);
                    //std::cout<<"bcPlane: "<<bb.bcPlane(d, dir)<<std::endl;
                    for (auto it  = simulation_.domain_.begin(ll);
                            it != simulation_.domain_.end(ll); ++it)
                    {
                        auto overlap=bc;
                        bc.level()=it->refinement_level();

                        auto scaling =  std::pow(2,it->refinement_level());
                        if(bc.overlap(it->data()->descriptor(),overlap, it->refinement_level()))
                        {
                            auto base=overlap.base();
                            auto max=overlap.max();
                            std::cout<<"Overlap: "<<overlap<<std::endl;
                            for (auto k = base[2]; k <= max[2]; ++k)
                            {
                                for (auto j = base[1]; j <= max[1]; ++j)
                                {
                                    for (auto i = base[0]; i <= max[0]; ++i)
                                    {
                                        //Lower left corner 
                                        if(dir==0)
                                            it->data()->get<source>(i,j,k)/=1.0;

                                        //Upper left corner 
                                        if(dir==1 && ll==lb+1)
                                        {
                                            it->data()->get<source>(i,j,k)*=1.0;
                                            //it->data()->get<source>(i,j,k)=it->data()->get<source_orig>(i,j,k);
                                            //it->data()->get<bla_field>(i,j,k)+=1.0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /*
     * Calculate the L2 and LInf errors.
     */
    void compute_errors()
    {
        auto L2   = 0.;
        auto LInf = -1.0;
        int count=0;

        for (auto it_t  = simulation_.domain_.begin_leafs();
             it_t != simulation_.domain_.end_leafs(); ++it_t)
        {

            auto base = it_t->data()->base();
            auto max  = it_t->data()->max();
            for (auto k = base[2]; k <= max[2]; ++k)
            {
                for (auto j = base[1]; j <= max[1]; ++j)
                {
                    for (auto i = base[0]; i <= max[0]; ++i)
                    {
                        it_t->data()->get<error>(i,j,k) = std::fabs(
                                it_t->data()->get<phi_num>(i,j,k) -
                                it_t->data()->get<phi_exact>(i,j,k));

                        it_t->data()->get<error2>(i,j,k) =
                            it_t->data()->get<error>(i,j,k) *
                            it_t->data()->get<error>(i,j,k);

                        L2 += it_t->data()->get<error2>(i,j,k);

                        if ( it_t->data()->get<error>(i,j,k) > LInf)
                        {
                            LInf = it_t->data()->get<error>(i,j,k);
                        }
                        ++count;
                    }
                }
            }
        }
        pcout << "L2   = " << std::sqrt(L2/count)<< std::endl;
        pcout << "LInf = " << LInf << std::endl;
    }

private:

    Simulation<domain_t>              simulation_;
    parallel_ostream::ParallelOstream pcout;
    lgf::LGF<lgf::Lookup>             lgf_;
    Convolution                       conv;
    fmm::Fmm                          fmm_;
    float_type                        dx;
};


#endif // IBLGF_INCLUDED_POISSON_HPP
