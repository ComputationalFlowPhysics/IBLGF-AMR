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

#ifndef IBLGF_MERGE_TREES_HPP
#define IBLGF_MERGE_TREES_HPP
// #include <cmath>
namespace iblgf
{
using namespace domain;
template<class Setup>
class MergeTrees
{
  public:
    // using tree_t = typename Setup::tree_t;
    using key_id_t= typename Setup::key_id_t;
    // using tree_t = typename Setup::domain_t::tree_t;

    MergeTrees() = default;
    MergeTrees(const MergeTrees&) = delete;
    MergeTrees(MergeTrees&&) = default;
    MergeTrees& operator=(const MergeTrees&) & = delete;
    MergeTrees& operator=(MergeTrees&&) & = default;
    ~MergeTrees() = default;

  public:
    static void merge()
    {
        boost::mpi::communicator world;
        if (world.rank() != 0) return;
        std::cout << "Merging trees..." << std::endl;
    }
    template<class tree_t>
    static void merger( tree_t& t1,  tree_t& t2,std::vector<key_id_t>& octs, std::vector<int>& leafs)
    {

        boost::mpi::communicator world;
        // std::cout << "Merging trees..." << std::endl;
        octs.clear();
        leafs.clear();
        if(world.rank()!=0) return;
        std::cout << "Merging trees..." << std::endl;
        for (auto it1 = t1->begin(); it1 != t1->end(); ++it1)
        {
            if(!it1->has_data()) continue;
            auto it2= t2->find_octant(it1->key());
            if(it2&&(it2->has_data()&&it1->has_data()))
            {
                
                int is_leaf=0;
                if((it1->is_correction() || it2->is_correction())&&it1->refinement_level()>0)
                {
                    // if(it1->is_correction()&& it2->is_correction()) continue;
                    // if(it1->is_correction()|| it2->is_correction()) continue;
                    // if(it1->refinement_level()>0&&!(it1->is_correction()|| it2->is_correction())) continue;
                    is_leaf=0;
                }
                // if(it1->is_leaf_search()&& it2->is_leaf_search())
                // {
                //     is_leaf=1;
                // }
                // else if(it1->is_leaf_search()|| it2->is_leaf_search())
                // {
                //     if(it1->refinement_level()>0&&!(it1->is_correction()|| it2->is_correction())) is_leaf=0;
                //     else
                //     {
                //         is_leaf=1;
                //     }
                // }
                else if (it1->is_leaf()|| it2->is_leaf())
                {
                    is_leaf=1;
                }
                // else if(it1->refinement_level()==0 && (it1->is_leaf() || it2->is_leaf()))
                // {
                //     is_leaf=1;
                // }
                // else if (it1->is_leaf() && !it2->is_correction())
                // {
                //     is_leaf=1;
                // }
                // else if (it2->is_leaf() && !it1->is_correction())
                // {
                //     is_leaf=1;
                // }
                


                // if((it1->is_leaf()|| it2->is_leaf()))
                // {
                //     if(it1->refinement_level()>0&&(it1->is_correction()|| it2->is_correction())) continue;
                //     // if(it1->is_leaf() && it2->is_leaf()) is_leaf=1;
                //     is_leaf=0;
                //     // if(it1->refinement_level()>=0&&(it1->is_correction()|| it2->is_correction())) is_leaf=false;
                //     // std::cout << "Common Leaf: " << it1->tree_coordinate() << std::endl;
                //     // std::cout<<"level: "<<it1->refinement_level()<<std::endl;   
                // }
                octs.emplace_back(it2->key().id());
                leafs.emplace_back(is_leaf);
            }
        }
    }

    template<class tree_t>
    static void get_ranks(tree_t& t1,std::vector<key_id_t>& octs, std::vector<int>& leafs)
    {
        boost::mpi::communicator world;
        // server contains all octs and leafs and client has empty vector. use function to get ranks from server and fill in vectors
        std::vector<std::vector<key_id_t>> send_octs(world.size()); //outer vector is rank
        std::vector<std::vector<int>> send_leafs(world.size());

        if(world.rank()==0)
        {
            std::cout<< "Getting ranks for octs..." << std::endl;
            for (auto& k : octs)
            {
                auto nn = t1->find_octant(k);
                if (!nn) continue;
                if (nn->rank() > 0)
                {
                    send_octs[nn->rank()].emplace_back(k);
                    send_leafs[nn->rank()].emplace_back(leafs[&k - &octs[0]]);
                }
            }

            world.barrier();
            for (int i = 1; i < world.size(); ++i)
            {
                world.send(i,i*2, send_octs[i]);
                world.send(i,i*2+1, send_leafs[i]);
            }
            world.barrier();
        }
        else
        {
            octs.clear();
            leafs.clear();
            world.barrier();
            world.recv(0, world.rank()*2, octs);
            world.recv(0, world.rank()*2+1, leafs);
            world.barrier();
        }

    }

    template<class tree_t,class key_t>
    static void getDelOcts(tree_t& t1, tree_t& t2, std::vector<key_t>& keys_to_del,std::vector<int>& level_change)
    {
        boost::mpi::communicator world;
        // get octs from t1 which are not in t2 and have data so that we can delete them in next step to adapt tree
        keys_to_del.clear();
        level_change.clear();
        if(world.rank()!=0) return;
        std::cout << "Getting octs to delete..." << std::endl;
        for (auto it1 = t1->begin(); it1 != t1->end(); ++it1)
        {
            if(!it1->has_data()) continue;
            if(it1->refinement_level()<0) continue; //
            if(!it1->is_leaf()&& it1->is_correction())
            {
                keys_to_del.emplace_back(it1->key().id());
                level_change.emplace_back(-1);
                continue;
            }
            if(!it1->is_leaf()) continue;
            // if(it1->refinement_level()<=0) continue; // only leafs with refinement level >=0 need to be adapted
            auto it2= t2->find_octant(it1->key());
            // if(!it2 || !it2->has_data()||(it2->is_correction() && it2->refinement_level()>0)||(it1->is_correction()&&it1->refinement_level()==0))
            // if(!it2 || !it2->has_data()||(it2->is_correction())||(it1->is_correction()&&it1->refinement_level()==0))
            if(!it2 || !it2->has_data()||!it2->physical()||(it2->is_correction()))
            {
                keys_to_del.emplace_back(it1->key().id());
                level_change.emplace_back(-1);
            }
            else if(it1->refinement_level()==0&&(it1->is_correction() && !it2->is_leaf()))
            {
                keys_to_del.emplace_back(it1->key().id());
                level_change.emplace_back(-1);
            }

       }
    }

    template<class tree_t, class key_t>
    static void get_level_changes(tree_t& ref_tree, tree_t& old_tree, int ref_level,
                                  std::vector<key_t>& octs, std::vector<int>& level_change)
    {
        boost::mpi::communicator world;
        // adapt old_tree to ref_tree 
        // need to add octs not in old_tree and delete blocks not in ref_tree from certain level
        octs.clear();
        level_change.clear();
        if(world.rank()!=0) return;
        std::cout << "Getting level changes..." << std::endl;
        int old_level=ref_level+old_tree->base_level();
        int new_level=ref_level+ref_tree->base_level();
        // first gets octs in old_tree which of not in ref_tree
        for(auto it1=old_tree->begin(old_level); it1!= old_tree->end(old_level); ++it1)
        {
            if(!it1->has_data()) continue;
            if(!it1->is_leaf()) continue;
            auto it2= ref_tree->find_octant(it1->key());
            if(!it2 || !it2->has_data()||!it2->physical()||(it2->is_correction()))
            {
                octs.emplace_back(it1->key().id());
                level_change.emplace_back(-1);
            }
        }

        // second gets octs in ref_tree which are not in old_tree
        for(auto it1=ref_tree->begin(new_level); it1!= ref_tree->end(new_level); ++it1)
        {
            if(!it1->has_data()) continue;
            if(!it1->is_leaf()) continue;
            auto it2= old_tree->find_octant(it1->key());
            if(!it2 || !it2->has_data()||!it2->physical()||(it2->is_correction()))
            {
                auto ptag= it1->key().parent();
                auto tt= old_tree->find_octant(ptag);
                while(!tt)
                {
                    ptag= ptag.parent();
                    tt= old_tree->find_octant(ptag);
                }
                
                octs.emplace_back(tt->key().parent().id());
                level_change.emplace_back(1);
            }
        }
  

    }

    template<class tree_t, class key_t>
    static void symgrid(tree_t& ref_tree, int ref_level, std::vector<key_t>& octs, std::vector<int>& level_change, const bool toPrint=false)
    {
        boost::mpi::communicator world;
        // adapt old_tree to ref_tree 
        // need to add octs not in old_tree and delete blocks not in ref_tree from certain level
        octs.clear();
        level_change.clear();
        if(world.rank()!=0) return;
        std::cout << "Getting symmetry changes..." << std::endl;
        int new_level=ref_level+ref_tree->base_level();
        for(auto it1=ref_tree->begin(new_level); it1!= ref_tree->end(new_level); ++it1)
        {
            if(!it1->has_data()) continue;
            if(!it1->is_leaf()||it1->is_correction()) continue;
            // check if opposite block is also a leaf. if its not add it to be deleted
            auto coord= it1->tree_coordinate();
            auto key= it1->key();
            auto level=it1->key().level();
            // std::cout << "Checking symmetry for block: " << key << std::endl;
            // std::cout << "Coordinate: " << coord << std::endl;
            // std::cout<< "Level: " << level << std::endl;
            // std::array<int, 3> shift = {0, std::pow2(level), 0};
            auto opposite_coord = coord;
            // opposite_coord[0] = coord[0];
            // opposite_coord[1] = std::pow2(level)-coord[1];
            // 128 = 1792/14 =extent on baselevel
            opposite_coord[1] = 128*(1<<ref_level)- (coord[1]+1);

            //   std::cout << "Checking symmetry for block: " << key << std::endl;
            //     std::cout << "Coordinate: " << coord << std::endl;
            //     std::cout<< "Level: " << level << std::endl;
            //     std::cout << "Shifted coordinate: " << opposite_coord << std::endl;
            auto it2=ref_tree->find_octant(key_t(opposite_coord, level));
            if(!it2)
            {
                if(toPrint)
                {
                    std::cout << "No opposite block found for: " << it1->key() << std::endl;
                    std::cout<<"shifted coord: " << opposite_coord << std::endl;
                }
                // std::cout << "No opposite block found for: " << it1->key() << std::endl;
                // std::cout<<"shifted coord: " << opposite_coord << std::endl;
                octs.emplace_back(it1->key().id());
                level_change.emplace_back(-1);
                // std::cout << "Found opposite block: " << it2->key() << std::endl;
                continue;
            }
            if(!it2->is_leaf()||(ref_level==0 && it2->is_correction()))
            {
             
                // std::cout << "Opposite block is not a leaf: " << it2->key() << std::endl;
                if(toPrint)
                {
                    std::cout << "No opposite leaf block found for: " << it1->key() << std::endl;
                    std::cout<<"shifted coord: " << opposite_coord << std::endl;
                    std::cout<<"key:"<<it2->key()<<std::endl;
                }
                octs.emplace_back(it1->key().id());
                level_change.emplace_back(-1);
                continue;
            }

        }


    }
};

}// namespace iblgf
#endif // IBLGF_MERGE_TREES_HPP