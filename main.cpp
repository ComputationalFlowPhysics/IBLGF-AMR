#include <iostream>
#include <fstream>
#include <vector>
#include <climits>
#include <iomanip>
#include <set>
#include "SpatialDomain/key.h"
#include "SpatialDomain/linearTree.cpp"

#include "SpatialDomain/vector.h"
#include <math.h>


const int Dim = 3;
using namespace octree;

int main(int argc, char *argv[])
{

    LinearTree<Dim, double>::coordinate_t c((1));
    Key<Dim> key(c,3);
    key.print_binary();

    LinearTree<Dim, double>::coordinate_t point((1));
    point[0] = 3;
    point[1] = 4;
    point[2] = 5;

    auto extent = point;
    LinearTree<Dim,double> lt(extent);


    //Initialze:
    for(auto it = lt.begin_nodes(); it != lt.end_nodes(); ++it)
        it->data(0.0);
    
    std::ofstream ofs0("points.txt");
    
    for(auto it = lt.begin_nodes(); it != lt.end_nodes(); ++it)
        ofs0 << it->real_coordinate() << std::endl;


    //Refine 3 times:
    for (int i = 0; i < 4; ++i)
    {
        int count = 0;
        for (auto it = lt.begin_nodes(); it != lt.end_nodes(); ++it)
        {
            if (count == 1)
            {
                lt.refine(it, [&i](const LinearTree<Dim,double>::leaf_t& _t){
                        _t.data(static_cast<double>(i));});
            }
            ++count;
        }
    }

    //Coarsend the mesh again
    std::ofstream ofs1("points_refined.txt");
    for (auto it = lt.begin_nodes(); it != lt.end_nodes(); ++it)
        ofs1<<it->real_coordinate()<<std::endl;

    for (int i = 0; i < 4; ++i)
    {
        int count = 0;
        for (auto it = lt.begin_nodes(); it != lt.end_nodes(); ++it)
        {
            if (count == 1)
            {
                lt.coarsen(it);
            }
            ++count;
        }
    }

    std::ofstream ofs2("points_coarsened.txt");
    for (auto it = lt.begin_nodes(); it != lt.end_nodes(); ++it)
        ofs2 << it->real_coordinate() << std::endl;

    return 0;
}
