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

#include <string>     // std::string, std::to_string

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/filesystem.hpp>

#include "postProcessIntrp.hpp"
#include <iblgf/dictionary/dictionary.hpp>

using namespace iblgf;

void write_post_info(std::string dir, int i)
{
	boost::mpi::communicator world;
    if (world.rank()==0)
    {
        std::ofstream ofs(
                "./"+dir+"/postProc_info",
                std::ofstream::out);
        if (!ofs.is_open())
        {
            throw std::runtime_error("Could not open file for info write ");
        }

        ofs<<"postProcLast = " << i << ";" << std::endl;
        ofs.close();
    }
}


int main(int argc, char *argv[])
{

	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	std::string input="./";
    input += std::string("configFile");

    if (argc>1 && argv[1][0] != '-')
    {
        input = argv[1];
    }

    // Read in dictionary
    Dictionary dictionary(input, argc, argv);

    // Find output directory
    auto dict_out=dictionary
            .get_dictionary("simulation_parameters")->get_dictionary("output");
    std::string dir = dict_out->template get<std::string>("directory");

    int nLevels=dictionary
            .get_dictionary("simulation_parameters")->template get<int>("nLevels");

    int baseSteps=dictionary
            .get_dictionary("simulation_parameters")->template get<int>("nBaseLevelTimeSteps");

    int tot_steps=baseSteps*pow(2,nLevels);

    // find the when it finished last time from reading the postProc_info
    std::string info_dir="./"+dir+"/postProc_info";
    std::ifstream f(info_dir);
    int postStartIdx = 0;
    if (f.good())
    {
        Dictionary info_d(info_dir);
        postStartIdx = info_d.template get_or<int>("postProcLast",0);
    }
    else
        postStartIdx = 0;


    for (int i=postStartIdx; i<=tot_steps; ++i)
    {
        std::string flow_file = "./"+dir+"/flow_"+std::to_string(i)+".hdf5";
        std::string tree_file = "./"+dir+"/tree_info_"+std::to_string(i)+".bin";

        if ( boost::filesystem::exists(flow_file) && boost::filesystem::exists(tree_file))
        {
            if (world.rank()==0)
            {
                std::cout << tree_file << std::endl;
                std::cout << flow_file << std::endl;
            }

            //Instantiate setup
            PostProcessIntrp setup(&dictionary,
                    tree_file,
                    flow_file);

            // run setup
            setup.run("postProc_"+std::to_string(i));
            write_post_info(dir, i);
        }
    }

    return 0;
}
