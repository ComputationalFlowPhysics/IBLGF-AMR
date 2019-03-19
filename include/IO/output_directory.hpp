#ifndef IBLGF_INCLUDED_OUTPUTDIRECTORY_HPP
#define IBLGF_INCLUDED_OUTPUTDIRECTORY_HPP

#include <iostream>
#include <sstream>
#include <map>

#include <global.hpp>
#include <dictionary/dictionary.hpp>
#include <boost/filesystem.hpp>

namespace io
{

struct outputParams_singleton;            
outputParams_singleton& output();
	
struct outputParams_singleton // singleton
{
private:
	
	/** @brief Default constructor */
	outputParams_singleton(){};
	/** @brief Function for instantiating the singleton is a friend */
	friend outputParams_singleton& output(); 

    boost::filesystem::path directory_output = boost::filesystem::current_path();
    std::string rel_directory_str = "./";

    boost::filesystem::path directory_restartLoad = boost::filesystem::current_path();
    std::string rel_directory_restartLoad_str = "./";

    boost::filesystem::path directory_restartSave = boost::filesystem::current_path();
    std::string rel_directory_restartSave_str = "./";

    std::string sim_name_="outFile";

public:
	
    using dictionary_t= dictionary::Dictionary;

    //No copy constructor or assign-operator
	outputParams_singleton(const outputParams_singleton&) = delete;
	outputParams_singleton& operator=(const outputParams_singleton&) = delete;

    //void set_outputParams(std::string _outputDir ){ outputParams_=_outputDir; }
    void set_directory(std::shared_ptr<dictionary_t>& _dict_output )
    { 

        sim_name_= _dict_output->
            template get_or<std::string>("name","my_simulation");

        std::string dir="./";
        boost::mpi::communicator world;
		if(_dict_output-> has_key("directory")){ 

            dir= _dict_output-> template get<std::string>("name");
            boost::filesystem::path outdir(dir);
            directory_output=outdir;
            rel_directory_str=dir;
            if(world.rank()==0){
                boost::filesystem::create_directories(directory_output);
            }
        }
    }

    void set_restart_directory(std::shared_ptr<dictionary_t>& _dict_output )
    { 
        std::string dir="./";
        boost::mpi::communicator world;
		if(_dict_output-> has_key("load_directory"))
        { 

            dir= _dict_output-> template get<std::string>("load_directory");
            boost::filesystem::path outdir(dir);
            directory_restartLoad=outdir;
            rel_directory_restartLoad_str=dir;
        }else
        {
            directory_restartLoad=directory_output;
            rel_directory_restartLoad_str=rel_directory_str;
        }

        dir="./";
		if(_dict_output-> has_key("save_directory"))
        { 

            dir= _dict_output-> template get<std::string>("save_directory");
            boost::filesystem::path outdir(dir);
            directory_restartSave=outdir;
            rel_directory_restartSave_str=dir;

            if(world.rank()==0){
                boost::filesystem::create_directories(rel_directory_restartSave_str);
            }
        }
        else
        {
            directory_restartSave=directory_output;
            rel_directory_restartSave_str=rel_directory_str;
        }



    }

    boost::filesystem::path directory(){return directory_output; }
    std::string directory_str(){return rel_directory_str; }
    std::string name(){return sim_name_; }

    //Restart directories:
    boost::filesystem::path restart_load_directory(){return directory_restartLoad; }
    std::string restart_load_directory_str(){return rel_directory_restartLoad_str; }
    boost::filesystem::path restart_save_directory(){return directory_restartSave; }
    std::string restart_save_directory_str(){return rel_directory_restartSave_str; }
	
    
};


   

/** @brief Get a reference single instance  */
inline outputParams_singleton& output()
{
	static outputParams_singleton params;
	return params;
}



/** @brief Helper to set output directory in the simulation */
struct setOutput
{

    using dictionary_t= dictionary::Dictionary;
    setOutput( dictionary_t* dict_)
    {
        this->set(dict_); 
    }


private:

    void set( dictionary_t* dict_)
    {
        std::shared_ptr<dictionary_t> subdict_ptr;
        if( dict_->get_dictionary("output",subdict_ptr) ){  
            output().set_directory(subdict_ptr );
        }
        if( dict_->get_dictionary("restart",subdict_ptr) ){  
            output().set_restart_directory(subdict_ptr );
        }

    }

};


}
#endif
