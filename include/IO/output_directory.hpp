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

struct IO_parameters;
IO_parameters& output();

struct IO_parameters // singleton
{
private:

	/** @brief Default constructor */
	IO_parameters(){};
	/** @brief Function for instantiating the singleton is a friend */
	friend IO_parameters& output();


public:

    using dictionary_t= dictionary::Dictionary;

    //No copy constructor or assign-operator
	IO_parameters(const IO_parameters&) = delete;
	IO_parameters& operator=(const IO_parameters&) = delete;

    void set_directory(std::shared_ptr<dictionary_t>& _dict_output ) noexcept
    {

        std::string dir="./";
        boost::mpi::communicator world;
		if(_dict_output-> has_key("directory")){

            dir= _dict_output-> template get<std::string>("directory");
            boost::filesystem::path outdir(dir);

            directory_output=outdir;
            rel_directory_str=dir;
            if(world.rank()==0){
                boost::filesystem::create_directories(directory_output);
            }
        }
    }

    void set_restart_directory(std::shared_ptr<dictionary_t>& _dict_output ) noexcept
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

    std::string dir()const noexcept{return rel_directory_str; }
    std::string restart_load_dir()const noexcept{return rel_directory_restartLoad_str; }
    std::string restart_save_dir()const noexcept{return rel_directory_restartSave_str; }


private:

    boost::filesystem::path directory_output = boost::filesystem::current_path();
    std::string rel_directory_str = "./";

    boost::filesystem::path directory_restartLoad = boost::filesystem::current_path();
    std::string rel_directory_restartLoad_str = "./";

    boost::filesystem::path directory_restartSave = boost::filesystem::current_path();
    std::string rel_directory_restartSave_str = "./";

};




/** @brief Get a reference single instance  */
inline IO_parameters& output()
{
	static IO_parameters params;
	return params;
}



/** @brief Helper to set output directory in the simulation */
struct IO_init
{

    using dictionary_t= dictionary::Dictionary;
    IO_init( dictionary_t* dict_)
    {
        this->set(dict_);
    }

private:

    void set( dictionary_t* dict_) const noexcept
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
