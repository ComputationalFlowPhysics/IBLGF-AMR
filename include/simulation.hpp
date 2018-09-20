#ifndef IBLGF_INCLUDED_SIMULATION_HPP
#define IBLGF_INCLUDED_SIMULATION_HPP

#include <dictionary/dictionary.hpp>
#include <IO/vtk_io.hpp>

using namespace dictionary;

template<class Domain>
class Simulation
{
    
public: 
    using domain_type = Domain;


public: 


    Simulation(const Simulation& other) = delete;
    Simulation(Simulation&& other) = default;
    Simulation& operator=(const Simulation& other) & = delete;
    Simulation& operator=(Simulation&& other) & = default;
    ~Simulation() = default;

    Simulation(std::shared_ptr<Dictionary> _dictionary)
    :dictionary_(_dictionary),domain_(_dictionary->get_dictionary("domain"))
    {
    }

    friend std::ostream& operator<<(std::ostream& os, Simulation& s)
    {
        os
        <<"Domain: \n"<<s.domain_<<" "
        <<std::endl;

        return os;

    }

    void write(std::string _filename)
    {
        writer.write_vtk(_filename, domain_);
    }


public:
  std::shared_ptr<Dictionary> dictionary_;
  Domain domain_;
  boost::mpi::communicator world_;
  io::Vtk_io<Domain> writer;

};





#endif
