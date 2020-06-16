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

#ifndef INCLUDED_DICTIONARY_HPP
#define INCLUDED_DICTIONARY_HPP

#include <stdio.h>
#include <string>
#include <map>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ostream>
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <memory>

#include <dictionary/string_utils.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

namespace dictionary
{
class Dictionary
{
  public: // Types:
    using vector_map_type = std::map<std::string, std::vector<std::string>>;
    using scalar_map_type = std::map<std::string, std::string>;
    using sd_vector_type = std::vector<std::shared_ptr<Dictionary>>;

    template<typename T, std::size_t N>
    using vector_type = std::array<T, N>;

  public: //Ctors:
    Dictionary() = default;
    ~Dictionary() = default;
    Dictionary(const Dictionary& rhs) = default;
    Dictionary& operator=(const Dictionary&) & = default;

    Dictionary(Dictionary&& rhs) = default;
    Dictionary& operator=(Dictionary&&) & = default;

    Dictionary(const std::string _file, int argc = 0, char** argv = nullptr)
    : root(true)
    {
        content_ = read_file(_file);

        //remove comments
        string_utilities::remove_single_line_comments(content_, "//");
        string_utilities::erase_all_contents(content_, "/*", "*/");

        init();

        //By default cmd arguments belong to first declared dictionary
        this->parse_cmd_args(argc, argv);
    }

    Dictionary(std::string _name, std::string _content)
    : name_(_name)
    , content_(_content)
    {
        init();
    }

  private: //Exceptions
    class DictionaryExcpetion : public std::runtime_error
    {
      public:
        DictionaryExcpetion(std::string _name, std::string _key)
        : std::runtime_error(
              "dictionary " + _name + ": key " + _key + " not found!")
        {
        }
        DictionaryExcpetion(std::string _name)
        : std::runtime_error("dictionary " + _name + " does not exist")
        {
        }
    };

  private: //Helper functions
    void init()
    {
        auto dicts = string_utilities::get_dictionaries(
            content_, opening_delimiter_, closing_delimiter_);
        string_utilities::erase_dictionaries(
            content_, opening_delimiter_, closing_delimiter_);
        for (auto& d : dicts)
        {
            sub_dictionaries_.emplace_back(
                std::make_shared<Dictionary>(d.first, d.second));
            sub_dictionaries_.back()->parent_ = this;
        }
        parse_variables();
    }

    std::string read_file(std::string _filename) const
    {
        std::ifstream ifs(_filename);
        if (!ifs.good() || !ifs.is_open())
        { throw std::runtime_error("Could not open file: " + _filename); }
        return std::string((std::istreambuf_iterator<char>(ifs)),
            std::istreambuf_iterator<char>());
    }

    void parse_variables() { return parse_variables(this->content_); }
    void parse_variables(const std::string& _content)
    {
        auto variables = string_utilities::get_dictionaries(_content, "=", ";");
        for (auto& var : variables)
        {
            auto elements_str =
                string_utilities::get_dictionaries(var.second, "(", ")");
            std::vector<std::string> elements;
            if (elements_str.size() > 0)
            {
                for (auto& var : elements_str)
                { elements = string_utilities::split_string(var.second); }
                auto itp = vector_variables_.insert(
                    std::make_pair(var.first, elements));
                if (!itp.second) itp.first->second = elements;
            }
            else
            {
                auto itp = scalar_variables_.insert(
                    std::make_pair(var.first, var.second));
                if (!itp.second) itp.first->second = var.second;
            }
        }
    }

  public: //Some access functions
    void parse_cmd_args(int argc, char** argv)
    {
        for (int i = 1; i < argc; ++i)
        {
            if (!(argv[i][0] == '-' && argv[i][1] == '-')) continue;

            std::string arg(argv[i]);
            arg += ";";
            arg = arg.substr(2, arg.size() - 2);
            const auto idx = arg.find_last_of("/");
            const bool subdict = idx != arg.npos;

            if (subdict)
            {
                std::string path = arg.substr(0, idx);
                arg = arg.substr(idx + 1);

                try
                {
                    auto sd = this->get_dictionary(path);
                    sd->parse_variables(arg);
                }
                catch (...)
                {
                    throw std::runtime_error(
                        "Cmd-Args: Dictionary path \"" + path +
                        "\" is not valid for dictionary: " + this->name_);
                }
            }
            else
            {
                parse_variables(arg);
            }
        }
    }

    auto get_subdictionary(const std::string& _name)
    {
        auto it =
            std::find_if(sub_dictionaries_.cbegin(), sub_dictionaries_.cend(),
                [&_name](const auto& d) { return d->name_ == _name; });
        if (it == sub_dictionaries_.cend())
        {
            //for(auto & sd: sub_dictionaries_) std::cout<<"name :"<<sd->name_<<std::endl;
            throw DictionaryExcpetion(name_);
        }
        return *it;
    }

    sd_vector_type get_all_dictionaries(std::string _name)
    {
        sd_vector_type res;
        for (auto& sd : sub_dictionaries_)
        {
            if (sd->name() == _name) res.push_back(sd);
        }
        return res;
    }

    std::shared_ptr<Dictionary> get_dictionary(
        const std::string& _dictionary_path)
    {
        auto split = string_utilities::split_string(_dictionary_path, '/');
        auto sub_dict = get_subdictionary(split[0]);
        for (std::size_t i = 1; i < split.size(); ++i)
        { sub_dict = sub_dict->get_subdictionary(split[i]); }
        return sub_dict;
    }

    bool get_dictionary(
        const std::string& _dictionary_path, std::shared_ptr<Dictionary>& _dict)
    {
        auto split = string_utilities::split_string(_dictionary_path, '/');
        try
        {
            auto sub_dict = get_subdictionary(split[0]);
            for (std::size_t i = 1; i < split.size(); ++i)
            { sub_dict = sub_dict->get_subdictionary(split[i]); }
            _dict = sub_dict;
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

  public: //Access functions
    bool has_key(const std::string& key) const
    {
        auto iter = scalar_variables_.find(key);
        if (iter != scalar_variables_.cend()) return true;
        else
            return (vector_variables_.find(key) != vector_variables_.cend());
    }

    template<typename T>
    T get(const std::string& _key) const
    {
        auto it = scalar_variables_.find(_key);
        if (it == scalar_variables_.cend())
        { throw DictionaryExcpetion(name_, _key); }
        return string_utilities::lexical_cast<T>(it->second);
    }

    template<typename T, std::size_t N>
    auto get(const std::string& _key) const
    {
        auto it = vector_variables_.find(_key);
        if (it == vector_variables_.cend())
        { throw DictionaryExcpetion(name_, _key); }
        vector_type<T, N> arr;
        for (std::size_t i = 0; i < it->second.size(); ++i)
        { arr[i] = string_utilities::lexical_cast<T>(it->second[i]); }
        return arr;
    }

    template<typename T>
    T get_or(const std::string& _key, const T& _default) const
    {
        try
        {
            return get<T>(_key);
        }
        catch (...)
        {
            return _default;
        }
    }
    template<typename T, std::size_t N>
    auto get_or(
        const std::string& _key, const vector_type<T, N>& _default) const
    {
        try
        {
            return get<T, N>(_key);
        }
        catch (...)
        {
            return _default;
        }
    }

    template<typename T>
    void set(const std::string& _key, T _value)
    {
        auto it = scalar_variables_.find(_key);
        if (it == scalar_variables_.cend())
        { throw DictionaryExcpetion(name_, _key); }
        it->second = std::to_string(_value);
    }

    auto parent_dictionary() const noexcept { return parent_; }

    std::ostream& print(std::ostream& os, std::string _tab = "") const
    {
        if (!root)
        {
            os << _tab << name_ << std::endl;
            os << _tab << "{" << std::endl;
            for (auto& e : scalar_variables_)
            {
                os << _tab << '\t' << e.first << "=" << e.second << ";"
                   << std::endl;
            }
            for (auto& e : vector_variables_)
            {
                os << _tab << '\t' << e.first << "=(";
                for (auto& ee : e.second) os << ee << ",";
                os << ");" << std::endl;
            }
        }
        std::string ntab = _tab + "    ";
        for (auto& d : sub_dictionaries_) d->print(os, root ? _tab : ntab);
        if (!root) os << _tab << "}\n" << std::endl;
        return os;
    }

    friend std::ostream& operator<<(std::ostream& os, const Dictionary& d)
    {
        d.print(os);
        return os;
    }

    std::string name() const noexcept { return name_; }

  private:
    bool              root = false;
    std::string       name_ = "default";
    const std::string opening_delimiter_ = "{";
    const std::string closing_delimiter_ = "}";

    std::string     content_;
    sd_vector_type  sub_dictionaries_;
    vector_map_type vector_variables_;
    scalar_map_type scalar_variables_;
    Dictionary*     parent_ = nullptr;
};

} // namespace dictionary

#endif //Included dictionary
