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

#ifndef INCLUDED_STRING_UTILS_HPP
#define INCLUDED_STRING_UTILS_HPP

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>

namespace string_utilities
{
namespace detail
{
bool
content_between_impl(const std::string& _str, size_t& _opening_delimiter_pos,
    size_t& _closing_delimiter_pos, std::string _opening_delimiter,
    std::string _closing_delimiter)
{
    bool found_oBracket, found_cBracket;
    int  delimeter_balance = 0;
    _closing_delimiter_pos = 0;
    _opening_delimiter_pos = 0;

    _opening_delimiter_pos = _str.find(_opening_delimiter);
    if (_opening_delimiter_pos == std::string::npos)
    {
        _opening_delimiter_pos = 0;
        return false;
    }

    bool found = false;
    for (size_t i = _opening_delimiter_pos;
         i < _str.length() - _closing_delimiter.length() + 1; i++)
    {
        found_oBracket =
            (_str.substr(i, _opening_delimiter.length()) == _opening_delimiter);
        found_cBracket =
            (_str.substr(i, _closing_delimiter.length()) == _closing_delimiter);

        if (found_oBracket || found_cBracket)
        {
            if (found_oBracket) ++delimeter_balance;
            else
                --delimeter_balance;

            if (delimeter_balance == 0)
            {
                _closing_delimiter_pos = i;
                found = true;
                break;
            }
        }
    }
    return found;
}

std::vector<std::string>
split(std::string _str, std::string _delimiters)
{
    std::vector<std::string> split;
    std::stringstream        stringStream(_str);
    std::string              line;
    while (std::getline(stringStream, line))
    {
        std::size_t prev = 0, pos = 0;
        while (
            (pos = line.find_first_of(_delimiters, prev)) != std::string::npos)
        {
            if (pos > prev) { split.push_back(line.substr(prev, pos - prev)); }
            prev = pos + 1;
        }
        if (prev < line.length())
        { split.push_back(line.substr(prev, std::string::npos)); }
    }
    return split;
}

} //namespace detail

template<typename T>
T
lexical_cast(const std::string& _str)
{
    //return boost::lexical_cast<T>(_str);
    T                 res;
    std::stringstream ss(_str);
    ss >> std::boolalpha >> res;
    return res;
}

bool
is_numeric(std::string _str)
{
    //maybe count the number of apperaces for +-.
    int  dot_count = 0;
    int  sign_count = 0;
    bool found_digit = false;
    for (auto& c : _str)
    {
        if (c == '-' || c == '+')
        {
            if (sign_count > 1 || found_digit) return false;
            sign_count++;
        }
        else if (c == '.')
        {
            if (dot_count > 1) return false;
            dot_count++;
        }
        else
        {
            if (isdigit(c)) found_digit = true;
            else
                return false;
        }
    }
    return true;
}

std::vector<std::string>
split_string(std::string _str, char _delimiter = ',')
{
    std::vector<std::string> res;
    std::stringstream        ss(_str);
    std::string              token;
    while (std::getline(ss, token, _delimiter)) { res.push_back(token); }
    return res;
}

bool
find_dictionary_positions(const std::string& _str, size_t& _identifier_pos,
    size_t& _opening_delimiter_pos, size_t& _closing_delimiter_pos,
    std::string _opening_delimiter, std::string _closing_delimiter)
{
    bool res = detail::content_between_impl(_str, _opening_delimiter_pos,
        _closing_delimiter_pos, _opening_delimiter, _closing_delimiter);

    std::string str = _str.substr(0, _opening_delimiter_pos);
    std::reverse(str.begin(), str.end());
    std::stringstream iss(str);
    std::string       last_token;
    iss >> last_token;
    std::reverse(last_token.begin(), last_token.end());
    _identifier_pos = _str.rfind(last_token, _opening_delimiter_pos);

    return res;
}

auto
get_dictionaries(const std::string& _str, std::string _opening_delimiter,
    std::string _closing_delimiter)
{
    std::vector<std::pair<std::string, std::string>> results;
    auto                                             input = _str;

    size_t opening_delimiter_pos, closing_delimiter_pos, identifier_pos;
    while (
        find_dictionary_positions(input, identifier_pos, opening_delimiter_pos,
            closing_delimiter_pos, _opening_delimiter, _closing_delimiter))
    {
        std::string dictionary_content =
            input.substr(opening_delimiter_pos + _opening_delimiter.length(),
                closing_delimiter_pos - _closing_delimiter.length() -
                    opening_delimiter_pos);

        if (identifier_pos == std::string::npos)
            throw std::runtime_error("no identifier found");

        std::stringstream iss(input.substr(
            identifier_pos, opening_delimiter_pos - identifier_pos));
        std::string       dictionary_identifier;
        iss >> dictionary_identifier;
        std::string remaining =
            input.substr(closing_delimiter_pos + _closing_delimiter.length());
        results.push_back(
            std::make_pair(dictionary_identifier, dictionary_content));
        input = remaining;
        if (remaining == "") break;
    }
    return results;
}

auto
erase_dictionaries(std::string& _str, std::string _opening_delimiter,
    std::string _closing_delimiter)
{
    size_t opening_delimiter_pos, closing_delimiter_pos, identifier_pos;
    while (
        find_dictionary_positions(_str, identifier_pos, opening_delimiter_pos,
            closing_delimiter_pos, _opening_delimiter, _closing_delimiter))
    {
        if (identifier_pos != std::string::npos &&
            closing_delimiter_pos != std::string::npos &&
            closing_delimiter_pos > identifier_pos)
        {
            _str.erase(identifier_pos, closing_delimiter_pos - identifier_pos +
                                           _closing_delimiter.length());
        }
    }
}

void
remove_single_line_comments(std::string& _str, std::string _comment_identifier)
{
    std::string       res;
    std::stringstream stringStream(_str);
    std::string       line;
    while (std::getline(stringStream, line))
    {
        auto pos = line.find(_comment_identifier);
        if (pos != std::string::npos) line.erase(pos);
        res += line + '\n';
    }
    _str = res;
}

void
erase_all_contents(std::string& _str, std::string _opening_delimiter,
    std::string _closing_delimiter)
{
    std::string result, remaining;
    size_t      opening_delimiter_pos, closing_delimiter_pos;
    while (detail::content_between_impl(_str, opening_delimiter_pos,
        closing_delimiter_pos, _opening_delimiter, _closing_delimiter))
    {
        _str.erase(opening_delimiter_pos, closing_delimiter_pos -
                                              opening_delimiter_pos +
                                              _closing_delimiter.length());
        if (_str == "") break;
    }
}

} //namespace string_utilities

#endif //string utils

