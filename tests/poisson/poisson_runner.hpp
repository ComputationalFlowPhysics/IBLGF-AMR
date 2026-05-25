#ifndef IBLGF_INCLUDED_POISSON_RUNNER_HPP
#define IBLGF_INCLUDED_POISSON_RUNNER_HPP

#include <string>
#include <vector>

namespace iblgf::poisson
{

struct RunResult
{
    double      measured_linf_error = 0.0;
    double      expected_linf_error = 0.0;
    double      difference = 0.0;
    std::string config_path;
    std::string working_directory;
};

RunResult run_from_config(const std::string&              config_path,
    const std::vector<std::string>& cli_overrides = {});

} // namespace iblgf::poisson

#endif
