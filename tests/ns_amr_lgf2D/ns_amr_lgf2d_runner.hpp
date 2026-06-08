#ifndef IBLGF_INCLUDED_NS_AMR_LGF2D_RUNNER_HPP
#define IBLGF_INCLUDED_NS_AMR_LGF2D_RUNNER_HPP

#include <string>
#include <vector>

namespace iblgf::ns_amr_lgf2d
{

struct RunResult
{
    double      measured_linf_error = 0.0;
    double      expected_linf_error = 0.0;
    double      difference = 0.0;
    double      overall_u1_linf_error = 0.0;
    double      fine_u1_linf_error = 0.0;
    double      fine_u2_linf_error = 0.0;
    std::string config_path;
    std::string working_directory;
};

RunResult run_from_config(const std::string&              config_path,
    const std::vector<std::string>& cli_overrides = {});

} // namespace iblgf::ns_amr_lgf2d

#endif
