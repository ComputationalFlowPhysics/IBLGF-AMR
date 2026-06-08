#include "ns_amr_lgf_runner.hpp"

#include "ns_amr_lgf.hpp"

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <iblgf/dictionary/dictionary.hpp>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

namespace iblgf::ns_amr_lgf
{
namespace
{

class WorkingDirectoryGuard
{
  public:
    explicit WorkingDirectoryGuard(const std::filesystem::path& new_path)
    : original_(std::filesystem::current_path())
    {
        std::filesystem::current_path(new_path);
    }

    ~WorkingDirectoryGuard() noexcept
    {
        try
        {
            std::filesystem::current_path(original_);
        }
        catch (...)
        {
        }
    }

  private:
    std::filesystem::path original_;
};

void ensure_mpi_initialized()
{
    static std::unique_ptr<boost::mpi::environment> env;

    if (boost::mpi::environment::finalized())
    {
        throw std::runtime_error(
            "MPI was already finalized in this Python process.");
    }

    if (!boost::mpi::environment::initialized())
    {
        int argc = 1;
        std::vector<std::string> init_args = {"iblgf_ns_amr_lgf"};
        std::vector<char*>       argv_storage = {init_args[0].data()};
        char**                   argv = argv_storage.data();

        env = std::make_unique<boost::mpi::environment>(argc, argv, false);
    }
}

std::vector<char*> build_argv(std::vector<std::string>& args)
{
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (auto& arg : args) argv.push_back(arg.data());
    return argv;
}

} // namespace

RunResult run_from_config(const std::string&              config_path,
    const std::vector<std::string>& cli_overrides)
{
    ensure_mpi_initialized();

    const std::filesystem::path config = std::filesystem::absolute(config_path);
    if (!std::filesystem::exists(config))
    {
        throw std::runtime_error("Config file does not exist: " + config.string());
    }

    WorkingDirectoryGuard cwd_guard(config.parent_path());

    std::vector<std::string> argv_strings;
    argv_strings.reserve(cli_overrides.size() + 1);
    argv_strings.push_back("iblgf_ns_amr_lgf");
    argv_strings.insert(
        argv_strings.end(), cli_overrides.begin(), cli_overrides.end());

    auto argv = build_argv(argv_strings);
    int  argc = static_cast<int>(argv.size());

    dictionary::Dictionary dictionary(config.string(), argc, argv.data());

    NS_AMR_LGF setup(&dictionary);
    const double overall_u1 = setup.run();
    const double fine_u1 = setup.u1_Linf_fine();
    const double fine_u2 = setup.u2_Linf_fine();
    const double fine_u3 = setup.u3_Linf_fine();
    const double expected =
        dictionary.get_dictionary("simulation_parameters")
            ->template get_or<double>("EXP_LInf", 0.0);

    return RunResult{
        fine_u1,
        expected,
        fine_u1 - expected,
        overall_u1,
        fine_u1,
        fine_u2,
        fine_u3,
        config.string(),
        config.parent_path().string(),
    };
}

} // namespace iblgf::ns_amr_lgf
