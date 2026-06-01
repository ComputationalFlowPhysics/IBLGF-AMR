#include "poisson_runner.hpp"

#include "vortexrings.hpp"

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

namespace iblgf::poisson
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
        std::vector<std::string> init_args = {"iblgf_poisson"};
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
    argv_strings.push_back("iblgf_poisson");
    argv_strings.insert(
        argv_strings.end(), cli_overrides.begin(), cli_overrides.end());

    auto argv = build_argv(argv_strings);
    int  argc = static_cast<int>(argv.size());

    dictionary::Dictionary dictionary(config.string(), argc, argv.data());

    VortexRingTest setup(&dictionary);
    const double   measured = setup.run();
    const double   expected =
        dictionary.get_dictionary("simulation_parameters")
            ->template get_or<double>("EXP_LInf", 0.0);

    return RunResult{
        measured,
        expected,
        measured - expected,
        config.string(),
        config.parent_path().string(),
    };
}

} // namespace iblgf::poisson

namespace iblgf
{

double vortex_run(std::string input, int argc, char** argv)
{
    std::vector<std::string> cli_overrides;
    for (int i = 1; i < argc; ++i)
    {
        if (argv[i] == nullptr) continue;
        if (!(argv[i][0] == '-' && argv[i][1] == '-')) continue;
        cli_overrides.emplace_back(argv[i]);
    }

    return poisson::run_from_config(input, cli_overrides).measured_linf_error;
}

} // namespace iblgf
