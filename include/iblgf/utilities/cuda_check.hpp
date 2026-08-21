#ifndef INCLUDED_CUDA_CHECK_IBLGF_HPP
#define INCLUDED_CUDA_CHECK_IBLGF_HPP

// This header is only ever included from GPU-only translation units
// (convolution_GPU.hpp/.cu) or from inside an IBLGF_COMPILE_CUDA-guarded
// block (lgf.hpp), so it does not gate itself on that flag -- unlike the
// *test* entry points, convolution_GPU.cu never defines IBLGF_COMPILE_CUDA.
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>

namespace iblgf
{
inline void cuda_check(cudaError_t err, const char* what, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("[CUDA ERROR] ") + what + " failed at " + file + ":" +
            std::to_string(line) + " -- " + cudaGetErrorString(err));
    }
}

inline void cufft_check(cufftResult res, const char* what, const char* file, int line)
{
    if (res != CUFFT_SUCCESS)
    {
        throw std::runtime_error(std::string("[CUFFT ERROR] ") + what + " failed at " + file + ":" +
            std::to_string(line) + " -- code " + std::to_string(static_cast<int>(res)));
    }
}
} // namespace iblgf

// Wrap an allocation/setup call: throws immediately with file:line + the CUDA
// error string if it fails, instead of letting a null/garbage pointer get used
// later and silently corrupt the CUDA context (which then makes every
// unrelated subsequent CUDA call fail with "illegal memory access").
#define IBLGF_CUDA_CHECK(call) ::iblgf::cuda_check((call), #call, __FILE__, __LINE__)
#define IBLGF_CUFFT_CHECK(call) ::iblgf::cufft_check((call), #call, __FILE__, __LINE__)

// Call right after a kernel launch to catch launch-configuration errors
// (invalid grid/block dims, too much shared memory, etc.) immediately.
// Note: kernel launches are async, so this does NOT catch faults that occur
// during kernel execution (e.g. an out-of-bounds access) -- those still need
// a synchronize (or compute-sanitizer) to localize precisely.
#define IBLGF_CUDA_CHECK_LAST_ERROR() ::iblgf::cuda_check(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)

#endif // INCLUDED_CUDA_CHECK_IBLGF_HPP
