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

#ifndef __FFTWWrapper_h__
#define __FFTWWrapper_h__

// Typical use of FFTW entails the following steps:
// 1. Allocate input and output buffers.
// 2. Compute a "plan" struct that. This tells FFTW what
//    algorithms it should use when actually computing the FFT.
// 3. Execute the FFT/IFFT operation on the buffers from step 1.
//
// This file contains two classes that wrap this process nicely.
// Currently only one-dimensional transforms of real data to
// complex and back are supported.
//
//

#include <cassert>
#include <cstring>
#include <vector>
#include <complex.h>
#include <fftw3.h>

namespace fft
{
// Usage: (after initializing the class)
// 1. Fill input_buffer with input containing n_real_samples double
//    numbers (note, set_input_zeropadded will copy your buffer with
//    optional zero padding)
// 2. Run execute().
// 3. Extract output by calling get_output() or directly access
//    output_buffer[0], ..., output_buffer[output_size-1].
//    Note that the output is composed of n_real_samples/2 + 1
//    complex numbers.
//
// These 3 steps can be repeated many times.
template<typename T1, typename T2>
class fftw_wrapper
{
  public:
    const int input_size;
    T1* const input_buffer;

    const int output_size;
    T2* const output_buffer;

  public:
    fftw_wrapper(const fftw_wrapper& other) = delete;
    fftw_wrapper(fftw_wrapper&& other) = default;
    fftw_wrapper& operator=(const fftw_wrapper& other) & = delete;
    fftw_wrapper& operator=(fftw_wrapper&& other) & = default;
    ~fftw_wrapper() = default;

    // Constructors
    fftw_wrapper(int n_real_samples)
    : input_size(n_real_samples)
    , input_buffer(fftw_alloc_real(n_real_samples))
    , output_size(n_real_samples / 2 + 1)
    , output_buffer(fftw_alloc_complex(n_real_samples / 2 + 1))
    {
        plan = fftw_plan_dft_1d(
            n_real_samples, input_buffer, output_buffer, FFTW_ESTIMATE);
    }

  public:
    void set_input_zeropadded(const T1* buffer, int size)
    {
        assert(size <= input_size);
        memcpy(input_buffer, buffer, sizeof(T1) * size);
        memset(&input_buffer[size], 0, sizeof(T1) * (input_size - size));
    }

    void set_input_zeropadded(const std::vector<T1>& vec)
    {
        set_input_zeropadded(&vec[0], vec.size());
    }

    void execute() { fftw_execute(plan); }

    std::vector<T2> get_output()
    {
        return vector<T2>(output_buffer, output_buffer + output_size);
    }

  private:
    fftw_plan plan;
};

} // namespace fft
#endif

//std::vector<std::complex<float>, boost::alignment::aligned_allocator_adaptor<std::allocator<std::complex<float>>,32>>
//dft(float* data, unsigned int nx, unsigned int ny, unsigned int nz, int num_threads)
//{
//	// initialize multi-threading
//	int status = fftw_init_threads();
//	FFTW_CHECK_ERROR(status, "fftw: could not initialize threads")
//
//	// set number of threads to use with following plans
//	fftwf_plan_with_nthreads(num_threads);
//
//	// provide space for result
//	std::vector<std::complex<float>, boost::alignment::aligned_allocator_adaptor<std::allocator<std::complex<float>>,32>> res(nz*ny*((nx/2)+1));
//
//	// make plan
//	fftwf_plan plan = fftwf_plan_dft_r2c_3d(static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx),
//	                                        data, reinterpret_cast<fftwf_complex*>(&res[0]),
//	                                        FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
//
//	// execute plan
//	fftwf_execute(plan);
//
//	// destroy plan
//	fftwf_destroy_plan(plan);
//
//	// clean up
//	fftwf_cleanup_threads();
//
//	// return result
//	return std::move(res);
//}

