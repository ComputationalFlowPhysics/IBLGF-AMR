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


#include <iblgf/utilities/convolution_GPU.hpp>

namespace iblgf
{
namespace fft
{
dfft_r2c_gpu::dfft_r2c_gpu(dims_3D _dims_padded, dims_3D _dims_non_zero)
:dims_input_3D(_dims_padded)
, input_(_dims_padded[2] * _dims_padded[1] * _dims_padded[0], 0.0)
, output_(_dims_padded[2] * _dims_padded[1] * ((_dims_padded[0] / 2) + 1))
{
    const int NX = _dims_padded[0];
    const int NY = _dims_padded[1];
    const int NZ = _dims_padded[2];

    const int NX_out = NX / 2 + 1;

    const size_t real_size = sizeof(float_type) * NX * NY * NZ;
    const size_t complex_size = sizeof(std::complex<float_type>) * NX_out * NY * NZ;

    cudaMalloc((void**)&input_cu_, real_size);
    cudaMalloc((void**)&output_cu_, complex_size);

    cufftPlan3d(&plan, NZ, NY, NX, CUFFT_D2Z);
}
// template<class Vector>
// void dfft_r2c_gpu::copy_input(const Vector& _v, dims_3D _dims_v)
// {
//     const int NX = _dims_v[0];
//     const int NY = _dims_v[1];
//     const int NZ = _dims_v[2];

//     const size_t real_size = sizeof(float_type) * NX * NY * NZ;

//     cudaMemcpy(input_cu_, _v.data(), real_size, cudaMemcpyHostToDevice);
// }
template<class Vector>
void dfft_r2c_gpu::copy_input(const Vector& _v, dims_3D _dims_v)
{
    if (_v.size() == input_.size())
    { std::copy(_v.begin(), _v.end(), input_.begin()); }
    else
    {
        std::cout<<" _v.size(): " << _v.size() << " input_.size(): " << input_.size() << std::endl;
        std::cout<<"_dims_v: " << _dims_v << " dims_input_3D: " << dims_input_3D << std::endl;
        throw std::runtime_error("ERROR! LGF SIZE NOT MATCHING");
    }
}

template void dfft_r2c_gpu::copy_input<std::vector<double>>(
    const std::vector<double>& _v, dims_3D _dims_v);

void dfft_r2c_gpu::execute_whole()
{
    cudaMemcpy(input_cu_, input_.data(), input_.size() * sizeof(float_type), cudaMemcpyHostToDevice);
    cufftExecD2Z(plan, (cufftDoubleReal*)input_cu_, (cufftDoubleComplex*)output_cu_);
    // i think we want to add 
    cudaMemcpy(output_.data(), output_cu_, output_.size() * sizeof(std::complex<float_type>), cudaMemcpyDeviceToHost);

}

dfft_r2c_gpu::~dfft_r2c_gpu() {
    // Free device memory if allocated
    if (input_cu_) {
        cudaError_t err = cudaFree(input_cu_);
        if (err != cudaSuccess)
            std::cerr << "cudaFree(input_cu_) failed: " << cudaGetErrorString(err) << "\n";
        input_cu_ = nullptr;
    }

    if (output_cu_) {
        cudaError_t err = cudaFree(output_cu_);
        if (err != cudaSuccess)
            std::cerr << "cudaFree(output_cu_) failed: " << cudaGetErrorString(err) << "\n";
        output_cu_ = nullptr;
    }

    // Destroy cuFFT plan if it exists
    if (plan != 0) {  // plan default-initialized to 0 or CUFFT_INVALID_PLAN
        cufftResult res = cufftDestroy(plan);
        if (res != CUFFT_SUCCESS)
            std::cerr << "cufftDestroy(plan) failed: " << res << "\n";
        plan = 0;
    }
}

void dfft_r2c_gpu::execute()
{
    cudaMemcpy(input_cu_, input_.data(), input_.size() * sizeof(float_type), cudaMemcpyHostToDevice);
    cufftExecD2Z(plan, (cufftDoubleReal*)input_cu_, (cufftDoubleComplex*)output_cu_);
    // i think we want to add 
    cudaMemcpy(output_.data(), output_cu_, output_.size() * sizeof(std::complex<float_type>), cudaMemcpyDeviceToHost);

}

dfft_c2r_gpu::dfft_c2r_gpu(dims_3D _dims, dims_3D _dims_small)
: input_(
      _dims[2] * _dims[1] * ((_dims[0] / 2) + 1), std::complex<float_type>(0.0))
, output_(_dims[2] * _dims[1] * _dims[0], 0.0)
{
    const int NX = _dims[0];
    const int NY = _dims[1];
    const int NZ = _dims[2];

    const int NX_out = NX / 2 + 1;

    const size_t real_size = sizeof(float_type) * NX * NY * NZ;
    const size_t complex_size = sizeof(std::complex<float_type>) * NX_out * NY * NZ;

    cudaMalloc((void**)&input_cu_, complex_size);
    cudaMalloc((void**)&output_cu_, real_size);

    cufftPlan3d(&plan, NZ, NY, NX, CUFFT_Z2D);


}
dfft_c2r_gpu::~dfft_c2r_gpu() {
    // Free device memory if allocated
    if (input_cu_) {
        cudaError_t err = cudaFree(input_cu_);
        if (err != cudaSuccess)
            std::cerr << "cudaFree(input_cu_) failed: " << cudaGetErrorString(err) << "\n";
        input_cu_ = nullptr;
    }

    if (output_cu_) {
        cudaError_t err = cudaFree(output_cu_);
        if (err != cudaSuccess)
            std::cerr << "cudaFree(output_cu_) failed: " << cudaGetErrorString(err) << "\n";
        output_cu_ = nullptr;
    }

    // Destroy cuFFT plan if it exists
    if (plan != 0) {  // plan default-initialized to 0 or CUFFT_INVALID_PLAN
        cufftResult res = cufftDestroy(plan);
        if (res != CUFFT_SUCCESS)
            std::cerr << "cufftDestroy(plan) failed: " << res << "\n";
        plan = 0;
    }
}

void dfft_c2r_gpu::execute()
{
    cudaMemcpy(input_cu_, input_.data(), input_.size() * sizeof(std::complex<float_type>), cudaMemcpyHostToDevice);
    cufftExecZ2D(plan, (cufftDoubleComplex*)input_cu_, (cufftDoubleReal*)output_cu_);
    cudaMemcpy(output_.data(), output_cu_, output_.size() * sizeof(float_type), cudaMemcpyDeviceToHost);
}


} //namespace fft
} // namespace iblgf