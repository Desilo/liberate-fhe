#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void randint_cuda_kernel(torch::PackedTensorAccessor32<double, 1> input,
                                    torch::PackedTensorAccessor32<int64_t, 1> rand_bytes){
    
    // Where am I?
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    auto coef = input[index];
    auto sign_bit = signbit(coef);
    auto abs_coef = fabs(coef);
    
    auto integ = floor(abs_coef);
    auto frac = abs_coef - integ;
    int64_t intinteg = static_cast<int64_t>(integ);
    
    // Convert a double to a signed 64-bit int in round-to-nearest-even mode.
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__CAST.html#group__CUDA__MATH__INTRINSIC__CAST
    constexpr double rounder = static_cast<double>(0x100000000);
    int64_t ifrac = __double2ll_rn(frac * rounder);
    
    // Random round.
    // The bool value must be 1 for True.
    int64_t round = rand_bytes[index] < ifrac;
    
    // Round and recover sign.
    int64_t sign = (sign_bit)? -1 : 1;
    int64_t rounded = sign * (intinteg + round);
    
    // Put back into the rand_bytes.
    rand_bytes[index] = rounded;
}

    
// The wrapper.
void randround_cuda(torch::Tensor input, torch::Tensor rand_bytes){
    
    // rand_bytes has the dim C x N x 16.
    const int dim_block = BLOCK_SIZE;
    const int dim_grid = rand_bytes.size(0) / dim_block;
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = rand_bytes.device().index();
    cudaSetDevice(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // Run the cuda kernel.
    auto input_access = input.packed_accessor32<double, 1>();
    auto rand_bytes_access = rand_bytes.packed_accessor32<int64_t, 1>();
    randint_cuda_kernel<<<dim_grid, dim_block, 0, stream>>>(input_access, rand_bytes_access);
}
