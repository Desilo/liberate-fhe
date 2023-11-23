#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "chacha20_cuda_kernel.h"

#define BLOCK_SIZE 256

__global__ void chacha20_cuda_kernel(
    torch::PackedTensorAccessor32<int64_t, 2> input,
    torch::PackedTensorAccessor32<int64_t, 2> dest,
    size_t step) {
    
    // input is configured as N x 16
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int64_t x[BLOCK_SIZE][16];
    
    #pragma unroll
    for(int i=0; i<16; ++i){
        x[threadIdx.x][i] = dest[index][i];
        
        // Vectorized load.
        // Not much beneficial for our case, though ...
        //reinterpret_cast<int4*>(x[threadIdx.x])[i] =
        //    *reinterpret_cast<int4*>(&(dest[index][i]));
    }
    
    // Repeat 10 times for chacha20.
    #pragma unroll
    for(int i=0; i<10; ++i){
        ONE_ROUND(x, threadIdx.x);
    }
    
    #pragma unroll
    for(int i=0; i<16; ++i){
        dest[index][i] = (dest[index][i] + x[threadIdx.x][i]) & MASK;
    }
    
    // Step the state.
    input[index][12] += step;
    input[index][13] += (input[index][12] >> 32);
    input[index][12] &= MASK;
}
    
    
// The wrapper.
void chacha20_cuda(torch::Tensor input, torch::Tensor dest, size_t step){
    
    // Required number of blocks in a grid.
    // Note that we do not use grids here, since the
    // tensor we're dealing with must be chopped in 1-d.
    // input is configured as 16 x N
    // N must be a multitude of 1024.
    
    const int dim_block = BLOCK_SIZE;
    int dim_grid = input.size(0) / dim_block;
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = input.device().index();
    cudaSetDevice(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // Run the cuda kernel.
    auto input_acc = input.packed_accessor32<int64_t, 2>();
    auto dest_acc = dest.packed_accessor32<int64_t, 2>();
    chacha20_cuda_kernel<<<dim_grid, dim_block, 0, stream>>>(input_acc, dest_acc, step);
}
