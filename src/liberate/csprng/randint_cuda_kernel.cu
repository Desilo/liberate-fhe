#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "chacha20_cuda_kernel.h"

#define COMBINE_TWO(high, low)\
    ((static_cast<uint64_t>(high) << 32) | static_cast<uint64_t>(low))

#define BLOCK_SIZE 64

#define LUT_SIZE 128
__constant__ uint64_t Q[LUT_SIZE];

///////////////////////////////////////////////////////////
// The implementation

//----------------------------------------------------------
// The fast version, chacha20 fused.
//----------------------------------------------------------

__global__ void randint_fast_cuda_kernel(
    torch::PackedTensorAccessor32<int64_t, 3> states,
    torch::PackedTensorAccessor32<int64_t, 2> dst,
    int64_t shift,
    size_t step) {
    
    // Where am I?
    // blockDim -> BLOCK_SIZE_FAST
    // gridDim -> num_rns_channels, num_poly_orders / BLOCK_DIM_FAST
    const int thread_ind = threadIdx.x;
    const int poly_order = blockIdx.y * blockDim.x + threadIdx.x;
    const int rns_channel = blockIdx.x;
    
    __shared__ int64_t x[BLOCK_SIZE][16];
    
     #pragma unroll
    for(int i=0; i<16; ++i){
        x[thread_ind][i] = states[rns_channel][poly_order][i];
    }
    
    // Repeat 10 times for chacha20.
    #pragma unroll
    for(int i=0; i<10; ++i){
        ONE_ROUND(x, thread_ind);
    }
    
    #pragma unroll
    for(int i=0; i<16; ++i){
        x[thread_ind][i] = (states[rns_channel][poly_order][i] + x[thread_ind][i]) & MASK;
    }
    
    // Step the state.
    states[rns_channel][poly_order][12] += step;
    states[rns_channel][poly_order][13] += (states[rns_channel][poly_order][12] >> 32);
    states[rns_channel][poly_order][12] &= MASK;
    
    // Randint.
    #pragma unroll
    for(int i=0; i<16; i+=4){
        
        // What is my Q?
        auto p = Q[rns_channel];

        // Compose into 2 uint64 values, the 4 32 bit values stored in
        // the 4 int64 storage.
        uint64_t x_low = COMBINE_TWO(x[thread_ind][i], x[thread_ind][i+1]);

        // Use CUDA integer intrinsics to calculate
        // (x_low * p) >> 64.
        // Refer to https://github.com/apple/swift/pull/39143
        auto alpha = __umul64hi(p, x_low);

        // We need to calculate carry.
        auto pl = p & MASK;          // 1-32
        auto ph = p >> 32;           // 33-64
        //---------------------------------------
        auto xhh = x[thread_ind][i+2];
        auto xhl = x[thread_ind][i+3];
        //---------------------------------------
        auto plxhl = pl * xhl;       // 65-128
        auto plxhh = pl * xhh;       // 97-160
        auto phxhl = ph * xhl;       // 97-160
        auto phxhh = ph * xhh;       // 129-192
        //---------------------------------------
        auto carry = ((plxhl & MASK) + (alpha & MASK)) >> 32;
        carry = (carry +
                (plxhl >> 32) +
                (alpha >> 32) +
                (phxhl & MASK) +
                (plxhh & MASK)) >> 32;
        auto sample = (carry +
                     (phxhl >> 32) +
                     (plxhh >> 32) + phxhh);

        // Store the result.
        // Don't forget the shift!!!
        const int new_poly_order = poly_order * 4 + i/4;
        dst[rns_channel][new_poly_order] = sample + shift;
    }
}



//----------------------------------------------------------
// The normal version.
//----------------------------------------------------------


// rand_bytes are configured as C x N x 16, where C denotes the q channels.
__global__ void randint_cuda_kernel(torch::PackedTensorAccessor32<int64_t, 3> rand_bytes){
    
    // Where am I?
    const int index = blockIdx.y * blockDim.x + threadIdx.x;
    
    // i is the index of the starting element at the threadIdx.y row.
    const int i = threadIdx.y * 4;
    
    // What is my Q?
    auto p = Q[blockIdx.x];
    
    // Compose into 2 uint64 values, the 4 32 bit values stored in
    // the 4 int64 storage.
    uint64_t x_low = COMBINE_TWO(rand_bytes[blockIdx.x][index][i], rand_bytes[blockIdx.x][index][i+1]);

    // Use CUDA integer intrinsics to calculate
    // (x_low * p) >> 64.
    // Refer to https://github.com/apple/swift/pull/39143
    auto alpha = __umul64hi(p, x_low);
    
    // We need to calculate carry.
    auto pl = p & MASK;          // 1-32
    auto ph = p >> 32;           // 33-64
    //---------------------------------------
    auto xhh = rand_bytes[blockIdx.x][index][i+2];
    auto xhl = rand_bytes[blockIdx.x][index][i+3];
    //---------------------------------------
    auto plxhl = pl * xhl;       // 65-128
    auto plxhh = pl * xhh;       // 97-160
    auto phxhl = ph * xhl;       // 97-160
    auto phxhh = ph * xhh;       // 129-192
    //---------------------------------------
    auto carry = ((plxhl & MASK) + (alpha & MASK)) >> 32;
    carry = (carry +
            (plxhl >> 32) +
            (alpha >> 32) +
            (phxhl & MASK) +
            (plxhh & MASK)) >> 32;
    auto sample = (carry +
                 (phxhl >> 32) +
                 (plxhh >> 32) + phxhh);
    
     // Store the result.
    rand_bytes[blockIdx.x][index][i] = sample;
}

///////////////////////////////////////////////////////////
// The wrapper.

//----------------------------------------------------------
// The fast version, chacha20 fused.
//----------------------------------------------------------

torch::Tensor randint_fast_cuda(torch::Tensor states, uint64_t *q, int64_t shift, size_t step){

    // rand_bytes has the dim C x N x 16.
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid(states.size(0), states.size(1) / dim_block);
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = states.device().index();
    cudaSetDevice(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // Prepare the result.
    // 16 elements in each state turn into 4 random numbers.
    auto result = states.new_empty({states.size(0), states.size(1) * 4});
    
    // Fill in the LUT constant memory.
    cudaMemcpyToSymbol(Q, q, states.size(0) * sizeof(uint64_t));
    
    // Run the cuda kernel.
    auto states_acc = states.packed_accessor32<int64_t, 3>();
    auto result_acc = result.packed_accessor32<int64_t, 2>();
    randint_fast_cuda_kernel<<<dim_grid, dim_block, 0, stream>>>(states_acc, result_acc, shift, step);
    
    return result;
}



//----------------------------------------------------------
// The normal version.
//----------------------------------------------------------


void randint_cuda(torch::Tensor rand_bytes, uint64_t* q){
    
    // rand_bytes has the dim C x N x 16.
    dim3 dim_block(BLOCK_SIZE, 4);
    dim3 dim_grid(rand_bytes.size(0), rand_bytes.size(1) / dim_block.x);
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = rand_bytes.device().index();
    cudaSetDevice(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // Fill in the LUT constant memory.
    cudaMemcpyToSymbol(Q, q, rand_bytes.size(0) * sizeof(uint64_t));
    
    // Run the cuda kernel.
    auto access = rand_bytes.packed_accessor32<int64_t, 3>();
    randint_cuda_kernel<<<dim_grid, dim_block, 0, stream>>>(access);
}
