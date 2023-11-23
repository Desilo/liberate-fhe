#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "chacha20_cuda_kernel.h"

#define GE(x_high, x_low, y_high, y_low)\
    (((x_high) > (y_high)) | (((x_high) == (y_high)) & ((x_low) >= (y_low))))
    
#define COMBINE_TWO(high, low)\
    ((static_cast<uint64_t>(high) << 32) | static_cast<uint64_t>(low))

// Use 256 BLOCK_SIZE. 64 * 4 = 256.
#define BLOCK_SIZE 64

#define LUT_SIZE 128
__constant__ uint64_t LUT[LUT_SIZE];

///////////////////////////////////////////////////////////
// The implementation

//----------------------------------------------------------
// The fast version, chacha20 fused.
//----------------------------------------------------------

__global__ void discrete_gaussian_fast_cuda_kernel(
    torch::PackedTensorAccessor32<int64_t, 2> states,
    torch::PackedTensorAccessor32<int64_t, 1> dst,
    int btree_size,
    int depth,
    size_t step){
    
    // Where am I?
    const int thread_ind = threadIdx.x;
    const int poly_order = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int64_t x[BLOCK_SIZE][16];
    
     #pragma unroll
    for(int i=0; i<16; ++i){
        x[thread_ind][i] = states[poly_order][i];
    }
    
    // Repeat 10 times for chacha20.
    #pragma unroll
    for(int i=0; i<10; ++i){
        ONE_ROUND(x, thread_ind);
    }
    
    #pragma unroll
    for(int i=0; i<16; ++i){
        x[thread_ind][i] = (states[poly_order][i] + x[thread_ind][i]) & MASK;
    }
    
    // Step the state.
    states[poly_order][12] += step;
    states[poly_order][13] += (states[poly_order][12] >> 32);
    states[poly_order][12] &= MASK;
    
    
    // Discrete gaussian
    for(int i=0; i<16; i+=4){
        // Traverse the tree in the LUT.
        // Note that, out of the 16 32-bit randon numbers,
        // we generate 4 discrete gaussian samples.
        int jump = 1;
        int current = 0;
        int counter = 0;

        // Compose into 2 uint64 values, the 4 32 bit values stored in
        // the 4 int64 storage.
        uint64_t x_low = COMBINE_TWO(x[thread_ind][i], x[thread_ind][i+1]);
        uint64_t x_high = COMBINE_TWO(x[thread_ind][i+2], x[thread_ind][i+3]);

        // Reserve a sign bit.
        // Since we are dealing with the half plane,
        // The CDT values in the LUT are at most 0.5, which means
        // that the values are 127 bits.
        // Also, rigorously speaking, we need to take out the MSB from the x_high
        // value to take the sign, but every bit has probability of occurrence=0.5.
        // Hence, it doesn't matter where we take the bit.
        // For convenience, take the LSB of x_high.
        int64_t sign_bit = x_high & 1;
        x_high >>= 1;

        // Traverse the binary search tree.
        for(int j=0; j<depth; j++){
            int ge_flag = GE(x_high, x_low, 
                          LUT[counter+current+btree_size],
                          LUT[counter+current]);

            // Update the current location.
            current = 2 * current + ge_flag;

            // Update the counter.
            counter += jump;

            // Update the jump
            jump *= 2;    
        }
        int64_t sample = (sign_bit * 2 - 1) * static_cast<int64_t>(current);

        // Store the result.
        const int new_poly_order = poly_order * 4 + i/4;
        dst[new_poly_order] = sample;
    }
}




//----------------------------------------------------------
// The normal version.
//----------------------------------------------------------

// rand_bytes are configured as N x 16.
__global__ void discrete_gaussian_cuda_kernel(
    torch::PackedTensorAccessor32<int64_t, 2> rand_bytes,
    int btree_size,
    int depth){
    // Where am I?
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // i is the index of the starting element at the threadIdx.y row.
    const int i = threadIdx.y * 4;
    
    // Traverse the tree in the LUT.
    // Note that, out of the 16 32-bit randon numbers,
    // we generate 4 discrete gaussian samples.
    int jump = 1;
    int current = 0;
    int counter = 0;

    // Compose into 2 uint64 values, the 4 32 bit values stored in
    // the 4 int64 storage.
    uint64_t x_low = COMBINE_TWO(rand_bytes[index][i], rand_bytes[index][i+1]);
    uint64_t x_high = COMBINE_TWO(rand_bytes[index][i+2], rand_bytes[index][i+3]);

    // Reserve a sign bit.
    // Since we are dealing with the half plane,
    // The CDT values in the LUT are at most 0.5, which means
    // that the values are 127 bits.
    // Also, rigorously speaking, we need to take out the MSB from the x_high
    // value to take the sign, but every bit has probability of occurrence=0.5.
    // Hence, it doesn't matter where we take the bit.
    // For convenience, take the LSB of x_high.
    int64_t sign_bit = x_high & 1;
    x_high >>= 1;

    // Traverse the binary search tree.
    for(int j=0; j<depth; j++){
        int ge_flag = GE(x_high, x_low, 
                      LUT[counter+current+btree_size],
                      LUT[counter+current]);

        // Update the current location.
        current = 2 * current + ge_flag;

        // Update the counter.
        counter += jump;

        // Update the jump
        jump *= 2;    
    }
    int64_t sample = (sign_bit * 2 - 1) * static_cast<int64_t>(current);

    // Store the result.
    rand_bytes[index][i] = sample;
}



///////////////////////////////////////////////////////////
// The wrapper.

//----------------------------------------------------------
// The fast version, chacha20 fused.
//----------------------------------------------------------

torch::Tensor discrete_gaussian_fast_cuda(torch::Tensor states,
                            uint64_t* btree,
                            int btree_size,
                            int depth,
                            size_t step){
    
    // rand_bytes has the dim N x 16.
    int dim_block = BLOCK_SIZE;
    int dim_grid = states.size(0) / dim_block;
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = states.device().index();
    cudaSetDevice(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // Prepare the result.
    // 16 elements in each state turn into 4 random numbers.
    auto result = states.new_empty({states.size(0) * 4});
    
    // Fill in the LUT constant memory.
    cudaMemcpyToSymbol(LUT, btree, btree_size * 2 * sizeof(uint64_t));
    
    // Run the cuda kernel.
    auto states_acc = states.packed_accessor32<int64_t, 2>();
    auto result_acc = result.packed_accessor32<int64_t, 1>();
    discrete_gaussian_fast_cuda_kernel<<<dim_grid, dim_block, 0, stream>>>(
        states_acc, result_acc, btree_size, depth, step);
    
    return result;
}


//----------------------------------------------------------
// The normal version.
//----------------------------------------------------------

void discrete_gaussian_cuda(torch::Tensor rand_bytes,
                            uint64_t* btree,
                            int btree_size,
                            int depth){
    
    // rand_bytes has the dim N x 16.
    dim3 dim_block(BLOCK_SIZE, 4);
    int dim_grid = rand_bytes.size(0) / dim_block.x;
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = rand_bytes.device().index();
    cudaSetDevice(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // Fill in the LUT constant memory.
    cudaMemcpyToSymbol(LUT, btree, btree_size * 2 * sizeof(uint64_t));
    
    // Run the cuda kernel.
    auto access = rand_bytes.packed_accessor32<int64_t, 2>();
    discrete_gaussian_cuda_kernel<<<dim_grid, dim_block, 0, stream>>>(access, btree_size, depth);
}
