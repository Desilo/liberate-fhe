#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

//------------------------------------------------------------------
// pointwise mont_mult
//------------------------------------------------------------------

template<typename scalar_t> __device__ __forceinline__ scalar_t
mont_mult_scalar_cuda_kernel(
    const scalar_t a, const scalar_t b,
    const scalar_t ql, const scalar_t qh,
    const scalar_t kl, const scalar_t kh) {
    
    // Masks.
    constexpr scalar_t one = 1;
    constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
    constexpr scalar_t half_nbits =  sizeof(scalar_t) * 4 - 1;
    constexpr scalar_t fb_mask = ((one << nbits) - one);
    constexpr scalar_t lb_mask = (one << half_nbits) - one;
    
    const scalar_t al = a & lb_mask;
    const scalar_t ah = a >> half_nbits;
    const scalar_t bl = b & lb_mask;
    const scalar_t bh = b >> half_nbits;

    const scalar_t alpha = ah * bh;
    const scalar_t beta = ah * bl + al * bh;
    const scalar_t gamma = al * bl;

    // s = xk mod R
    const scalar_t gammal = gamma & lb_mask;
    const scalar_t gammah = gamma >> half_nbits;
    const scalar_t betal = beta & lb_mask;
    const scalar_t betah = beta >> half_nbits;

    scalar_t upper = gammal * kh;
    upper = upper + (gammah + betal) * kl;
    upper = upper << half_nbits;
    scalar_t s = upper + gammal * kl;
    s = upper + gammal * kl;
    s = s & fb_mask;

    // t = x + sq
    // u = t/R
    const scalar_t sl = s & lb_mask;
    const scalar_t sh = s >> half_nbits;
    const scalar_t sqb = sh * ql + sl * qh;
    const scalar_t sqbl = sqb & lb_mask;
    const scalar_t sqbh = sqb >> half_nbits;

    scalar_t carry = (gamma + sl * ql) >> half_nbits;
    carry = (carry + betal + sqbl) >> half_nbits;
    
    return alpha + betah + sqbh + carry + sh * qh;
}


//------------------------------------------------------------------
// mont_mult
//------------------------------------------------------------------

template<typename scalar_t>
__global__ void mont_mult_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2>a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 2>b_acc,
    torch::PackedTensorAccessor32<scalar_t, 2>c_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>kh_acc){
    
    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    
    // Inputs.
    const scalar_t a = a_acc[i][j];
    const scalar_t b = b_acc[i][j];
    const scalar_t ql = ql_acc[i];
    const scalar_t qh = qh_acc[i];
    const scalar_t kl = kl_acc[i];
    const scalar_t kh = kh_acc[i];
    
    // Store the result.
    c_acc[i][j] = mont_mult_scalar_cuda_kernel(a, b, ql, qh, kl, kh);
}

template<typename scalar_t>
void mont_mult_cuda_typed(
    const torch::Tensor a,
    const torch::Tensor b,
    torch::Tensor c,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    
    // Use a preallocated pytorch stream.
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // The problem dimension.
    auto C = a.size(0);
    auto N = a.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid (C, N / BLOCK_SIZE);
    
    // Run the cuda kernel.
    const auto a_acc = a.packed_accessor32<scalar_t, 2>();
    const auto b_acc = b.packed_accessor32<scalar_t, 2>();
    auto c_acc = c.packed_accessor32<scalar_t, 2>();
    const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
    const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
    const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
    const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();
    mont_mult_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
        a_acc, b_acc, c_acc, ql_acc, qh_acc, kl_acc, kh_acc);
}


torch::Tensor mont_mult_cuda(
    const torch::Tensor a,
    const torch::Tensor b,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
        
    // Prepare the output.
    torch::Tensor c = torch::empty_like(a);
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_mont_mult_cuda", ([&] {
    mont_mult_cuda_typed<scalar_t>(a, b, c, ql, qh, kl, kh);
    }));
    
    return c;
}



//------------------------------------------------------------------
// mont_enter
//------------------------------------------------------------------

template<typename scalar_t>
__global__ void mont_enter_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2>a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>Rs_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>kh_acc){
    
    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    
    // Inputs.
    const scalar_t a = a_acc[i][j];
    const scalar_t Rs = Rs_acc[i];
    const scalar_t ql = ql_acc[i];
    const scalar_t qh = qh_acc[i];
    const scalar_t kl = kl_acc[i];
    const scalar_t kh = kh_acc[i];
    
    // Store the result.
    a_acc[i][j] = mont_mult_scalar_cuda_kernel(a, Rs, ql, qh, kl, kh);
}


template<typename scalar_t>
void mont_enter_cuda_typed(
    torch::Tensor a,
    const torch::Tensor Rs,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    
    // Use a preallocated pytorch stream.
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // The problem dimension.
    auto C = a.size(0);
    auto N = a.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid (C, N / BLOCK_SIZE);
    
    // Run the cuda kernel.
    auto a_acc = a.packed_accessor32<scalar_t, 2>();
    const auto Rs_acc = Rs.packed_accessor32<scalar_t, 1>();
    const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
    const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
    const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
    const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();
    mont_enter_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
        a_acc, Rs_acc, ql_acc, qh_acc, kl_acc, kh_acc);
}

void mont_enter_cuda(
    torch::Tensor a,
    const torch::Tensor Rs,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_mont_enter_cuda", ([&] {
    mont_enter_cuda_typed<scalar_t>(a, Rs, ql, qh, kl, kh);
    }));
}





//------------------------------------------------------------------
// ntt
//------------------------------------------------------------------

template<typename scalar_t>
__global__ void ntt_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2>a_acc,
    const torch::PackedTensorAccessor32<int, 2>even_acc,
    const torch::PackedTensorAccessor32<int, 2>odd_acc,
    const torch::PackedTensorAccessor32<scalar_t, 3>psi_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>_2q_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>kh_acc,
    const int level){
    
    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    
    // Montgomery inputs.
    const scalar_t _2q = _2q_acc[i];
    const scalar_t ql = ql_acc[i];
    const scalar_t qh = qh_acc[i];
    const scalar_t kl = kl_acc[i];
    const scalar_t kh = kh_acc[i];
    
    // Butterfly.
    const int even_j = even_acc[level][j];
    const int odd_j = odd_acc[level][j];
    
    const scalar_t U = a_acc[i][even_j];
    const scalar_t S = psi_acc[i][level][j];
    const scalar_t O = a_acc[i][odd_j];
    const scalar_t V = mont_mult_scalar_cuda_kernel(S, O, ql, qh, kl, kh);
    
    // Store back.
    const scalar_t UplusV = U + V;
    const scalar_t UminusV = U + _2q - V;
    
    a_acc[i][even_j] = (UplusV < _2q)? UplusV : UplusV - _2q;
    a_acc[i][odd_j] = (UminusV < _2q)? UminusV : UminusV - _2q;
}


template<typename scalar_t>
void ntt_cuda_typed(
    torch::Tensor a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    
    // Use a preallocated pytorch stream.
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // The problem dimension.
    const auto C = ql.size(0);
    const auto logN = even.size(0);
    const auto N = even.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid (C, N / BLOCK_SIZE);
    
    // Run the cuda kernel.
    auto a_acc = a.packed_accessor32<scalar_t, 2>();
    
    const auto even_acc = even.packed_accessor32<int, 2>();
    const auto odd_acc = odd.packed_accessor32<int, 2>();
    const auto psi_acc = psi.packed_accessor32<scalar_t, 3>();
    
    const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
    const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
    const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
    const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
    const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();
    
    for(int i=0; i<logN; ++i){
        ntt_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
        a_acc, even_acc, odd_acc, psi_acc,
        _2q_acc, ql_acc, qh_acc, kl_acc, kh_acc, i);
    }
}



void ntt_cuda(
    torch::Tensor a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_ntt_cuda", ([&] {
    ntt_cuda_typed<scalar_t>(a, even, odd, psi, _2q, ql, qh, kl, kh);
    }));
}


//------------------------------------------------------------------
// enter_ntt
//------------------------------------------------------------------

template<typename scalar_t>
void enter_ntt_cuda_typed(
    torch::Tensor a,
    const torch::Tensor Rs,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    
    // Use a preallocated pytorch stream.
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // The problem dimension.
    // Be careful. even and odd has half the length of the a.
    const auto C = ql.size(0);
    const auto logN = even.size(0);
    const auto N_half = even.size(1);
    const auto N = a.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid_ntt (C, N_half / BLOCK_SIZE);
    dim3 dim_grid_enter (C, N / BLOCK_SIZE);
    
    // Run the cuda kernel.
    auto a_acc = a.packed_accessor32<scalar_t, 2>();
    const auto Rs_acc = Rs.packed_accessor32<scalar_t, 1>();
    
    const auto even_acc = even.packed_accessor32<int, 2>();
    const auto odd_acc = odd.packed_accessor32<int, 2>();
    const auto psi_acc = psi.packed_accessor32<scalar_t, 3>();
    
    const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
    const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
    const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
    const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
    const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();
    
    // enter.
    mont_enter_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block, 0, stream>>>(
        a_acc, Rs_acc, ql_acc, qh_acc, kl_acc, kh_acc);

    // ntt.
    for(int i=0; i<logN; ++i){
        ntt_cuda_kernel<scalar_t><<<dim_grid_ntt, dim_block, 0, stream>>>(
        a_acc, even_acc, odd_acc, psi_acc,
        _2q_acc, ql_acc, qh_acc, kl_acc, kh_acc, i);
    }
}


void enter_ntt_cuda(
    torch::Tensor a,
    const torch::Tensor Rs,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_enter_ntt_cuda", ([&] {
    enter_ntt_cuda_typed<scalar_t>(a, Rs, even, odd, psi, _2q, ql, qh, kl, kh);
    }));
}





//------------------------------------------------------------------
// intt
//------------------------------------------------------------------

template<typename scalar_t>
__global__ void intt_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2>a_acc,
    const torch::PackedTensorAccessor32<int, 2>even_acc,
    const torch::PackedTensorAccessor32<int, 2>odd_acc,
    const torch::PackedTensorAccessor32<scalar_t, 3>psi_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>_2q_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>kh_acc,
    const int level){
    
    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    
    // Montgomery inputs.
    const scalar_t _2q = _2q_acc[i];
    const scalar_t ql = ql_acc[i];
    const scalar_t qh = qh_acc[i];
    const scalar_t kl = kl_acc[i];
    const scalar_t kh = kh_acc[i];
    
    // Butterfly.
    const int even_j = even_acc[level][j];
    const int odd_j = odd_acc[level][j];
    
    const scalar_t U = a_acc[i][even_j];
    const scalar_t S = psi_acc[i][level][j];
    const scalar_t V = a_acc[i][odd_j];
    
    const scalar_t UminusV = U + _2q - V;
    const scalar_t O = (UminusV < _2q)? UminusV : UminusV - _2q;
    
    const scalar_t W = mont_mult_scalar_cuda_kernel(S, O, ql, qh, kl, kh);
    a_acc[i][odd_j] = W;
    
    const scalar_t UplusV = U + V;
    a_acc[i][even_j] = (UplusV < _2q)? UplusV : UplusV - _2q;
}


template<typename scalar_t>
void intt_cuda_typed(
    torch::Tensor a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor Ninv,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    
    // Use a preallocated pytorch stream.
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // The problem dimension.
    // Be careful. even and odd has half the length of the a.
    const auto C = ql.size(0);
    const auto logN = even.size(0);
    const auto N_half = even.size(1);
    const auto N = a.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid_ntt (C, N_half / BLOCK_SIZE);
    dim3 dim_grid_enter (C, N / BLOCK_SIZE);
    
    // Run the cuda kernel.
    auto a_acc = a.packed_accessor32<scalar_t, 2>();
    
    const auto even_acc = even.packed_accessor32<int, 2>();
    const auto odd_acc = odd.packed_accessor32<int, 2>();
    const auto psi_acc = psi.packed_accessor32<scalar_t, 3>();
    const auto Ninv_acc = Ninv.packed_accessor32<scalar_t, 1>();
    
    const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
    const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
    const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
    const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
    const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();
    
    for(int i=0; i<logN; ++i){
        intt_cuda_kernel<scalar_t><<<dim_grid_ntt, dim_block, 0, stream>>>(
        a_acc, even_acc, odd_acc, psi_acc,
        _2q_acc, ql_acc, qh_acc, kl_acc, kh_acc, i);
    }
    
    // Normalize.
    mont_enter_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block, 0, stream>>>(
        a_acc, Ninv_acc, ql_acc, qh_acc, kl_acc, kh_acc);
}

void intt_cuda(
    torch::Tensor a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor Ninv,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_intt_cuda", ([&] {
    intt_cuda_typed<scalar_t>(a, even, odd, psi, Ninv, _2q, ql, qh, kl, kh);
    }));
}






//------------------------------------------------------------------
// mont_redc
//------------------------------------------------------------------

template<typename scalar_t>
__global__ void mont_redc_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2>a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>ql_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>qh_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>kl_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>kh_acc){
    
    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    
    // Masks.
    constexpr scalar_t one = 1;
    constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
    constexpr scalar_t half_nbits =  sizeof(scalar_t) * 4 - 1;
    constexpr scalar_t fb_mask = ((one << nbits) - one);
    constexpr scalar_t lb_mask = (one << half_nbits) - one;
    
    // Inputs.
    const scalar_t x = a_acc[i][j];
    const scalar_t ql = ql_acc[i];
    const scalar_t qh = qh_acc[i];
    const scalar_t kl = kl_acc[i];
    const scalar_t kh = kh_acc[i];
    
    // Implementation.
    // s= xk mod R
    const scalar_t xl = x & lb_mask;
    const scalar_t xh = x >> half_nbits;
    const scalar_t xkb = xh * kl + xl * kh;
    scalar_t s = (xkb << half_nbits) + xl * kl;
    s = s & fb_mask;

    // t = x + sq
    // u = t/R
    // Note that x gets erased in t/R operation if x < R.
    const scalar_t sl = s & lb_mask;
    const scalar_t sh = s >> half_nbits;
    const scalar_t sqb = sh * ql + sl * qh;
    const scalar_t sqbl = sqb & lb_mask;
    const scalar_t sqbh = sqb >> half_nbits;
    scalar_t carry = (x + sl * ql) >> half_nbits;
    carry = (carry + sqbl) >> half_nbits;
    
    // Assume we have satisfied the condition 4*q < R.
    // Return the calculated value directly without conditional subtraction.
    a_acc[i][j] = sqbh + carry + sh * qh;
}


template<typename scalar_t>
void mont_redc_cuda_typed(
    torch::Tensor a,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    
    // Use a preallocated pytorch stream.
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // The problem dimension.
    auto C = a.size(0);
    auto N = a.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid (C, N / BLOCK_SIZE);
    
    // Run the cuda kernel.
    auto a_acc = a.packed_accessor32<scalar_t, 2>();
    const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
    const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
    const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
    const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();
    mont_redc_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
        a_acc, ql_acc, qh_acc, kl_acc, kh_acc);
}

void mont_redc_cuda(
    torch::Tensor a,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_mont_redc_cuda", ([&] {
    mont_redc_cuda_typed<scalar_t>(a, ql, qh, kl, kh);
    }));
}


//------------------------------------------------------------------
// Chained intt series.
//------------------------------------------------------------------

/**************************************************************/
/* CUDA kernels                                               */
/**************************************************************/

template<typename scalar_t>
__global__ void reduce_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2>a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>_2q_acc){
    
    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    
    // Inputs.
    constexpr scalar_t one = 1;
    const scalar_t a = a_acc[i][j];
    const scalar_t q = _2q_acc[i] >> one;
    
    // Reduce.
    a_acc[i][j] = (a < q)? a : a - q;
}

template<typename scalar_t>
__global__ void make_signed_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2>a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>_2q_acc){
    
    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    
    // Inputs.
    constexpr scalar_t one = 1;
    const scalar_t a = a_acc[i][j];
    const scalar_t q = _2q_acc[i] >> one;
    const scalar_t q_half = q >> one;
    
    // Make signed.
    a_acc[i][j] = (a <= q_half)? a : a - q;
}


/**************************************************************/
/* Typed functions                                            */
/**************************************************************/

///////////////////////////////////////////////////////////////
// intt exit

template<typename scalar_t>
void intt_exit_cuda_typed(
    torch::Tensor a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor Ninv,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    
    // Use a preallocated pytorch stream.
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // The problem dimension.
    // Be careful. even and odd has half the length of the a.
    const auto C = ql.size(0);
    const auto logN = even.size(0);
    const auto N_half = even.size(1);
    const auto N = a.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid_ntt (C, N_half / BLOCK_SIZE);
    dim3 dim_grid_enter (C, N / BLOCK_SIZE);
    
    // Run the cuda kernel.
    auto a_acc = a.packed_accessor32<scalar_t, 2>();
    
    const auto even_acc = even.packed_accessor32<int, 2>();
    const auto odd_acc = odd.packed_accessor32<int, 2>();
    const auto psi_acc = psi.packed_accessor32<scalar_t, 3>();
    const auto Ninv_acc = Ninv.packed_accessor32<scalar_t, 1>();
    
    const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
    const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
    const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
    const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
    const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();
    
    for(int i=0; i<logN; ++i){
        intt_cuda_kernel<scalar_t><<<dim_grid_ntt, dim_block, 0, stream>>>(
        a_acc, even_acc, odd_acc, psi_acc,
        _2q_acc, ql_acc, qh_acc, kl_acc, kh_acc, i);
    }
    
    // Normalize.
    mont_enter_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block, 0, stream>>>(
        a_acc, Ninv_acc, ql_acc, qh_acc, kl_acc, kh_acc);
    
    // Exit.
    mont_redc_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block, 0, stream>>>(
        a_acc, ql_acc, qh_acc, kl_acc, kh_acc);
}

///////////////////////////////////////////////////////////////
// intt exit reduce

template<typename scalar_t>
void intt_exit_reduce_cuda_typed(
    torch::Tensor a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor Ninv,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    
    // Use a preallocated pytorch stream.
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // The problem dimension.
    // Be careful. even and odd has half the length of the a.
    const auto C = ql.size(0);
    const auto logN = even.size(0);
    const auto N_half = even.size(1);
    const auto N = a.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid_ntt (C, N_half / BLOCK_SIZE);
    dim3 dim_grid_enter (C, N / BLOCK_SIZE);
    
    // Run the cuda kernel.
    auto a_acc = a.packed_accessor32<scalar_t, 2>();
    
    const auto even_acc = even.packed_accessor32<int, 2>();
    const auto odd_acc = odd.packed_accessor32<int, 2>();
    const auto psi_acc = psi.packed_accessor32<scalar_t, 3>();
    const auto Ninv_acc = Ninv.packed_accessor32<scalar_t, 1>();
    
    const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
    const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
    const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
    const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
    const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();
    
    for(int i=0; i<logN; ++i){
        intt_cuda_kernel<scalar_t><<<dim_grid_ntt, dim_block, 0, stream>>>(
        a_acc, even_acc, odd_acc, psi_acc,
        _2q_acc, ql_acc, qh_acc, kl_acc, kh_acc, i);
    }
    
    // Normalize.
    mont_enter_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block, 0, stream>>>(
        a_acc, Ninv_acc, ql_acc, qh_acc, kl_acc, kh_acc);
    
    // Exit.
    mont_redc_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block, 0, stream>>>(
        a_acc, ql_acc, qh_acc, kl_acc, kh_acc);
    
    // Reduce.
    reduce_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block, 0, stream>>>(a_acc, _2q_acc);
}


///////////////////////////////////////////////////////////////
// intt exit reduce signed

template<typename scalar_t>
void intt_exit_reduce_signed_cuda_typed(
    torch::Tensor a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor Ninv,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Retrieve the device index, then set the corresponding device and stream.
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    
    // Use a preallocated pytorch stream.
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    // The problem dimension.
    // Be careful. even and odd has half the length of the a.
    const auto C = ql.size(0);
    const auto logN = even.size(0);
    const auto N_half = even.size(1);
    const auto N = a.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid_ntt (C, N_half / BLOCK_SIZE);
    dim3 dim_grid_enter (C, N / BLOCK_SIZE);
    
    // Run the cuda kernel.
    auto a_acc = a.packed_accessor32<scalar_t, 2>();
    
    const auto even_acc = even.packed_accessor32<int, 2>();
    const auto odd_acc = odd.packed_accessor32<int, 2>();
    const auto psi_acc = psi.packed_accessor32<scalar_t, 3>();
    const auto Ninv_acc = Ninv.packed_accessor32<scalar_t, 1>();
    
    const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
    const auto ql_acc = ql.packed_accessor32<scalar_t, 1>();
    const auto qh_acc = qh.packed_accessor32<scalar_t, 1>();
    const auto kl_acc = kl.packed_accessor32<scalar_t, 1>();
    const auto kh_acc = kh.packed_accessor32<scalar_t, 1>();
    
    for(int i=0; i<logN; ++i){
        intt_cuda_kernel<scalar_t><<<dim_grid_ntt, dim_block, 0, stream>>>(
        a_acc, even_acc, odd_acc, psi_acc,
        _2q_acc, ql_acc, qh_acc, kl_acc, kh_acc, i);
    }
    
    // Normalize.
    mont_enter_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block, 0, stream>>>(
        a_acc, Ninv_acc, ql_acc, qh_acc, kl_acc, kh_acc);
    
    // Exit.
    mont_redc_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block, 0, stream>>>(
        a_acc, ql_acc, qh_acc, kl_acc, kh_acc);
    
    // Reduce.
    reduce_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block, 0, stream>>>(a_acc, _2q_acc);
    
    // Make signed.
    make_signed_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block, 0, stream>>>(a_acc, _2q_acc);
}




/**************************************************************/
/* Connectors                                                 */
/**************************************************************/

///////////////////////////////////////////////////////////////
// intt exit

void intt_exit_cuda(
    torch::Tensor a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor Ninv,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_intt_exit_cuda", ([&] {
    intt_exit_cuda_typed<scalar_t>(a, even, odd, psi, Ninv, _2q, ql, qh, kl, kh);
    }));
}

///////////////////////////////////////////////////////////////
// intt exit reduce

void intt_exit_reduce_cuda(
    torch::Tensor a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor Ninv,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_intt_exit_reduce_cuda", ([&] {
    intt_exit_reduce_cuda_typed<scalar_t>(a, even, odd, psi, Ninv, _2q, ql, qh, kl, kh);
    }));
}

///////////////////////////////////////////////////////////////
// intt exit reduce signed

void intt_exit_reduce_signed_cuda(
    torch::Tensor a,
    const torch::Tensor even,
    const torch::Tensor odd,
    const torch::Tensor psi,
    const torch::Tensor Ninv,
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
    
    // Dispatch to the correct data type.
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_intt_exit_reduce_signed_cuda", ([&] {
    intt_exit_reduce_signed_cuda_typed<scalar_t>(a, even, odd, psi, Ninv, _2q, ql, qh, kl, kh);
    }));
}


//------------------------------------------------------------------
// Misc
//------------------------------------------------------------------

template<typename scalar_t>
__global__ void make_unsigned_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2>a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>_2q_acc){
    
    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    
    // Inputs.
    constexpr scalar_t one = 1;
    const scalar_t q = _2q_acc[i] >> one;
    
    // Make unsigned.
    a_acc[i][j] += q;
}

template<typename scalar_t>
__global__ void tile_unsigned_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1>a_acc,
    torch::PackedTensorAccessor32<scalar_t, 2>dst_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>_2q_acc){
    
    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    
    // Inputs.
    constexpr scalar_t one = 1;
    const scalar_t q = _2q_acc[i] >> one;
    const scalar_t a = a_acc[j];
    
    // Make unsigned.
    dst_acc[i][j] = a + q;
}

template<typename scalar_t>
__global__ void mont_add_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2>a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 2>b_acc,
    torch::PackedTensorAccessor32<scalar_t, 2>c_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>_2q_acc){
    
    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    
    // Inputs.
    constexpr scalar_t one = 1;
    const scalar_t a = a_acc[i][j];
    const scalar_t b = b_acc[i][j];
    const scalar_t _2q = _2q_acc[i];
    
    // Add.
    const scalar_t aplusb = a + b;
    c_acc[i][j] = (aplusb < _2q)? aplusb : aplusb - _2q;
}

template<typename scalar_t>
__global__ void mont_sub_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2>a_acc,
    const torch::PackedTensorAccessor32<scalar_t, 2>b_acc,
    torch::PackedTensorAccessor32<scalar_t, 2>c_acc,
    const torch::PackedTensorAccessor32<scalar_t, 1>_2q_acc){
    
    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    
    // Inputs.
    constexpr scalar_t one = 1;
    const scalar_t a = a_acc[i][j];
    const scalar_t b = b_acc[i][j];
    const scalar_t _2q = _2q_acc[i];
    
    // Sub.
    const scalar_t aminusb = a + _2q - b;
    c_acc[i][j] = (aminusb < _2q)? aminusb : aminusb - _2q;
}

template<typename scalar_t>
void reduce_2q_cuda_typed(torch::Tensor a, const torch::Tensor _2q) {
    
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    const auto C = a.size(0);
    const auto N = a.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid (C, N / BLOCK_SIZE);
    
    auto a_acc = a.packed_accessor32<scalar_t, 2>();
    const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
    
    reduce_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(a_acc, _2q_acc);
}

template<typename scalar_t>
void make_signed_cuda_typed(torch::Tensor a, const torch::Tensor _2q) {
    
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    const auto C = a.size(0);
    const auto N = a.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid (C, N / BLOCK_SIZE);
    
    auto a_acc = a.packed_accessor32<scalar_t, 2>();
    const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();

    make_signed_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(a_acc, _2q_acc);
}

template<typename scalar_t>
void tile_unsigned_cuda_typed(const torch::Tensor a, torch::Tensor dst, const torch::Tensor _2q) {
    
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    const auto C = _2q.size(0);
    const auto N = a.size(0);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid (C, N / BLOCK_SIZE);
    
    const auto a_acc = a.packed_accessor32<scalar_t, 1>();
    auto dst_acc = dst.packed_accessor32<scalar_t, 2>();
    const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();

    tile_unsigned_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(a_acc, dst_acc, _2q_acc);
}

template<typename scalar_t>
void make_unsigned_cuda_typed(torch::Tensor a, const torch::Tensor _2q) {
    
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    const auto C = a.size(0);
    const auto N = a.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid (C, N / BLOCK_SIZE);
    
    auto a_acc = a.packed_accessor32<scalar_t, 2>();
    const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();

    make_unsigned_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(a_acc, _2q_acc);
}

template<typename scalar_t>
void mont_add_cuda_typed(
    const torch::Tensor a,
    const torch::Tensor b,
    torch::Tensor c,
    const torch::Tensor _2q) {
    
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    auto C = a.size(0);
    auto N = a.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid (C, N / BLOCK_SIZE);
    
    // Run the cuda kernel.
    const auto a_acc = a.packed_accessor32<scalar_t, 2>();
    const auto b_acc = b.packed_accessor32<scalar_t, 2>();
    auto c_acc = c.packed_accessor32<scalar_t, 2>();
    const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
    mont_add_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(a_acc, b_acc, c_acc, _2q_acc);
}

template<typename scalar_t>
void mont_sub_cuda_typed(
    const torch::Tensor a,
    const torch::Tensor b,
    torch::Tensor c,
    const torch::Tensor _2q) {
    
    auto device_id = a.device().index();
    cudaSetDevice(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    
    auto C = a.size(0);
    auto N = a.size(1);
    
    int dim_block = BLOCK_SIZE;
    dim3 dim_grid (C, N / BLOCK_SIZE);
    
    // Run the cuda kernel.
    const auto a_acc = a.packed_accessor32<scalar_t, 2>();
    const auto b_acc = b.packed_accessor32<scalar_t, 2>();
    auto c_acc = c.packed_accessor32<scalar_t, 2>();
    const auto _2q_acc = _2q.packed_accessor32<scalar_t, 1>();
    mont_sub_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(a_acc, b_acc, c_acc, _2q_acc);
}

void reduce_2q_cuda(torch::Tensor a, const torch::Tensor _2q) {
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_reduce_2q_cuda", ([&] {
    reduce_2q_cuda_typed<scalar_t>(a, _2q);
    }));
}

void make_signed_cuda(torch::Tensor a, const torch::Tensor _2q) {
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_make_signed_cuda", ([&] {
    make_signed_cuda_typed<scalar_t>(a, _2q);
    }));
}

void make_unsigned_cuda(torch::Tensor a, const torch::Tensor _2q) {
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_make_unsigned_cuda", ([&] {
    make_unsigned_cuda_typed<scalar_t>(a, _2q);
    }));
}

torch::Tensor tile_unsigned_cuda(const torch::Tensor a, const torch::Tensor _2q) {
    a.squeeze_();
    const auto C = _2q.size(0);
    const auto N = a.size(0);
    auto c = a.new_empty({C, N});
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_tile_unsigned_cuda", ([&] {
    tile_unsigned_cuda_typed<scalar_t>(a, c, _2q);
    }));
    return c;
}

torch::Tensor mont_add_cuda(const torch::Tensor a, const torch::Tensor b, const torch::Tensor _2q) {
    torch::Tensor c = torch::empty_like(a);
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_mont_add_cuda", ([&] {
    mont_add_cuda_typed<scalar_t>(a, b, c, _2q);
    }));
    return c;
}

torch::Tensor mont_sub_cuda(const torch::Tensor a, const torch::Tensor b, const torch::Tensor _2q) {
    torch::Tensor c = torch::empty_like(a);
    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "typed_mont_sub_cuda", ([&] {
    mont_sub_cuda_typed<scalar_t>(a, b, c, _2q);
    }));
    return c;
}
