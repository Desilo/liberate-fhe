
#include <torch/extension.h>
#include <vector>

// Check types.
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK(x.dtype() == torch::kInt64, #x, " must be a kInt64 tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_LONG(x)


// Forward declaration.
void discrete_gaussian_cuda(torch::Tensor rand_bytes,
                            uint64_t* btree,
                            int btree_size,
                            int depth);

torch::Tensor discrete_gaussian_fast_cuda(torch::Tensor states,
                            uint64_t* btree,
                            int btree_size,
                            int depth,
                            size_t step);



// The main function.
//----------------
// Normal version.
void discrete_gaussian(std::vector<torch::Tensor> inputs,
                       size_t btree_ptr,
                       int btree_size,
                       int depth) {
    
    // reinterpret pointers from numpy.
    uint64_t *btree = reinterpret_cast<uint64_t*>(btree_ptr);
        
    for (auto &rand_bytes : inputs){
        CHECK_INPUT(rand_bytes);

        // Run in cuda.
        discrete_gaussian_cuda(rand_bytes, btree, btree_size, depth);
    }
}

//--------------
// Fast version.

std::vector<torch::Tensor> discrete_gaussian_fast(std::vector<torch::Tensor> states,
                       size_t btree_ptr,
                       int btree_size,
                       int depth,
                       size_t step) {
    
    // reinterpret pointers from numpy.
    uint64_t *btree = reinterpret_cast<uint64_t*>(btree_ptr);
        
    std::vector<torch::Tensor> outputs;
    
    for (auto &my_states : states){
        auto result = discrete_gaussian_fast_cuda(my_states, btree, btree_size, depth, step);
        outputs.push_back(result);
    }
    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("discrete_gaussian", &discrete_gaussian, "discrete gaussian sampling (128 bits).");
    m.def("discrete_gaussian_fast", &discrete_gaussian_fast, "discrete gaussian sampling fast (chacha20 fused, 128 bits).");
}
