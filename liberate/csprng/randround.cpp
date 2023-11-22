
#include <torch/extension.h>
#include <vector>

// Check types.
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK(x.dtype() == torch::kInt64, #x, " must be a kInt64 tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_LONG(x)


// Forward declaration.
void randround_cuda(torch::Tensor input, torch::Tensor rand_bytes);

// The main function.
// rand_bytes are N 1D uint64_t tensors.
// Inputs are N 1D double tensors.
// The output will be returned in rand_bytes.
void randround(std::vector<torch::Tensor> inputs,
               std::vector<torch::Tensor> rand_bytes) {
    
    for (auto i=0; i<inputs.size(); i++){
        CHECK_INPUT(rand_bytes[i]);
        
        // Run in cuda.
        randround_cuda(inputs[i], rand_bytes[i]);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("randround", &randround, "random rounding.");
}
