#include <torch/extension.h>
#include <vector>

// chacha20 is a mutating function.
// That means the input is mutated and there's no need to return a value.

// Check types.
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK(x.dtype() == torch::kInt64, #x, " must be a kInt64 tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_LONG(x)


// Forward declaration.
void chacha20_cuda(torch::Tensor input, torch::Tensor dest, size_t step);

std::vector<torch::Tensor> chacha20(std::vector<torch::Tensor> inputs, size_t step) {
    // The input must be a contiguous long tensor of size 16 x N.
    // Also, the tensor must be contiguous to enable pointer arithmetic,
    // and must be stored in a cuda device.
    // Note that the input is a vector of inputs in different devices.
    
    std::vector<torch::Tensor> outputs;
    
    for (auto &input : inputs){
        CHECK_INPUT(input);
    
        // Prepare an output.
        auto dest = input.clone();
        
        // Run in cuda.
        chacha20_cuda(input, dest, step);
        
        // Store to the dest.
        outputs.push_back(dest);
    }
    
    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("chacha20", &chacha20, "CHACHA20 (CUDA)");
}
