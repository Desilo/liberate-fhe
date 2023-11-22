#include <torch/extension.h>
#include <vector>

//------------------------------------------------------------------
// Main functions
//------------------------------------------------------------------

torch::Tensor mont_mult_cuda(const torch::Tensor a,
                              const torch::Tensor b,
                              const torch::Tensor ql,
                              const torch::Tensor qh,
                              const torch::Tensor kl,
                              const torch::Tensor kh);

void mont_enter_cuda(torch::Tensor a,
                      const torch::Tensor Rs,
                      const torch::Tensor ql,
                      const torch::Tensor qh,
                      const torch::Tensor kl,
                      const torch::Tensor kh);

void ntt_cuda(torch::Tensor a,
              const torch::Tensor even,
              const torch::Tensor odd,
              const torch::Tensor psi,
              const torch::Tensor _2q,
              const torch::Tensor ql,
              const torch::Tensor qh,
              const torch::Tensor kl,
              const torch::Tensor kh);

void enter_ntt_cuda(torch::Tensor a,
              const torch::Tensor Rs,
              const torch::Tensor even,
              const torch::Tensor odd,
              const torch::Tensor psi,
              const torch::Tensor _2q,
              const torch::Tensor ql,
              const torch::Tensor qh,
              const torch::Tensor kl,
              const torch::Tensor kh);

void intt_cuda(torch::Tensor a,
              const torch::Tensor even,
              const torch::Tensor odd,
              const torch::Tensor psi,
              const torch::Tensor Ninv,
              const torch::Tensor _2q,
              const torch::Tensor ql,
              const torch::Tensor qh,
              const torch::Tensor kl,
              const torch::Tensor kh);


void mont_redc_cuda(torch::Tensor a,
                    const torch::Tensor ql,
                    const torch::Tensor qh,
                    const torch::Tensor kl,
                    const torch::Tensor kh);


void intt_exit_cuda(torch::Tensor a,
              const torch::Tensor even,
              const torch::Tensor odd,
              const torch::Tensor psi,
              const torch::Tensor Ninv,
              const torch::Tensor _2q,
              const torch::Tensor ql,
              const torch::Tensor qh,
              const torch::Tensor kl,
              const torch::Tensor kh);

void intt_exit_reduce_cuda(torch::Tensor a,
              const torch::Tensor even,
              const torch::Tensor odd,
              const torch::Tensor psi,
              const torch::Tensor Ninv,
              const torch::Tensor _2q,
              const torch::Tensor ql,
              const torch::Tensor qh,
              const torch::Tensor kl,
              const torch::Tensor kh);

void intt_exit_reduce_signed_cuda(torch::Tensor a,
              const torch::Tensor even,
              const torch::Tensor odd,
              const torch::Tensor psi,
              const torch::Tensor Ninv,
              const torch::Tensor _2q,
              const torch::Tensor ql,
              const torch::Tensor qh,
              const torch::Tensor kl,
              const torch::Tensor kh);

void reduce_2q_cuda(torch::Tensor a,
              const torch::Tensor _2q);

void make_signed_cuda(torch::Tensor a,
              const torch::Tensor _2q);

void make_unsigned_cuda(torch::Tensor a,
              const torch::Tensor _2q);

torch::Tensor mont_add_cuda(const torch::Tensor a,
                            const torch::Tensor b,
                            const torch::Tensor _2q);

torch::Tensor mont_sub_cuda(const torch::Tensor a,
                            const torch::Tensor b,
                            const torch::Tensor _2q);

torch::Tensor tile_unsigned_cuda(torch::Tensor a,
              const torch::Tensor _2q);


//------------------------------------------------------------------
// Main functions
//------------------------------------------------------------------

std::vector<torch::Tensor> mont_mult(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> b, 
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh) {
    
    std::vector<torch::Tensor> outputs;
    
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        auto c = mont_mult_cuda(a[i],
                                b[i], 
                                ql[i],
                                qh[i],
                                kl[i],
                                kh[i]);
        
        outputs.push_back(c);
        
    }
    
    return outputs;
}

void mont_enter(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> Rs, 
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh) {
        
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        mont_enter_cuda(a[i],
                        Rs[i], 
                        ql[i],
                        qh[i],
                        kl[i],
                        kh[i]);        
    }
}


void ntt(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> even,
    const std::vector<torch::Tensor> odd, 
    const std::vector<torch::Tensor> psi,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh) {
        
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        ntt_cuda(a[i],
                even[i],
                odd[i], 
                psi[i],
                _2q[i],
                ql[i],
                qh[i],
                kl[i],
                kh[i]);        
    }
}

void enter_ntt(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> Rs,
    const std::vector<torch::Tensor> even,
    const std::vector<torch::Tensor> odd, 
    const std::vector<torch::Tensor> psi,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh) {
        
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        enter_ntt_cuda(a[i],
                Rs[i],
                even[i],
                odd[i], 
                psi[i],
                _2q[i],
                ql[i],
                qh[i],
                kl[i],
                kh[i]);        
    }
}


void intt(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> even,
    const std::vector<torch::Tensor> odd, 
    const std::vector<torch::Tensor> psi,
    const std::vector<torch::Tensor> Ninv,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh) {
        
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        intt_cuda(a[i],
                even[i],
                odd[i], 
                psi[i],
                Ninv[i],
                _2q[i],
                ql[i],
                qh[i],
                kl[i],
                kh[i]);        
    }
}


void mont_redc(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh) {
        
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        mont_redc_cuda(a[i],
                        ql[i],
                        qh[i],
                        kl[i],
                        kh[i]);        
    }
}



void intt_exit(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> even,
    const std::vector<torch::Tensor> odd, 
    const std::vector<torch::Tensor> psi,
    const std::vector<torch::Tensor> Ninv,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh) {
        
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        intt_exit_cuda(a[i],
                even[i],
                odd[i], 
                psi[i],
                Ninv[i],
                _2q[i],
                ql[i],
                qh[i],
                kl[i],
                kh[i]);        
    }
}

void intt_exit_reduce(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> even,
    const std::vector<torch::Tensor> odd, 
    const std::vector<torch::Tensor> psi,
    const std::vector<torch::Tensor> Ninv,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh) {
        
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        intt_exit_reduce_cuda(a[i],
                even[i],
                odd[i], 
                psi[i],
                Ninv[i],
                _2q[i],
                ql[i],
                qh[i],
                kl[i],
                kh[i]);        
    }
}

void intt_exit_reduce_signed(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> even,
    const std::vector<torch::Tensor> odd, 
    const std::vector<torch::Tensor> psi,
    const std::vector<torch::Tensor> Ninv,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh) {
        
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        intt_exit_reduce_signed_cuda(a[i],
                even[i],
                odd[i], 
                psi[i],
                Ninv[i],
                _2q[i],
                ql[i],
                qh[i],
                kl[i],
                kh[i]);        
    }
}

void reduce_2q(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> _2q) {
        
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        reduce_2q_cuda(a[i], _2q[i]);        
    }
}

void make_signed(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> _2q) {
        
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        make_signed_cuda(a[i], _2q[i]);        
    }
}

void make_unsigned(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> _2q) {
        
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        make_unsigned_cuda(a[i], _2q[i]);        
    }
}

std::vector<torch::Tensor> mont_add(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> b, 
    const std::vector<torch::Tensor> _2q) {
    
    std::vector<torch::Tensor> outputs;
    
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        auto c = mont_add_cuda(a[i], b[i], _2q[i]);
        outputs.push_back(c);   
    }
    return outputs;
}

std::vector<torch::Tensor> mont_sub(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> b, 
    const std::vector<torch::Tensor> _2q) {
    
    std::vector<torch::Tensor> outputs;
    
    const auto num_devices = a.size();    
    for (int i=0; i<num_devices; ++i){
        auto c = mont_sub_cuda(a[i], b[i], _2q[i]);
        outputs.push_back(c);   
    }
    return outputs;
}

std::vector<torch::Tensor> tile_unsigned(
    std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> _2q) {
    
    std::vector<torch::Tensor> outputs;
    
    const auto num_devices = _2q.size();    
    for (int i=0; i<num_devices; ++i){
        auto result = tile_unsigned_cuda(a[i], _2q[i]);
        outputs.push_back(result);
    }
    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mont_mult", &mont_mult, "MONTGOMERY MULTIPLICATION");
    m.def("mont_enter", &mont_enter, "ENTER MONTGOMERY");
    m.def("ntt", &ntt, "FORWARD NTT");
    m.def("enter_ntt", &enter_ntt, "ENTER -> FORWARD NTT");
    m.def("intt", &intt, "INVERSE NTT");
    m.def("mont_redc", &mont_redc, "MONTGOMERY REDUCTION");
    m.def("intt_exit", &intt_exit, "INVERSE NTT -> EXIT");
    m.def("intt_exit_reduce", &intt_exit_reduce, "INVERSE NTT -> EXIT -> REDUCE");
    m.def("intt_exit_reduce_signed", &intt_exit_reduce_signed, "INVERSE NTT -> EXIT -> REDUCE -> MAKE SIGNED");
    m.def("reduce_2q", &reduce_2q, "REDUCE RANGE TO 2q");
    m.def("make_signed", &make_signed, "MAKE SIGNED");
    m.def("make_unsigned", &make_unsigned, "MAKE UNSIGNED");
    m.def("mont_add", &mont_add, "MONTGOMERY ADDITION");
    m.def("mont_sub", &mont_sub, "MONTGOMERY SUBTRACTION");
    m.def("tile_unsigned", &tile_unsigned, "TILE -> MAKE UNSIGNED");
}
