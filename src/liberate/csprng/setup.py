from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name="randint_cuda",
        sources=["randint.cpp", "randint_cuda_kernel.cu"],
    ),
    CUDAExtension(
        name="randround_cuda",
        sources=["randround.cpp", "randround_cuda_kernel.cu"],
    ),
    CUDAExtension(
        name="discrete_gaussian_cuda",
        sources=["discrete_gaussian.cpp", "discrete_gaussian_cuda_kernel.cu"],
    ),
    CUDAExtension(
        name="chacha20_cuda",
        sources=["chacha20.cpp", "chacha20_cuda_kernel.cu"],
    ),
]

setup(
    name="csprng",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    script_args=["build_ext"],
    options={
        "build_ext": {
            "inplace": True,
        }
    },
)
