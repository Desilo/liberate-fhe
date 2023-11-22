from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name="randint_cuda",
        sources=[
            "liberate/csprng/randint.cpp",
            "liberate/csprng/randint_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="randround_cuda",
        sources=[
            "liberate/csprng/randround.cpp",
            "liberate/csprng/randround_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="discrete_gaussian_cuda",
        sources=[
            "liberate/csprng/discrete_gaussian.cpp",
            "liberate/csprng/discrete_gaussian_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="chacha20_cuda",
        sources=[
            "liberate/csprng/chacha20.cpp",
            "liberate/csprng/chacha20_cuda_kernel.cu",
        ],
    ),
]

ext_modules_ntt = [
    CUDAExtension(
        name="ntt_cuda",
        sources=[
            "liberate/ntt/ntt.cpp",
            "liberate/ntt/ntt_cuda_kernel.cu",
        ],
    )
]

if __name__ == "__main__":
    setup(
        name="csprng",
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        script_args=["build_ext"],
        options={
            "build": {
                "build_lib": "liberate/csprng",
            }
        },
    )

    setup(
        name="ntt",
        ext_modules=ext_modules_ntt,
        script_args=["build_ext"],
        cmdclass={"build_ext": BuildExtension},
        options={
            "build": {
                "build_lib": "liberate/ntt",
            }
        },
    )
