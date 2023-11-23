from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name="randint_cuda",
        sources=[
            "src/liberate/csprng/randint.cpp",
            "src/liberate/csprng/randint_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="randround_cuda",
        sources=[
            "src/liberate/csprng/randround.cpp",
            "src/liberate/csprng/randround_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="discrete_gaussian_cuda",
        sources=[
            "src/liberate/csprng/discrete_gaussian.cpp",
            "src/liberate/csprng/discrete_gaussian_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="chacha20_cuda",
        sources=[
            "src/liberate/csprng/chacha20.cpp",
            "src/liberate/csprng/chacha20_cuda_kernel.cu",
        ],
    ),
]

ext_modules_ntt = [
    CUDAExtension(
        name="ntt_cuda",
        sources=[
            "src/liberate/ntt/ntt.cpp",
            "src/liberate/ntt/ntt_cuda_kernel.cu",
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
                "build_lib": "src/liberate/csprng",
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
                "build_lib": "src/liberate/ntt",
            }
        },
    )

