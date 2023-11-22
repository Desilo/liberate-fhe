from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name="ntt_cuda",
        sources=[
            "ntt.cpp",
            "ntt_cuda_kernel.cu",
        ],
    )
]
if __name__ == "__main__":
    setup(
        name="ntt",
        ext_modules=ext_modules,
        script_args=["build_ext"],
        cmdclass={"build_ext": BuildExtension},
        #  options={
        #  "build":{
        #  "build_lib":"liberate/ntt",
        #  }
        #  }
    )
