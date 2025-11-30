# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# compile for all possible CUDA architectures
# all_cuda_archs = cuda.get_gencode_flags().replace("compute=", "arch=").split()
# alternatively, you can list cuda archs that you want, eg:
all_cuda_archs = ["-gencode", "arch=compute_89,code=sm_89"]

all_cuda_archs = [
    "-gencode",
    "arch=compute_61,code=sm_61",  # GeForce GTX 1080 Ti
    "-gencode",
    "arch=compute_75,code=sm_75",  # GeForce RTX 2080 Ti
    "-gencode",
    "arch=compute_89,code=sm_89",  # RTX 6000 (Spectre, Lovelace), RTX 4060
]

setup(
    name="curope",
    ext_modules=[
        CUDAExtension(
            name="curope",
            sources=[
                "curope.cpp",
                "kernels.cu",
            ],
            extra_compile_args=dict(
                nvcc=["-O3", "--ptxas-options=-v", "--use_fast_math"] + all_cuda_archs,
                cxx=["-O3"],
            ),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
