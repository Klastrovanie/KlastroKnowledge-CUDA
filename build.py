# build.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_cuda_args = [
    '-O3',
    '--use_fast_math',
    '-std=c++17',
    '--expt-relaxed-constexpr',
    '-gencode=arch=compute_60,code=sm_60',
    '-gencode=arch=compute_70,code=sm_70',
    '-gencode=arch=compute_75,code=sm_75',
    '-gencode=arch=compute_80,code=sm_80',
    '-gencode=arch=compute_86,code=sm_86',
    '-gencode=arch=compute_89,code=sm_89',
    '-gencode=arch=compute_90,code=sm_90',   # RTX 4090 (Ada)
    '-gencode=arch=compute_100,code=sm_100', # RTX 5090 (Blackwell)
]

setup(
    name='klastroknowledge',
    ext_modules=[
        CUDAExtension(
            name='klastroknowledge',
            sources=['klastroknowledge.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': extra_cuda_args,
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
