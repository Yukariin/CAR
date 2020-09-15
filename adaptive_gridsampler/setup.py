from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='adaptive_gridsampler_cuda',
    ext_modules=[
        CUDAExtension('adaptive_gridsampler_cuda', [
            'adaptive_gridsampler_cuda.cpp',
            'adaptive_gridsampler_kernel.cu'
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
