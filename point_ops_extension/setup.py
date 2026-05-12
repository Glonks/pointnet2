from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='point_ops',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='point_ops._C',
            sources=[
                'src/module.cpp',
                'src/register_ops.cpp',
                # farthest point sampling
                'src/farthest_point_sampling/farthest_point_sampling_cpu.cpp',
                'src/farthest_point_sampling/farthest_point_sampling_cuda.cu',
                # ball query
                # 'src/ball_query/ball_query.cpp',
                # 'src/ball_query/ball_query.cu',
            ],
            extra_compile_args={
                'cxx': [
                    '-std=c++17',
                    '-O3'
                ],
                'nvcc': [
                    '-std=c++17',
                    '-O3'
                ]
            },
        ),
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    options={'bdist_wheel': {'py_limited_api': 'cp39'}}  # Minimum python version: 3.9 (Why? idk)
)
