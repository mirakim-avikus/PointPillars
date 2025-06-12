from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

class BuildExtension(_build_ext):
    def run(self):
        from torch.utils.cpp_extension import BuildExtension as TorchBuildExtension
        from torch.utils.cpp_extension import CUDAExtension

        self.distribution.ext_modules = [
            CUDAExtension(
                name='ops.voxel_op',
                sources=[
                    'ops/voxelization/voxelization.cpp',
                    'ops/voxelization/voxelization_cpu.cpp',
                    'ops/voxelization/voxelization_cuda.cu',
                ],
                define_macros=[('WITH_CUDA', None)],
            ),
            CUDAExtension(
                name='ops.iou3d_op',
                sources=[
                    'ops/iou3d/iou3d.cpp',
                    'ops/iou3d/iou3d_kernel.cu',
                ],
                define_macros=[('WITH_CUDA', None)],
            ),
        ]

        build_ext = TorchBuildExtension(self.distribution)
        build_ext.run()

setup(
    name='pointpillars',
    packages=find_packages(where='pointpillars'),  # 반드시 이 부분 추가
    package_dir={'':'pointpillars'},
    ext_modules=[],
    cmdclass={'build_ext': BuildExtension},
)
