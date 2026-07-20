from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import sysconfig, torch, os

BAD_FLAG = "-fdebug-default-version=4"


class PatchedBuildExtension(BuildExtension):
    def build_extensions(self):
        original_spawn = self.compiler.spawn

        def patched_spawn(cmd):
            cmd = [arg for arg in cmd if arg != BAD_FLAG]
            return original_spawn(cmd)

        self.compiler.spawn = patched_spawn

        for attr in ["compiler_so", "compiler", "compiler_cxx", "linker_so"]:
            if hasattr(self.compiler, attr):
                val = getattr(self.compiler, attr)
                if isinstance(val, list):
                    setattr(self.compiler, attr, [x for x in val if x != BAD_FLAG])

        super().build_extensions()


python_include = sysconfig.get_paths()["include"]
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
cuda_lib_dir  = "/usr/local/cuda/lib64"

rpath_flags = [
    f"-Wl,-rpath,{torch_lib_dir}",
    f"-Wl,-rpath,{cuda_lib_dir}",
]

setup(
    name="moce_cuda",
    ext_modules=[
        CUDAExtension(
            name="moce_cuda",
            sources=[
                "moce_layer_cuda.cpp",
                "moce_kernel.cu",
            ],
            include_dirs=[python_include],
            extra_link_args=rpath_flags,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-Xcompiler", "-fPIC",
                ],
            },
        )
    ],
    cmdclass={"build_ext": PatchedBuildExtension},
)