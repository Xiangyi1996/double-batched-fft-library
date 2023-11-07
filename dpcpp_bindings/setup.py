from setuptools import setup

import torch
import intel_extension_for_pytorch
from torch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension


class CustomDpcppBuildExtension(DpcppBuildExtension):
    def build_extensions(self):
        # Enable compiler warnings and notes
        for extension in self.extensions:
            self._add_compiler_flags(extension)

        super().build_extensions()

    def _add_compiler_flags(self, extension):
        # flags_to_add = ["-Wall", "-Wextra", "-Wconversion", "-Wformat"]
        # flags_to_add = ["-Wall", "-Wextra", "-Wformat"]
        flags_to_add = ["-w", "-Werror"]

        # Check if extra_compile_args is a dictionary with 'cxx' key
        if (
            isinstance(extension.extra_compile_args, dict)
            and "cxx" in extension.extra_compile_args
        ):
            extension.extra_compile_args["cxx"] += flags_to_add
        else:
            if isinstance(extension.extra_compile_args, dict):
                # Initialize 'cxx' key if it doesn't exist
                extension.extra_compile_args["cxx"] = []
            else:
                extension.extra_compile_args = []
            extension.extra_compile_args.append({"cxx": flags_to_add})


setup(
    name="tiny_nn",
    ext_modules=[
        DPCPPExtension(
            "tiny_nn",
            [
                "pybind_module.cpp",
                "../source/tnn_api.cpp",
                "../source/common/common.cpp",
                "../source/common/common_host.cpp",
                "../source/common/DeviceMem.cpp",
                "../source/SwiftNetMLP.cpp",
                "../source/network_with_encodings.cpp",
            ],
            include_dirs=[
                "/nfs/site/home/yuankai/code/tiny-dpcpp-nn/include",
                "/nfs/site/home/yuankai/code/tiny-dpcpp-nn/include/common",
                "/nfs/site/home/yuankai/code/tiny-dpcpp-nn/include/Network",
                "/nfs/site/home/yuankai/code/tiny-dpcpp-nn/include/Losses",
                "/nfs/site/home/yuankai/code/tiny-dpcpp-nn/include/Optimizers",
            ],
        )
    ],
    cmdclass={"build_ext": CustomDpcppBuildExtension},
)
