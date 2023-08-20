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
                "../source/common.cpp",
                "../source/DeviceMem.cpp",
                "../source/SwiftNetMLP.cpp",
            ],
            include_dirs=[
                "../include",
                "../include/Network",
                "../include/Losses",
                "../include/Optimizers",
            ],
        )
    ],
    cmdclass={"build_ext": CustomDpcppBuildExtension},
    # cmdclass={"build_ext": DpcppBuildExtension},
)
