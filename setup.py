import sys
import functools
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

this_dir = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "torch_lap_cuda"


@functools.lru_cache(maxsize=None)
def cuda_archs() -> str:
    return os.getenv("torch_lap_cuda_ARCHS", "75;80;89;90;100;120").split(";")


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f"linux_{platform.uname().machine}"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]


cmdclass = {}
ext_modules = []

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

check_if_cuda_home_none("torch_lap_cuda")
nvcc_flags = []

_, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
if bare_metal_version < Version("10.0"):
    raise RuntimeError(
        "torch_lap_cuda is only supported on CUDA 10.0 and above.  "
        "Note: make sure nvcc has a supported version by running nvcc -V."
    )


nvcc_flags.append("-gencode")
nvcc_flags.append(f"arch=compute_{75},code=sm_{75}")
for arch in cuda_archs():
    if bare_metal_version >= Version("11.1") and arch == "80":
        nvcc_flags.append("-gencode")
        nvcc_flags.append("arch=compute_90,code=sm_90")
    if bare_metal_version >= Version("11.8") and arch == "89":
        nvcc_flags.append("-gencode")
        nvcc_flags.append("arch=compute_89,code=sm_89")
    if bare_metal_version >= Version("12.0") and arch == "90":
        nvcc_flags.append("-gencode")
        nvcc_flags.append("arch=compute_90,code=sm_90")
    if bare_metal_version >= Version("12.8") and arch == "100":
        nvcc_flags.append("-gencode")
        nvcc_flags.append("arch=compute_100,code=sm_100")
    elif bare_metal_version >= Version("12.8") and arch == "120":
        nvcc_flags.append("-gencode")
        nvcc_flags.append("arch=compute_120,code=sm_120")

ext_modules.append(
    CUDAExtension(
        name="torch_lap_cuda_lib",
        sources=["csrc/lap_torch.cu"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                ]
                + nvcc_flags
            ),
        },
        extra_link_args=[
            "-fopenmp",
            "-lcuda",
        ],
        include_dirs=[
            Path(this_dir) / "csrc" / "include",
        ],
    )
)


def get_package_version():
    with open(Path(this_dir) / "torch_lap_cuda" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("torch_lap_cuda_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "tests",
            "dist",
            "docs",
            "torch_lap_cuda.egg-info",
        )
    ),
    author="Dmitrii Kobylianskii",
    author_email="kobad2@gmail.com",
    description="PyTorch wrapper for HyLAC CUDA library for solving linear assignment problems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    python_requires=">=3.9",
    install_requires=[
        "torch",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
    cmdclass={"build_ext": BuildExtension},
)
