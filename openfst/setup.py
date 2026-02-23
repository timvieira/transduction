"""Build script for openfst_decomp Cython extension.

Two-phase build:
  1. CMake builds vendored OpenFST + our C++ lib as static archives.
  2. Cython compiles the .pyx and links against those static archives.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension
from Cython.Build import cythonize

HERE = Path(__file__).parent.resolve()
BUILD_DIR = HERE / "build" / "cmake"
OPENFST_INCLUDE = HERE / "third_party" / "openfst" / "src" / "include"
SRC_DIR = HERE / "src"


def cmake_build():
    """Run CMake configure + build if the static libs aren't already built."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    fst_static = BUILD_DIR / "libfst_static.a"
    decomp_static = BUILD_DIR / "libopenfst_decomp.a"

    if fst_static.exists() and decomp_static.exists():
        return  # Already built

    subprocess.check_call(
        ["cmake", str(HERE), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=str(BUILD_DIR),
    )
    subprocess.check_call(
        ["cmake", "--build", ".", "-j", str(os.cpu_count() or 4)],
        cwd=str(BUILD_DIR),
    )


cmake_build()

ext = Extension(
    "_openfst_decomp",
    sources=["cython/_openfst_decomp.pyx"],
    language="c++",
    include_dirs=[
        str(SRC_DIR),
        str(OPENFST_INCLUDE),
    ],
    library_dirs=[str(BUILD_DIR)],
    # Link order matters: our lib first, then OpenFST
    extra_objects=[
        str(BUILD_DIR / "libopenfst_decomp.a"),
        str(BUILD_DIR / "libfst_static.a"),
    ],
    extra_compile_args=["-std=c++17", "-O2"],
    extra_link_args=["-lpthread", "-ldl"],
)

setup(
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
)
