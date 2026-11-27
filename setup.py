import os
import platform
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Distribution
from setuptools.command.build_py import build_py


class BuildPython(build_py):
    def run(self):
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("The CUDA wheel must be built on Linux x86_64")

        root = Path(__file__).resolve().parent
        native = root / "comfy_aimdo" / "_native.abi3.so"
        if (root / "rust" / "Cargo.toml").is_file():
            env = {**os.environ, "AIMDO_BUILD_PYTHON": sys.executable}
            subprocess.run(["bash", "scripts/build-linux-aimdo.sh"], cwd=root, env=env, check=True)
        if not native.is_file():
            raise RuntimeError("The build did not produce comfy_aimdo/_native.abi3.so")
        super().run()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    cmdclass={"build_py": BuildPython},
    distclass=BinaryDistribution,
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
