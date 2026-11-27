import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Distribution
from setuptools.command.build_py import build_py


class BuildPython(build_py):
    def run(self):
        root = Path(__file__).resolve().parent
        if not (root / "comfy_aimdo" / "_native.abi3.so").is_file():
            env = {**os.environ, "AIMDO_BUILD_PYTHON": sys.executable}
            subprocess.run(["bash", "scripts/build-linux-aimdo.sh"], cwd=root, env=env, check=True)
        super().run()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    cmdclass={"build_py": BuildPython},
    distclass=BinaryDistribution,
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
