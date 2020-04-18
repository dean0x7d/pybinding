import re
import os
import sys
import shutil
import platform

from subprocess import check_call, check_output, CalledProcessError
from distutils.version import LooseVersion

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.egg_info import manifest_maker


if sys.version_info[:2] < (3, 6):
    print("Python >= 3.6 is required.")
    sys.exit(-1)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake not found. Version 3.1 or newer is required")

        cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < "3.1.0":
            raise RuntimeError("CMake 3.1 or newer is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
                      "-DPYTHON_EXECUTABLE=" + sys.executable]
        cmake_args += ["-DPB_WERROR=" + os.environ.get("PB_WERROR", "OFF"),
                       "-DPB_TESTS=" + os.environ.get("PB_TESTS", "OFF"),
                       "-DPB_NATIVE_SIMD=" + os.environ.get("PB_NATIVE_SIMD", "ON"),
                       "-DPB_MKL=" + os.environ.get("PB_MKL", "OFF"),
                       "-DPB_CUDA=" + os.environ.get("PB_CUDA", "OFF")]

        cfg = os.environ.get("PB_BUILD_TYPE", "Release")
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/v:m", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            if "-j" not in os.environ.get("MAKEFLAGS", ""):
                parallel_jobs = 2 if not os.environ.get("READTHEDOCS") else 1
                build_args += ["--", "-j{}".format(parallel_jobs)]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DCPB_VERSION=\\"{}\\"'.format(env.get("CXXFLAGS", ""),
                                                             self.distribution.get_version())

        def build():
            os.makedirs(self.build_temp, exist_ok=True)
            check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        try:
            build()
        except CalledProcessError:  # possible CMake error if the build cache has been copied
            shutil.rmtree(self.build_temp)  # delete build cache and try again
            build()


def about(package):
    ret = {}
    filename = os.path.join(os.path.dirname(__file__), package, "__about__.py")
    with open(filename, 'rb') as file:
        exec(compile(file.read(), filename, 'exec'), ret)
    return ret


def changelog():
    """Return the changes for the latest version only"""
    if not os.path.exists("changelog.md"):
        return ""

    with open("changelog.md", encoding="utf-8") as file:
        log = file.read()
    match = re.search(r"## ([\s\S]*?)\n##\s", log)
    return match.group(1) if match else ""


info = about("pybinding")
manifest_maker.template = "setup.manifest"
setup(
    name=info['__title__'],
    version=info['__version__'],
    description=info['__summary__'],
    long_description="Documentation: http://pybinding.site/\n\n" + changelog(),
    url=info['__url__'],
    license=info['__license__'],
    keywords="pybinding tight-binding solid-state physics cmt",

    author=info['__author__'],
    author_email=info['__email__'],

    platforms=['Unix', 'Windows'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
    ],

    packages=find_packages(exclude=['cppcore', 'cppwrapper', 'test*']) + ['pybinding.tests'],
    package_dir={'pybinding.tests': 'tests'},
    include_package_data=True,
    ext_modules=[CMakeExtension('_pybinding')],
    install_requires=['numpy>=1.12', 'scipy>=0.19', 'matplotlib>=2.0', 'pytest>=5.0'],
    zip_safe=False,
    cmdclass=dict(build_ext=CMakeBuild)
)
