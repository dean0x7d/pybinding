import re
import os
import sys
import shutil
import platform
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.egg_info import manifest_maker


if sys.version_info[:2] < (3, 4):
    print("Python >= 3.4 is required.")
    sys.exit(-1)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = sourcedir


class CMakeBuild(build_ext):
    def run(self):
        if shutil.which('cmake') is None:
            print("CMake 3.1 or newer is required to build pybinding")
            sys.exit(-1)

        for ext in self.extensions:
            build_dir = os.path.join(os.path.dirname(__file__), 'build', 'cmake')
            os.makedirs(build_dir, exist_ok=True)
            cmake_dir = os.path.abspath(ext.sourcedir)

            extpath = self.get_ext_fullpath(ext.name)
            extfulldir = os.path.abspath(os.path.dirname(extpath))
            cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extfulldir,
                          '-DPYTHON_EXECUTABLE=' + sys.executable,
                          '-DPB_WERROR=' + os.environ.get("PB_WERROR", "OFF"),
                          '-DPB_TESTS=' + os.environ.get("PB_TESTS", "OFF"),
                          '-DPB_NATIVE_SIMD=' + os.environ.get("PB_NATIVE_SIMD", "ON"),
                          '-DPB_MKL=' + os.environ.get("PB_MKL", "OFF"),
                          '-DPB_CUDA=' + os.environ.get("PB_CUDA", "OFF")]
            build_args = ['--config', 'Release']

            if platform.system() == "Windows":
                cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=' + extfulldir]
                if sys.maxsize > 2**32:
                    cmake_args += ['-A', 'x64']
                build_args += ['--', '/m']
            else:
                build_args += ['--', '-j2']

            subprocess.check_call(['cmake', cmake_dir] + cmake_args, cwd=build_dir)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_dir)


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

    with open("changelog.md") as file:
        log = file.read()
    match = re.search(r"### ([\s\S]*?)\n###\s", log)
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
    ],

    packages=find_packages(exclude=['cppcore', 'cppwrapper', 'test*']) + ['pybinding.tests'],
    package_dir={'pybinding.tests': 'tests'},
    include_package_data=True,
    ext_modules=[CMakeExtension('_pybinding')],
    install_requires=['numpy>=1.9.0', 'scipy>=0.15', 'matplotlib>=1.5.0',
                      'py-cpuinfo>=0.2.3', 'pytest>=2.8'],
    zip_safe=False,
    cmdclass=dict(build_ext=CMakeBuild)
)
