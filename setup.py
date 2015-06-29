from setuptools import setup, find_packages
from distutils import sysconfig, spawn, dir_util
from distutils.command.build_py import build_py
from setuptools.command.develop import develop
import sys
import os
import platform

if sys.version_info.major < 3:
    print("Sorry, Python 3 is required")
    sys.exit(-1)


def inject_cmake(base_class):
    def cmake_build():
        if not spawn.find_executable('cmake'):
            print("CMake 3.0 or newer is required to build pybinding")
            sys.exit(-1)

        build_dir = os.path.join(os.path.split(__file__)[0], 'build')
        dir_util.mkpath(build_dir)
        cwd = os.getcwd()
        os.chdir(build_dir)

        try:
            spawn.spawn(['cmake', '..'])
            spawn.spawn(['make', '-j4'])
        except spawn.DistutilsExecError:
            sys.exit(-1)

        os.chdir(cwd)

    class ClassWithCMake(base_class):
        def run(self):
            if platform.system() != "Windows":
                cmake_build()
            super().run()

    return ClassWithCMake


setup(
    name='pybinding',
    version='0.6.0',
    description='Python tight-binding package',
    long_description='',
    url='https://github.com/dean0x7d/pybinding',
    license='BSD',
    keywords='pybinding tigh-binding tighbinding cmt',

    author='Dean Moldovan',
    author_email='dean.moldovan@uantwerpen.be',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: Implementation :: CPython',

    ],

    packages=find_packages(exclude=['boost_python', 'tests*']),
    package_data=dict(pybinding=['../_pybinding' + sysconfig.get_config_var('EXT_SUFFIX')]),
    install_requires=['numpy>=1.9.0', 'scipy>=0.15', 'matplotlib>=1.4.3',
                      'py-cpuinfo>=0.1.4', 'progressbar2>=2.7.3', 'pytest>=2.7'],
    zip_safe=False,
    cmdclass=dict(build_py=inject_cmake(build_py), develop=inject_cmake(develop))
)
