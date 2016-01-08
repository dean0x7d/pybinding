# pybinding

[![Documentation Status](https://readthedocs.org/projects/pybinding/badge/?version=latest)](http://pybinding.readthedocs.org/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/dean0x7d/pybinding.svg?branch=develop)](https://travis-ci.org/dean0x7d/pybinding)
[![Build status](https://ci.appveyor.com/api/projects/status/vd84e6gxixsu9l81/branch/develop?svg=true)](https://ci.appveyor.com/project/dean0x7d/pybinding/branch/develop)

Pybinding is a Python package for numerical tight-binding calculations in solid state physics.
The main features include:

* **Declarative model construction** - The user just needs to describe *what* the model should be,
but not *how* to build it. Pybinding will take care of the numerical details of building the
Hamiltonian matrix so users can concentrate on the physics: the quantum properties of the model.

* **Fast compute** - Pybinding's implementation of the kernel polynomial method allows for very fast
calculation of the Green's function of the Hamiltonian. Exact diagonalization is also available
through the use of scipy's eigensolvers. The framework is very flexible and allows the addition 
of user-defined computation routines.

* **Result analysis and visualization** - The package contains utility functions for post-processing
the raw result data. The included plotting functions are tailored for tight-binding problems to
help visualize the model structure and make sense of the results.

The main interface is written in Python with the aim to be as user-friendly and flexible as
possible. Under the hood, C++11 is used to accelerate demanding tasks to deliver high performance
with low memory usage.


### Current status: Alpha

* Alpha: Assume everything is broken.
* WIP: Preparing the [documentation] in anticipation of the first beta release.


### Install

Pybinding can be installed on Windows, Linux or Mac, with the following prerequisites:

* [Python] 3.4 or newer (Python 2.x is not supported)
* The [SciPy] stack of scientific packages, with required versions:
  * numpy >= v1.9
  * scipy >= v0.15
  * matplotlib >= v1.5
* If you're using Linux, you'll also need GCC >= v4.8 and CMake >= v3.0.

Detailed [install instructions] are part of the documentation, but if you already have all the
prerequisites, it's just a simple case of using `pip`, Python's usual package manager:

    pip install pybinding


[Python]: https://www.python.org/
[SciPy]: http://www.scipy.org/>
[install instructions]: http://pybinding.rtfd.org/en/latest/install/index.html
[documentation]: http://pybinding.readthedocs.org/
