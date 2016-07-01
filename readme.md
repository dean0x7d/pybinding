# pybinding

[![Build Status](https://travis-ci.org/dean0x7d/pybinding.svg?branch=master)](https://travis-ci.org/dean0x7d/pybinding)
[![Build status](https://ci.appveyor.com/api/projects/status/vd84e6gxixsu9l81/branch/master?svg=true)](https://ci.appveyor.com/project/dean0x7d/pybinding)
[![Documentation Status](https://readthedocs.org/projects/pybinding/badge/?version=latest)](http://docs.pybinding.site/)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/dean0x7d/pybinding?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Pybinding is a Python package for numerical tight-binding calculations in solid state physics.
The main features include:

* **Declarative model construction** - The user just needs to describe *what* the model should be,
  but not *how* to build it. Pybinding will take care of the numerical details of building the
  Hamiltonian matrix so users can concentrate on the physics, i.e. the quantum properties of the
  model.

* **Fast compute** - Pybinding's implementation of the kernel polynomial method allows for very
  fast calculation of the Green's function of the Hamiltonian. Exact diagonalization is also
  available through the use of scipy's eigensolvers. The framework is very flexible and allows
  the addition of user-defined computation routines.

* **Result analysis and visualization** - The package contains utility functions for post-processing
  the raw result data. The included plotting functions are tailored for tight-binding problems to
  help visualize the model structure and to make sense of the results.

The code interface is written in Python with the aim to be as user-friendly and flexible as
possible. Under the hood, C++11 is used to accelerate demanding tasks to deliver high performance
with low memory usage.

See the [documentation] for more details.

## Install

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


## Current status

Work in progress. The goal of the project is to develop a tight-binding code framework which is
fast, flexible and easy to use.
See the [documentation] for more details and a guide to get started.

#### Implemented features

* Construction of arbitrary tight-binding lattices and geometries: 1 to 3 dimensions
  (including multilayer 2D systems), periodic or finite size (with fine control of edges)
* Easy polygon shape definition for 2D systems and freeform shapes for n-dimensional systems
* Geometric deformations and defects: defined via displacement and state functions
* Fields and arbitrary effects: defined via hopping and onsite energy functions
* Green's function: fast kernel polynomial method implementation
* Eigensolvers: standard dense and sparse solvers (always available)
  and the [FEAST] solver (only available when compiled with Intel's MKL)
* Transport: scattering systems with semi-infinite leads can be constructed in Pybinding and then
  solved using the [Kwant compatibility] layer.
* Model and result objects have builtin plotting functions for easy visualization

#### Planned work

* Improvements for 3D systems (mostly related to plotting functions)
* Multiple orbitals and spins are already supported, but could use a nicer interface


## Benchmarks

One of the main features of Pybinding is an easy-to-use and fast model builder. This can be a
demanding task for large or complicated systems. Great care was taken to make this process fast.

The following figures compare the performance of Pybinding with the [Kwant] package. They present
the time and memory required to build a Hamiltonian matrix which describes a tight-binding system.
Pybinding features good performance and a low memory footprint by using contiguous data structures
and vectorized operations.

<p align="center">
  <img src="/docs/benchmarks/system_build.png?raw=true" alt="Tight-binding model build benchmark"/>
</p>

See the [benchmarks] section of the documentation for details on the testbed hardware and software,
as well as the source code which can be used to reproduce the results.


## Questions?

If you have any questions, feel free to join the [chat room on Gitter].
You can also open an issue at the [tracker].


[documentation]: http://docs.pybinding.site/
[install instructions]: http://docs.pybinding.site/page/install/index.html
[Python]: https://www.python.org/
[SciPy]: http://www.scipy.org/
[FEAST]: http://www.ecs.umass.edu/~polizzi/feast/index.htm
[Kwant compatibility]: http://docs.pybinding.site/page/advanced/kwant.html
[Kwant]: http://kwant-project.org/
[benchmarks]: http://docs.pybinding.site/page/benchmarks/index.html
[chat room on Gitter]: https://gitter.im/dean0x7d/pybinding
[tracker]: https://github.com/dean0x7d/pybinding/issues
