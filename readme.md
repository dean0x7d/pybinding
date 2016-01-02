# pybinding

[![Documentation Status](https://readthedocs.org/projects/pybinding/badge/?version=latest)](http://pybinding.readthedocs.org/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/dean0x7d/pybinding.svg?branch=develop)](https://travis-ci.org/dean0x7d/pybinding)

Pybinding is a Python package for numerical tight-binding calculations in solid state physics.
The main features include:

* Declarative model construction - The user just needs to describe *what* the model should be,
but not *how* to build it. Pybinding will take care of the numerical details of building the
Hamiltonian matrix so users can concentrate on the physics: the quantum properties of the model.

* Fast compute - Pybinding's implementation of the kernel polynomial method allows for very fast
calculation of the Green's function of the Hamiltonian. Exact diagonalization is also available
through the use of scipy's eigensolvers. The framework is very flexible and allows the addition 
of user-defined computation routines.

* Result analysis and visualization - The package contains utility functions for post-processing
the raw result data. The included plotting functions are tailored for tight-binding problems to
help visualize the model structure and make sense of the results.

The main interface is written in Python with the aim to be as user-friendly and flexible as
possible. Under the hood, C++11 is used to accelerate demanding tasks to deliver high performance
with low memory usage.

### Current status: Alpha

Preparing the [documentation](http://pybinding.readthedocs.org/) in anticipation of the first
beta release.
