About
=====

.. meta::
   :description: Pybinding is a Python package for numerical tight-binding calculations.
                 It features easy model construction, fast compute and result visualization tools.

Pybinding is a Python package for numerical tight-binding calculations in solid state physics.
The main features include:

* **Declarative model construction** - The user just needs to describe *what* the model should be,
  but not *how* to build it. Pybinding will take care of the numerical details of building the
  Hamiltonian matrix so users can concentrate on the physics, i.e. the quantum properties of the
  model.

* **Fast compute** - Pybinding's implementation of the kernel polynomial method allows for very
  fast calculation of various physical properties of tight-binding systems. Exact diagonalization
  is also available through the use of scipy's eigenvalue solvers. The framework is very flexible
  and allows the addition of user-defined computation routines.

* **Result analysis and visualization** - The package contains utility functions for post-processing
  the raw result data. The included plotting functions are tailored for tight-binding problems to
  help visualize the model structure and to make sense of the results.

The main interface is written in Python with the aim to be as user-friendly and flexible as
possible. Under the hood, C++11 is used to accelerate demanding tasks to deliver high performance
with low memory usage.


Background
----------

The tight-binding model is an approximate approach of calculating the electronic band structure
of solids using a basis of localized atomic orbitals. This model is applicable to a wide variety
of systems and phenomena in quantum physics. The approach does not require computing from first
principals, but instead simply uses parameterized matrix elements. In contrast to *ab initio*
calculations, the tight-binding model can scale to large system sizes on the order of millions
of atoms.

Python is a programming language which is easy to learn and a joy to use. It has deep roots in
the scientific community as evidenced by the rich scientific Python library collection: `SciPy
<http://www.scipy.org/>`_. As such, Python is the ideal choice as the main interface for pybinding.
In the core of the package, C++11 is used to accelerate model construction and the most demanding
calculations. This is done silently in the background.


Workflow
--------

The general workflow starts with model definition. Three main parts are required to describe a
tight-binding model:

* **The crystal lattice** - This step includes the specification of the primitive lattice vectors
  and the configuration of the unit cell (atoms, orbitals and spins). This can be user-defined,
  but the package also contains a repository of the pre-made specifications for several materials.

* **System geometry** - The model system can be infinite through the use of translational symmetry
  or it can be finite by specifying a shape. The two approaches can also be composed to create
  periodic systems with intricate structural patterns. The structure can be controlled up to fine
  details, e.g. to form specific edge types as well as various defects.

* **Fields** - Functions can be applied to the onsite and hopping energies of the model system
  to simulate external fields or various effects. These functions are be defined independently
  of any lattice or specific structure which makes them easily reusable and mutually composable.

Once the model description is complete, pybinding will build the tight-binding Hamiltonian matrix.
The next step is to apply computations to the matrix to obtain the values of the desired quantum
properties. To that end, there are the following possibilities:

* **Kernel polynomial method** - Pybinding implements a fast Chebyshev polynomial expansion routine
  which can be used to calculate various physical properties. For example, it's possible to
  quickly compute the local density of states or the transport characteristics of the system.

* **Exact diagonalization** - Eigensolvers may be used to calculate the eigenvalues and
  eigenvectors of the model system. Common dense and sparse matrix eigensolvers are available
  via SciPy.

* **User-defined compute** - Pybinding constructs the Hamiltonian in the standard sparse matrix
  CSR format which can be plugged into custom compute routines.

After the main computation is complete, various utility functions are available for post-processing
the raw result data. The included plotting functions are tailored for tight-binding problems to
help visualize the model structure and to make sense of the results.


Citing
------

Pybinding is free to use under the simple conditions of the BSD open source license (included
below). If you wish to use results produced with this package in a scientific publication, please
just mention the package name in the text and cite the Zenodo DOI of this project:

.. raw:: html

    <a href="https://zenodo.org/badge/latestdoi/20541/dean0x7d/pybinding">
        <img src="https://zenodo.org/badge/20541/dean0x7d/pybinding.svg" alt="Zenodo DOI">
    </a>

You'll find a *"Cite as"* section in the bottom right of the Zenodo page. You can select a citation
style from the dropdown menu or export the data in BibTeX and similar formats.


BSD License
-----------

.. include:: ../license.md
