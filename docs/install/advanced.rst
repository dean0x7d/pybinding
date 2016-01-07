Advanced Install
================

If you've completed the :doc:`quick` guide, you can skip right to the :doc:`/tutorial/index`.
This section is intended for users who wish to have more control over the install process or
to compile from source code. If you're looking for a simple solution, see the :doc:`quick` guide.


Windows
-------

We'll assume that you already have Python 3.x installed from `python.org`_ or anywhere else.

Binary install
**************

1. Install the `Visual C++ 2015 Runtime
   <https://www.microsoft.com/en-us/download/details.aspx?id=48145>`_.

2. Install numpy, scipy and matplotlib binaries from `Christoph Gohlke
   <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.

3. Pybinding is available as a binary wheel on `PyPI <https://pypi.python.org/pypi>`_.
   Install it with::

    pip3 install pybinding

Compile from source
*******************

To compile Pybinding from source, follow the first two steps from above. Then you will need to:

#. Install `Visual Studio 2015 Community.
   <https://www.visualstudio.com/products/visual-studio-community-vs>`_
#. Install `CMake`_.
#. Get the Pybinding source files. Build and install with::

    python3 setup.py install


Linux
-----

We'll assume that you have Python 3.x as part of your Linux distribution.
Building Pybinding from source is the only option on Linux.

#. Make sure you have gcc and g++ 4.8 or newer: type `g++ --version` in terminal.
   Refer to instruction from your Linux distribution in case you need to upgrade.
#. Install `CMake`_ >= v3.0 from their website or your package manager,
   e.g. `apt-get install cmake`.
#. Install numpy, scipy and matplotlib with the minimal versions as
   :doc:`stated previously </install/index>`. The easiest way to use you package manage, but note
   that the main repositories tend to keep outdated versions of SciPy packages. For instructions
   on how to compile the latest packages from source, see http://www.scipy.org/ .
#. Get the Pybinding source files. Build and install with::

    python3 setup.py install


Mac OS X
--------

We'll assume that you already have Python 3.x installed from `python.org`_
or from `homebrew <http://brew.sh/>`_.

Binary install
**************

All the required SciPy packages and Pybinding are available as a binary wheels on
`PyPI <https://pypi.python.org/pypi>`_, so the installation is very simple::

    pip3 install pybinding

Note that pip will resolve all the SciPy dependecies automatically.

Compile from source
*******************

#. Install `CMake <https://cmake.org/>`_ from their website or with homebrew: `brew install cmake`.
#. Get the Pybinding source files. Build and install with::

    python3 setup.py install


.. _python.org: https://www.python.org/
.. _CMake: https://cmake.org/
