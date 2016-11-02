Advanced Install
================

If you've completed the :doc:`quick` guide, you can skip right to the :doc:`/tutorial/index`.
This section is intended for users who wish to have more control over the install process or
to compile from source code. If you're looking for a simple solution, see the :doc:`quick` guide.


Without Anaconda
----------------

If you already have Python 3.x installed from `python.org`_ or anywhere else, you can use your
existing distribution instead of Anaconda (or Miniconda). Note that this does require manually
installing some dependencies.

.. rubric:: Windows

#. Install the `Visual C++ 2015 Runtime
   <https://www.microsoft.com/en-us/download/details.aspx?id=48145>`_.

#. Install numpy, scipy and matplotlib binaries from `Christoph Gohlke
   <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.

#. Pybinding is available as a binary wheel on `PyPI <https://pypi.python.org/pypi>`_.
   Install it with::

    pip3 install pybinding

.. rubric:: Linux

Building Pybinding from source is the only option on Linux.

#. Make sure you have gcc and g++ v4.8 or newer. To check, run `g++ --version` in your terminal.
   Refer to instruction from your Linux distribution in case you need to upgrade.
#. Install `CMake`_ >= v3.0 from their website or your package manager,
   e.g. `apt-get install cmake`.
#. Install numpy, scipy and matplotlib with the minimal versions as
   :doc:`stated previously </install/index>`. The easiest way is to use your package manager,
   but note that the main repositories tend to keep outdated versions of SciPy packages. For
   instructions on how to compile the latest packages from source, see http://www.scipy.org/.
#. Install Pybinding using pip::

    pip3 install pybinding

.. rubric:: macOS

All the required SciPy packages and Pybinding are available as binary wheels on
`PyPI <https://pypi.python.org/pypi>`_, so the installation is very simple::

    pip3 install pybinding

Note that pip will resolve all the SciPy dependencies automatically.


Compiling from source
---------------------

If you want to get the latest version (the master branch on GitHub), you will need to compile it
from source code. Before you proceed, you'll need to have numpy, scipy and matplotlib. They can
be installed either using Anaconda or following the procedure in the section just above this one.
Once you have everything, follow the steps below to compile and install Pybinding.

.. rubric:: Windows

#. Install `Visual Studio 2015 Community <https://www.visualstudio.com/products/visual-studio-community-vs>`_.
   The Visual C++ compiler is required, so make sure to select it during the customization step
   of the installation (C++ may not be installed by default).
#. Install `CMake`_.
#. Build and install Pybinding. The following command will instruct pip to download the latest
   source code from GitHub, compile everything and install the package::

    pip3 install git+https://github.com/dean0x7d/pybinding.git

.. rubric:: Linux

You'll need gcc/g++ >= v4.8 and CMake >= v3.0. See the previous section for details. If you have
everything, Pybinding can be installed from the latest source code using pip::

    pip3 install git+https://github.com/dean0x7d/pybinding.git

.. rubric:: macOS

#. Install `Homebrew <http://brew.sh/>`_.
#. Install CMake: `brew install cmake`
#. Build and install Pybinding. The following command will instruct pip to download the latest
   source code from GitHub, compile everything and install the package::

    pip3 install git+https://github.com/dean0x7d/pybinding.git


For development
---------------

If you would like to work on the Pybinding source code itself, you can install it in an editable
development environment. The procedure is similar to the "Compiling from source" section with
the exception of the final step:

#. Clone the repository using git (you can change the url to your own GitHub fork)::

    git clone --recursive https://github.com/dean0x7d/pybinding.git

#. Tell pip to install in development mode::

    cd pybinding
    pip3 install -e .


.. _python.org: https://www.python.org/
.. _CMake: https://cmake.org/
