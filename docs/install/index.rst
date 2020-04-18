Installation
============

Pybinding can be installed on Windows, Linux or Mac, with the following prerequisites:

* `Python`_ 3.6 or newer (Python 2.x is not supported)
* The `SciPy`_ stack of scientific packages, with required versions:

  * numpy >= v1.12
  * scipy >= v0.19
  * matplotlib >= v2.0

* If you're using Linux, you'll also need GCC >= v5.0 (or clang >= v3.5) and CMake >= v3.1.

You can install all of this in two ways:

.. toctree::
    :maxdepth: 1

    quick
    advanced

If you are new to Python/SciPy or if you're just not sure how to proceed, go with the :doc:`quick`
option. It will show you how to easily set up a new Python environment and install everything.
That quick guide will be everything you need in most cases. However, If you would like a custom
setup within your existing Python environment and you have experience compiling binary packages,
you can check out the :doc:`advanced` option.

.. _Python: https://www.python.org/
.. _SciPy: http://www.scipy.org/
