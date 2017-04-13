FEAST eigensolver
=================

The `FEAST <http://www.ecs.umass.edu/~polizzi/feast/>`_ eigensolver significantly differs from
traditional solvers like the ones found in LAPACK and ARPACK. It takes its inspiration from the
density-matrix representation and contour integration in quantum mechanics. When solving a series
of eigenvalue problems which are close to one another, as is the case for band structure
calculations, the results of the previous calculation can be used as the starting point for the
next. The algorithm also features natural parallelism where different eigenvalues can be computed
separately without overlap.

Pybinding has experimental support for this solver. It can be accessed via :func:`.solver.feast`.
However, it is disabled by default and you will need to recompile the package in order to install
it. Since FEAST requires Intel PARDISO, you will need to have
`Intel MKL <https://software.intel.com/en-us/intel-mkl>`_ installed before you continue. Next,
remove any existing pybinding installation by executing the following command in terminal::

    pip3 uninstall pybinding

Finally, reinstall it with MKL turned on::

    PB_MKL=ON pip3 install pybinding --no-binary pybinding

Note that `pybinding` is written twice. This is not a mistake. The `--no-binary pybinding` flag
tells pip to compile from source. Since this is all experimental: expect errors and no support.
