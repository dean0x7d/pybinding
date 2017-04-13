CUDA-based KPM
==============

`CUDA <https://developer.nvidia.com/about-cuda>`_ enables the execution of general purpose code
on Nvidia GPUs. It can be used to accelerate computational algorithms which feature natural
parallelism. Pybinding features experimental support for CUDA. It's used for kernel polynomial
method (KPM) calculations -- see :doc:`tutorial page</tutorial/kpm>` and
:mod:`API reference <.chebyshev>`.

The CUDA-base KPM implementation is available via the :func:`.kpm_cuda` function. It mirrors
the API of the regular CPU-based :func:`.kpm`. The only difference between them is where the
calculation will take place. Note that the CUDA implementation is still experimental and that only
diagonal Green's function elements will be computed on the GPU, while off-diagonal falls back to
regular CPU code. This will be addressed in a future version.

By default, CUDA support is disabled. You will need to turn it on manually by recompiling the
package. First, ensure that you have `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_
7.5 or newer installed. Next, remove any existing pybinding installation by executing the following
command in terminal::

    pip3 uninstall pybinding

Finally, reinstall it with CUDA turned on::

    PB_CUDA=ON pip3 install pybinding --no-binary pybinding

Note that `pybinding` is written twice. This is not a mistake. The `--no-binary pybinding` flag
tells pip to compile from source. Since this is all experimental: expect errors and no support.
