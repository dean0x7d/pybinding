Quick Install
=============

The easiest way to install Python and SciPy is with `Anaconda`_, a free scientific Python
distribution for Windows, Linux and Mac. The following install guide will show you how to
install the minimal version of Anaconda, `Miniconda`_, and then install pybinding.

.. note::
   If you run into any problems during the install process,
   check out the :ref:`troubleshooting` section.


.. _Anaconda: https://www.continuum.io/downloads
.. _Miniconda: http://conda.pydata.org/miniconda.html

.. _Miniconda3-latest-Windows-x86_64.exe:
   https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
.. _Miniconda3-latest-Linux-x86_64.sh:
   https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
.. _Miniconda3-latest-MacOSX-x86_64.sh:
   https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh


Windows
-------

#. Download the Miniconda Python 3.x installer: `Miniconda3-latest-Windows-x86_64.exe`_.
   Run it and accept the default options during the installation.

2. Open `Command Prompt` from the `Start` menu. Enter the following command to install
   the scientific Python packages with Miniconda::

    conda install numpy scipy matplotlib

3. The next command will download and install pybinding::

    pip install pybinding

That's it, all done. Check out the :doc:`Tutorial </tutorial/index>` for some example scripts to
get started. To run a script file, e.g. `example1.py`, enter the following command::

    python example1.py


Linux
-----

You will need gcc and g++ 5.0 or newer. To check, enter the following in terminal::

    g++ --version

If your version is outdated, check with your Linux distribution on how to upgrade.
If you have version 5.8 or newer, proceed with the installation.

#. Download the Miniconda Python 3.x installer: `Miniconda3-latest-Linux-x86_64.sh`_. Run it
   in your terminal window::

    bash Miniconda3-latest-Linux-x86_64.sh

   Follow the installation steps. You can accept most of the default values, but make sure
   that you type `yes` to add Miniconda to `PATH`::

       Do you wish the installer to prepend the Miniconda3 install location
       to PATH in your /home/<user_name>/.bashrc ? [yes|no]
       [no] >>> yes

   Now, close your terminal window and open a new one for the changes to take effect.

2. Install CMake and the scientific Python packages::

    conda install cmake numpy scipy matplotlib

3. The next command will download and install pybinding::

    pip install pybinding

That's it, all done. Check out the :doc:`Tutorial </tutorial/index>` for some example scripts to
get started. To run a script file, e.g. `example1.py`, enter the following command::

    python example1.py


Mac OS X
--------

#. Download the Miniconda Python 3.x installer: `Miniconda3-latest-MacOSX-x86_64.sh`_. Run it
   in your terminal window::

    bash Miniconda3-latest-MacOSX-x86_64.sh

   Follow the installation steps. You can accept most of the default values, but make sure
   that you type `yes` to add Miniconda to `PATH`::

        Do you wish the installer to prepend the Miniconda3 install location
        to PATH in your /Users/<user_name>/.bash_profile ? [yes|no]
        [yes] >>> yes

   Now, close your terminal window and open a new one for the changes to take effect.

2. Install CMake and the scientific Python packages::

    conda install cmake numpy scipy matplotlib

3. The next command will download and install pybinding::

    pip install pybinding

That's it, all done. Check out the :doc:`Tutorial </tutorial/index>` for some example scripts to
get started. To run a script file, e.g. `example1.py`, enter the following command::

    python example1.py


.. _troubleshooting:

Troubleshooting
---------------

If you already had Python installed, having multiple distributions may cause trouble in some cases.
Check the `PATH` environment variable and make sure the Miniconda has priority.
