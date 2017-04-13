# Changelog

### v0.9.0 | in development

##### Updated requirements

* This version includes extensive internal improvements and raises the minimum requirements for
  installation. Starting with this release, only Python >= 3.5 is supported. Newer versions of the
  scientific Python packages are also required: numpy >= 1.12, scipy >= 0.19 and matplotlib >= 2.0.

* On Linux, the minimum compiler requirements have also been increased to get access to C++14 for
  the core of the library. To compile from source, you'll need GCC >= 5.0 or clang >= 3.5.

##### New features

* Improved support for models with multiple orbitals, spins and any additional degrees of freedom.
  These can now be specified simply by inputing a matrix as the onsite or hopping term, instead of
  a scalar value. For more details, see the "Multi-orbital models" section of the documentation.

* Lifted all limits on the number of sublattices and hoppings which can be defined in a `Lattice`
  object. The previous version was limited to a maximum of 128 onsite and hopping terms per unit
  cell (but those could be repeated an unlimited number of times to form a complete system). All
  restrictions are now removed so that the unit cell size is only limited by available memory.
  In addition, the memory usage of the internal system format has been reduced.

* Added the `pb.save()` and `pb.load()` convenience functions for getting result objects into/out
  of files. The data is saved in a compressed binary format (Python's builtin `pickle` format set
  to protocol 4 + gzip). Loaded files can be immediately plotted: `result = pb.load("file.pbz")`
  and then `result.plot()` to see the data.

* Added `Lattice.add_aliases()` method. The old `Lattice.add_sublattice(..., alias=name)` way of
  creating aliases is deprecated.


### v0.8.2 | 2017-01-26

* Added support for Python 3.6 (pybinding is available as a binary wheel for Windows and macOS).

* Fixed compatibility with matplotlib v2.0.

* Fixed a few minor bugs.


### v0.8.1 | 2016-11-11

* Structure plotting functions have been improved with better automatic scaling of lattice site
  circle sizes and hopping line widths.

* Fixed Brillouin zone calculation for cases where the angle between lattice vectors is obtuse
  ([#1](https://github.com/dean0x7d/pybinding/issues/1)). Thanks to
  [@obgeneralao (Oliver B Generalao)](https://github.com/obgeneralao) for reporting the issue.

* Fixed a flaw in the example of a phosphorene lattice (there were extraneous t5 hoppings).
  Thanks to Longlong Li for pointing this out.

* Fixed missing CUDA source files in PyPI sdist package.

* Revised advanced installation instructions: compiling from source code and development.


### v0.8.0 | 2016-07-01

##### New features

* Added support for scattering models. Semi-infinite leads can be attached to a finite-sized
  scattering region. Take a look at the documentation, specifically section 10 of the "Basic
  Tutorial", for details on how to construct such models.

* Added compatibility with [Kwant](http://kwant-project.org/) for transport calculations. A model
  can be constructed in pybinding and then exported using the `Model.tokwant()` method. This makes
  it possible to use Kwant's excellent solver for transport problems. While Kwant does have its
  own model builder, pybinding is much faster in this regard: by two orders of magnitude, see the
  "Benchmarks" page in the documentation for a performance comparison.

* *Experimental:* Initial CUDA implementation of KPM Green's function (only for diagonal elements
  for now). See the "Experimental Features" section of the documentation.

##### Improvements

* The performance of the KPM Green's function implementation has been improved significantly:
  by a factor of 2.5x. The speedup was achieved with CPU code using portable SIMD intrinsics
  thanks to [libsimdpp](https://github.com/p12tic/libsimdpp).

* The Green's function can now be computed for multiple indices simultaneously.

* The spatial origin of a lattice can be adjusted using the `Lattice.offset` attribute. See the
  "Advanced Topics" section.

##### Breaking changes

* The interface for structure plotting (as used in `System.plot()` and `StructureMap`) has been
  greatly improved. Some of the changes are not backwards compatible and may require some minor
  code changes after upgrading. See the "Plotting Guide" section of the documentation for details.

* The interfaces for the `Bands` and `StructureMap` result objects have been revised. Specifically,
  structure maps are now more consistent with ndarrays, so the old `smap.filter(smap.x > 0)` is
  replaced by `smap2 = smap[smap.x > 0]`. The "Plotting Guide" has a few examples and there is a
  full method listing in the "API Reference" section.

##### Documentation

* The API reference has been completely revised and now includes a summary on the main page.

* A few advanced topics are now covered, including some aspects of plotting. A few more random
  examples have also been added.

* Experimental features are now documented.

##### Bug fixes

* Fixed translational symmetry skipping directions for some 2D systems.
* Fixed computation of off-diagonal Green's function elements with `opt_level > 0`
* Fixed some issues with shapes which were not centered at `(x, y) = (0, 0)`.


### v0.7.2 | 2016-03-14

* Lots of improvements to the documentation. The tutorial pages can now be downloaded and run
  interactively as Jupyter notebooks. The entire user guide is also available as a PDF file.

* The `sub_id` and `hop_id` modifier arguments can now be compared directly with their friendly
  string names. For example, this makes it possible to write `sub_id == 'A'` instead of the old
  `sub_id == lattice['A']` and `hop_id == 'gamma1'` instead of `hop_id == lattice('gamma1')`.

* The site state modifier can automatically remove dangling sites which have less than a certain
  number of neighbors (set using the `min_neighbors` decorator argument).

* Added optional `sites` argument for state, position, and onsite energy modifiers.
  It can be used instead of the `x, y, z, sub_id` arguments and contains a few helper methods.
  See the modifier API reference for more information.

* Fixed a bug where using a single KPM object for multiple calculations could return wrong results.

* *Experimental* `hopping_generator` which can be used to add a new hopping family connecting
  arbitrary sites independent of the main `Lattice` definition. This is useful for creating
  additional local hoppings, e.g. to model defects.


### v0.7.1 | 2016-02-08

* Added support for double-precision floating point. Single precision is used by default,
  but it will be switched automatically to double if required by an onsite or hopping modifier.

* Added support for the 32-bit version of Python

* Tests are now included in the installed package. They can be run with:

  ```python
  import pybinding as pb
  pb.tests()
  ```

* Available as a binary wheel for 32-bit and 64-bit Windows (Python 3.5 only)
  and OS X (Python 3.4 and 3.5)


### v0.7.0 | 2016-02-01

Initial release
