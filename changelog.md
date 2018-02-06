# Changelog

## v0.9.5 | In development

* Fixed installation errors due to the encoding of the `changelog.md` file 
  ([#7](https://github.com/dean0x7d/pybinding/issues/7)).


## v0.9.4 | 2017-07-13

* Fixed issues with multi-orbital models: matrix onsite terms were not set correctly if all the
  elements on the main diagonal were zero ([#5](https://github.com/dean0x7d/pybinding/issues/5)),
  hopping terms were being applied asymmetrically for large multi-orbital systems
  ([#6](https://github.com/dean0x7d/pybinding/issues/6)). Thanks to
  [@oroszl (László Oroszlány)](https://github.com/oroszl) for reporting the issues.

* Fixed KPM Hamiltonian scaling for models with all zeros on the main diagonal but asymmetric
  spectrum bounds (non-zero KPM scaling factor `b`).

* Fixed compilation on certain Linux distributions
  ([#4](https://github.com/dean0x7d/pybinding/issues/4)). Thanks to
  [@nu11us (Will Eggleston)](https://github.com/nu11us) for reporting the issue.

* Fixed compilation with Visual Studio 2017.

* Improved support for plotting slices of multi-layer systems. See "Plotting Guide" > "Model
  structure" > "Slicing layers" in the documentation.


## v0.9.3 | 2017-05-29

* Added support for Kwant v1.3.x and improved `Model.tokwant()` exporting of multi-orbital models.

* Fixed errors when compiling with GCC 6.


## v0.9.2 | 2017-05-26

#### New KPM features and improvements

* Added a method for calculating spatial LDOS using KPM. See the "Kernel Polynomial Method"
  tutorial page and the `KPM.calc_spatial_ldos` API reference.

* Improved single-threaded performance of `KPM.calc_dos` by ~2x by switching to a more efficient
  vectorization method. (Multiple random starter vectors are now computed simultaneously and
  accelerated using SIMD intrinsics.)

* Various KPM methods now take advantage of multiple threads. This improves performance depending
  on the number of cores on the target machine. (However, for large systems performance is limited
  by RAM bandwidth, not necessarily core count.)

* LDOS calculations for multiple orbitals also take advantage of the same vectorization and
  multi-threading improvements. Single-orbital LDOS does not benefit from this but it has received
  its own modest performance tweaks.

* Long running KPM calculation now have a progress indicator and estimated completion time.

#### General improvements and bug fixes

* `StructureMap` can now be sliced using a shape. E.g. `s = pb.rectangle(5, 5); smap2 = smap[s]`
  which returns a smaller structure map cut down to the given shape.

* Plotting the structure of large or periodic systems is slightly faster now.

* Added 2D periodic supercells to the "Shape and symmetry" section of the tutorial.

* Added a few more examples to the "Plotting guide" (view rotation, separating sites and hoppings
  and composing multiple plots).

* Fixed broken documentation links when using the online search function.

* Fixed slow Hamiltonian build when hopping generators are used.


## v0.9.1 | 2017-04-28

* Fixed an issue with multi-orbital models where onsite/hopping modifiers would return unexpected
  results if a new `energy` array was returned (rather than being modified in place).

* Fixed `Solver.calc_spatial_ldos` and `Solver.calc_probability` returning single-orbital results
  for multi-orbital models.

* Fixed slicing of `Structure` objects and made access to the `data` property of `SpatialMap` and
  `StructureMap` mutable again.


## v0.9.0 | 2017-04-14

#### Updated requirements

* This version includes extensive internal improvements and raises the minimum requirements for
  installation. Starting with this release, only Python >= 3.5 is supported. Newer versions of the
  scientific Python packages are also required: numpy >= 1.12, scipy >= 0.19 and matplotlib >= 2.0.

* On Linux, the minimum compiler requirements have also been increased to get access to C++14 for
  the core of the library. To compile from source, you'll need GCC >= 5.0 or clang >= 3.5.

#### Multi-orbital models

* Improved support for models with multiple orbitals, spins and any additional degrees of freedom.
  These can now be specified simply by inputing a matrix as the onsite or hopping term (instead of
  a scalar value). For more details, see the "Multi-orbital models" section of the documentation.

* Lifted all limits on the number of sublattices and hoppings which can be defined in a `Lattice`
  object. The previous version was limited to a maximum of 128 onsite and hopping terms per unit
  cell (but those could be repeated an unlimited number of times to form a complete system). All
  restrictions are now removed so that the unit cell size is only limited by available memory.
  In addition, the memory usage of the internal system format has been reduced.

* Added a 3-band model of group 6 transition metal dichalcogenides to the Material Repository.
  The available TMDs include: MoS2, WS2, MoSe2, WSe2, MoTe2, WTe2. These are all monolayers.

#### Composite shapes

* Complicated system geometries can now be created easily by composing multiple simple shapes.
  This is done using set operations, e.g. unions, intersections, etc. A complete guide for this
  functionality is available in the "Composite shapes" section of the documentation.

#### Kernel polynomial method

* The KPM implementation has been revised and significantly expanded. A guide and several examples
  are available in the "Kernel polynomial method" section of the documentation (part 9 of the
  Tutorial). For a complete overview of the available methods and kernels, see the `chebyshev`
  section of the API reference.

* New builtin computation methods include the stochastically-evaluated density of states (DOS)
  and electrical conductivity (using the Kubo-Bastin approach).

* The new low-level interface produces KPM expansion moments which allows users to create their
  own KPM-based computation routines.

* The performance of various KPM computations has been significantly improved for CPUs with AVX
  support (~1.5x speedup on average, but also up to 2x in some cases with complex numbers).

#### Miscellaneous

* Added the `pb.save()` and `pb.load()` convenience functions for getting result objects into/out
  of files. The data is saved in a compressed binary format (Python's builtin `pickle` format with
  protocol 4 and gzip). Loaded files can be immediately plotted: `result = pb.load("file.pbz")`
  and then `result.plot()` to see the data.

* The eigenvalue solvers now have a `calc_ldos` method for computing the local density of states
  as a function of energy (in addition to the existing `calc_spatial_ldos`).

* Improved plotting of `Lattice` objects. The view can now be rotated by passing the `axis="xz"`
  argument, or any other combination of x, y and z to define the plotting plane.

#### Deprecations and breaking changes

* Added `Lattice.add_aliases()` method. The old `Lattice.add_sublattice(..., alias=name)` way of
  creating aliases is deprecated.

* The `greens` module has been deprecated. This functionality is now covered by the KPM methods.

* The internal storage format of the `Lattice` and `System` classes has been revised. This
  shouldn't affect most users who don't need access to the low-level data.


## v0.8.2 | 2017-01-26

* Added support for Python 3.6 (pybinding is available as a binary wheel for Windows and macOS).

* Fixed compatibility with matplotlib v2.0.

* Fixed a few minor bugs.


## v0.8.1 | 2016-11-11

* Structure plotting functions have been improved with better automatic scaling of lattice site
  circle sizes and hopping line widths.

* Fixed Brillouin zone calculation for cases where the angle between lattice vectors is obtuse
  ([#1](https://github.com/dean0x7d/pybinding/issues/1)). Thanks to
  [@obgeneralao (Oliver B Generalao)](https://github.com/obgeneralao) for reporting the issue.

* Fixed a flaw in the example of a phosphorene lattice (there were extraneous t5 hoppings).
  Thanks to Longlong Li for pointing this out.

* Fixed missing CUDA source files in PyPI sdist package.

* Revised advanced installation instructions: compiling from source code and development.


## v0.8.0 | 2016-07-01

#### New features

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

#### Improvements

* The performance of the KPM Green's function implementation has been improved significantly:
  by a factor of 2.5x. The speedup was achieved with CPU code using portable SIMD intrinsics
  thanks to [libsimdpp](https://github.com/p12tic/libsimdpp).

* The Green's function can now be computed for multiple indices simultaneously.

* The spatial origin of a lattice can be adjusted using the `Lattice.offset` attribute. See the
  "Advanced Topics" section.

#### Breaking changes

* The interface for structure plotting (as used in `System.plot()` and `StructureMap`) has been
  greatly improved. Some of the changes are not backwards compatible and may require some minor
  code changes after upgrading. See the "Plotting Guide" section of the documentation for details.

* The interfaces for the `Bands` and `StructureMap` result objects have been revised. Specifically,
  structure maps are now more consistent with ndarrays, so the old `smap.filter(smap.x > 0)` is
  replaced by `smap2 = smap[smap.x > 0]`. The "Plotting Guide" has a few examples and there is a
  full method listing in the "API Reference" section.

#### Documentation

* The API reference has been completely revised and now includes a summary on the main page.

* A few advanced topics are now covered, including some aspects of plotting. A few more random
  examples have also been added.

* Experimental features are now documented.

#### Bug fixes

* Fixed translational symmetry skipping directions for some 2D systems.
* Fixed computation of off-diagonal Green's function elements with `opt_level > 0`
* Fixed some issues with shapes which were not centered at `(x, y) = (0, 0)`.


## v0.7.2 | 2016-03-14

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


## v0.7.1 | 2016-02-08

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


## v0.7.0 | 2016-02-01

Initial release
