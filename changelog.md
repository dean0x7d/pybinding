# Changelog

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
