# Changelog

### v0.7.2 | in development

* The `sub_id` modifier argument can now be compared directly with a sublattice name. For example,
  this makes it possible to write `sub_id == 'A'` instead of the old `sub_id == lattice['A']`.

* Added optional `sites` argument for state, position, and onsite energy modifiers.
  It can be used instead of the `x, y, z, sub_id` arguments and contains a few helper methods.
  See the modifier API reference for more information.

* Fixed a bug where using a single KPM object for multiple calculations could return wrong results


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
