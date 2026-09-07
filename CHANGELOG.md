# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `SolidBodyContact`.
- Added the arguments `facing` and `self_contact` to `SolidBodyContact`. A face-pair is
  only considered by the contact search if the outward unit normal vectors of both faces
  are opposed and if both faces belong to different bodies of the mesh. This makes it
  possible to use all faces on the outline of a mesh as contact surfaces, i.e. without
  restricting the boundary regions to the region of interest.
- Added a test suite in `tests/`, which covers the package by 100%.

### Fixed
- Fixed the package metadata: the project is named `felupe-contact` and its version is
  read from `felupe_contact.__about__`. Both were set to `felupe`, which made the build
  fail with `ModuleNotFoundError: No module named 'felupe'`.
- Fixed the `read-the-docs` dependency group: it now installs the documentation
  requirements of this package, `felupe-contact[docs]`, and the plotting- and
  example-requirements of FElupe, `felupe[all,examples]`. The group referenced
  `felupe[all,docs,examples]`, which was a self-reference to this package due to its
  wrong name.
- Fixed the reference to `SolidBodyContact` in its docstring-example, which was given as
  `fem.SolidBodyContact`. The class is imported from `felupe_contact`.
- Fixed the *See Also*-reference of `SolidBodyContact`, which pointed to the
  non-existent `felupe.ContactRigidPlane`. It now refers to `felupe.MultiPointContact`.

### Changed
- Changed the broad-phase contact search to query the faces of each body of the primary
  surface separately. Otherwise the faces of the own body of an integration point, which
  are always the closest ones, occupy the candidates of a query and hide the faces of
  the other bodies.
- Changed the array-layout of the contact kinematics to the array-layout of FElupe, i.e.
  the axes of the shape functions, of the components and of the natural element
  coordinates are the leading axes and the batch-axes are the trailing axes. This is the
  layout of the elements and of `felupe.math`, hence their functions are re-used instead
  of re-implemented. It also broadcasts the scalar-valued quantities of the face-pairs,
  like the gap, without a reshape and it evaluates about four times faster than the
  batched matrix products on a layout with leading batch-axes.

### Removed
- Removed `shape_function_quad()` and the hard-coded second derivative of the shape
  functions of a bilinear quad. Both are already available in `felupe.Quad`, which
  evaluates its shape functions, their gradients and their hessians for a batch of
  natural element coordinates.
- Removed `solve_2x2()` and `invert_2x2()` in favour of `felupe.math.det()`,
  `felupe.math.inv()` and `felupe.math.dot()`. A singular local equation system of the
  closest-point projection is still regularized, now by passing a regularized
  determinant to `felupe.math.inv()`.
- Removed the hand-written cross products, norms and dyadic products in favour of
  `felupe.math.cross()`, `felupe.math.norm()`, `felupe.math.dya()`,
  `felupe.math.transpose()` and `felupe.math.dot()`. The outward unit normal vectors of
  both contact surfaces are now evaluated by a single helper `unit_normals()`.
- Removed the hand-written interpolation of the deformed coordinates at the integration
  points of the secondary surface in favour of `felupe.Field.interpolate()`.
