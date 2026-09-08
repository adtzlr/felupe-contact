# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `SolidBodyContact`.
- Added support for two-dimensional problems in `SolidBodyContact`, i.e. for plane
  strain and axisymmetric problems. The faces of a boundary region of a two-dimensional
  mesh are its edges, hence a face is spanned by one instead of two natural element
  coordinates and its outward unit normal vector is its in-plane vector, rotated by
  minus 90 degrees. In 2d, the out-of-plane behaviour must be defined by the type of the
  displacement field: a `FieldPlaneStrain` or a `FieldAxisymmetric` is required and a
  (cartesian) `Field` is rejected. In 3d, only a `Field` is valid.
- Added support for all boundary regions of FElupe in `SolidBodyContact`, i.e. for
  `RegionQuadBoundary`, `RegionQuadraticQuadBoundary`, `RegionBiQuadraticQuadBoundary`,
  `RegionHexahedronBoundary`, `RegionQuadraticHexahedronBoundary` and
  `RegionTriQuadraticHexahedronBoundary`. The contact was limited to boundary regions of
  hexahedron meshes, i.e. to bilinear quad faces.
- Added `felupe_contact._element.BatchedElement`, which evaluates the shape functions of
  an element of FElupe and their first and second partial derivatives for a batch of
  natural element coordinates. The elements of FElupe evaluate their shape functions at
  a single point, the closest-point projection requires them at an individual point per
  face-pair. The shape functions are expanded exactly in the monomials of a polynomial
  basis on initiation and the expansion is verified, hence no shape functions are
  re-implemented.
- Added `felupe_contact._element.face_element()`, which returns the element formulation
  of the faces of a boundary region along with the indices of the points of a face
  within the points of a cell of the boundary region.
- Added `ContactSurfacePair.variations_gap()`, which returns the variation of the gap.
  It is used by both the vector- and the matrix-assembly.
- Added the examples Ex. 4 (plane strain), Ex. 5 (axisymmetric) and Ex. 6 (a boundary
  region with quadratic faces).
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
- Changed the assembly of the block of the contact stiffness matrix and of the part of
  the contact force vector, which contain only test- and trial-functions of the
  secondary surface, from an `IntegralForm` to the direct assembly which is already used
  for the coupling- and the primary-blocks. This unifies the treatment of both contact
  surfaces and it is required for axisymmetric fields: `IntegralForm` evaluates
  `fun[-1] / R` for an axisymmetric field, which is not defined on the rotation axis
  `R = 0`, and it raises an `UnboundLocalError` for a bilinear form with
  `grad_v=False` and `grad_u=False`. It is also faster, because an integral form
  assembles a dense sub-matrix for every cell of the boundary region while only a few
  integration points are in contact.
- Changed the argument `parallel` of `SolidBodyContact.assemble.vector()` and
  `SolidBodyContact.assemble.matrix()` to only activate the threaded tree-query of the
  broad-phase contact search. It activated the threaded assembly of the integral form of
  the secondary surface, which is no longer used.
- Changed `unit_normals()` to take the in-plane (tangent) vectors of the faces as a
  single array instead of two separate arrays. The normal vector of a face of a
  three-dimensional mesh is the cross product of its two in-plane vectors, the normal
  vector of a face (edge) of a two-dimensional mesh is its in-plane vector, rotated by
  minus 90 degrees.
- Changed `closest_point_projection()` to take the element formulation of the faces.
  The second derivatives of the surface coordinates are now evaluated by the hessian of
  the shape functions of the faces instead of the constant mixed second derivative of a
  bilinear quad, i.e. curved faces are supported.
- Changed the characteristic size of the faces of the contact surfaces to the
  `(dim - 1)`-th root of their mean differential volume. This is the edge length of a
  face of a three-dimensional mesh, as before, and the length of a face (edge) of a
  two-dimensional mesh.
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
- Removed the module-level constants `QUAD` and `D2HDRDS`, which hard-coded the faces of
  a boundary region as bilinear quads. The element formulation of the faces is now
  obtained from the cell type of the mesh of a boundary region.
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
