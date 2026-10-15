# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `SolidBodyContact`.
- Added the methods `SolidBodyContact.checkpoint()` and `SolidBodyContact.restore()`,
  which are used by the `CutbackPlugin` of FElupe (v11.2.0 and later). The faces of the
  primary surface, to which the integration points of the secondary surface are
  assigned, are modified by the iterations of the Newton-Raphson method (hysteresis of
  the contact search). On a cutback, the faces of the last converged increment are
  restored, i.e. the next attempt doesn't start with the faces of the failed attempt.
  The cached kinematics and the assembled force vector and stiffness matrix are reset.
- Added tests which check that `SolidBodyContact` is an item of FElupe
  (`felupe.mechanics.Item`, v11.3.0 and later) and that a large increment, which fails
  without a cutback, is solved with the `CutbackPlugin` and with the `LinesearchPlugin`.
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
- Added a mode of `SolidBodyContact` which works with solid bodies directly, e.g.
  `SolidBodyContact(solid)` or `SolidBodyContact(solid_a, solid_b)`. The boundary region
  is created for the region of the displacement field of a solid body by the dict
  `felupe_contact._boundary.BOUNDARY_REGIONS` (region -> boundary region, e.g.
  `RegionHexahedron` -> `RegionHexahedronBoundary`) with a displacement field of the
  same type, whose values are linked to the solid body. All faces on the outline of the
  mesh are used, or the faces within the new arguments `mask` and `mask_primary`. A
  field container on the region of a solid body is supported as well. The argument
  `field_primary` is now optional: if None, the secondary surface is also the primary
  surface. If no `items` are given, the solid bodies are used to estimate the penalty
  stiffness. Field containers on boundary regions are still supported.
- Added `felupe_contact._boundary.boundary_region()` and
  `felupe_contact._boundary.contact_surface()`, which create the contact surfaces of
  the solid bodies.
- Added `BatchedElement.evaluate()`, which evaluates the shape functions and all their
  first and second partial derivatives by a single evaluation of the monomials and a
  single matrix product. The coefficients of the derivatives are stacked on initiation.
- Added `BatchedElement.control_net()`, which returns the transformation of the points
  of a face into the control points of a Bézier patch on an extended range of the
  natural element coordinates. The convex hull of the control points contains the face,
  which gives rigorous bounds of the faces for the contact search.
- Added `ContactSurfacePair.bounds()`, `ContactSurfacePair.candidates()` and
  `ContactSurfacePair.discard()`, the phases of the contact search before the
  closest-point projection, see the changes of the contact search below.
- Added the property `ContactSurfacePair.nneighbours`, the maximum number of faces of
  the primary surface which share a point with a face of the secondary surface.
- Added a warning for self-contact, if integration points are in contact with their own
  body in the undeformed configuration, i.e. if `max_distance` exceeds the thickness of
  a body.
- Added `ContactSurfacePair.dofs()`, `ContactSurfacePair.vector()` and
  `ContactSurfacePair.matrix()`, which return the degrees of freedom and the force
  vectors and stiffness matrices of the face-pairs, and the helpers `sparse_vector()`,
  `sparsity_pattern()` and `sparse_matrix()` for their assembly.
- Added the tests `tests/test_solidbody_mode.py` and `tests/test_contact_search.py`.
- Added the documentation, which is built by Sphinx (with the PyData theme, like FElupe)
  on Read the Docs. It contains an API reference and a gallery of the examples, which
  are run by Sphinx-Gallery. The documentation was configured in `.readthedocs.yaml` but
  its source in `docs/` was missing.
- Added the first version of the content of the documentation, which presents
  felupe-contact as an extension package for FElupe: the landing page compares the
  contact formulations of FElupe and felupe-contact, a *Getting Started* guide adds a
  contact to a FElupe-script step by step and the *How-To* guides explain the definition
  of the contact surfaces, plane strain and axisymmetric contact, the penalty stiffness
  and the convergence (incl. the `CutbackPlugin` and the `LinesearchPlugin` of FElupe)
  and the contact results. The API reference lists the related classes of FElupe.
- Added a `README.md`, which is required by the package metadata, a contribution guide
  `CONTRIBUTING.md`, a code of conduct `CODE_OF_CONDUCT.md`, a `CITATION.cff` and issue
  templates for bug reports and enhancement suggestions, like in FElupe.
- Added the linting and the formatting of the code by ruff with the rules of FElupe,
  configured in `pyproject.toml`. The checks are run locally by the hooks of pre-commit
  (`.pre-commit-config.yaml`) and by the workflow `lint.yml` for each pull request.
- Added the examples of the docstrings to the tests, i.e. `pytest` runs the tests in
  `tests/` and the doctests in `src/felupe_contact`, configured in `pyproject.toml`. The
  plots of the examples are rendered off-screen and the examples are run in a temporary
  directory, see `conftest.py` (like FElupe). The code-examples of the documentation are
  tested as doctests of the rst-files in `docs/`, i.e. they are also tested with the
  development version of FElupe.
- Added the tox-environment `main`, which runs the tests with the development version
  of FElupe, i.e. with its `main` branch. The workflow `upload-codecov.yml` runs it for
  Python 3.14 after the tests with the latest release of FElupe, i.e. changes of the API
  of FElupe are detected before they are released.
- Added optional Coulomb friction to `SolidBodyContact` by the friction coefficient
  `friction` (default is 0.0, i.e. frictionless; `np.inf` prevents sliding). The
  tangential tractions are evaluated by a return mapping from the last converged
  increment with an elastic (penalty) stick, like the friction of `ContactRigidPlane` of
  FElupe. The anchor (stick point) of an integration point is a material point of the
  primary surface, i.e. it moves with the primary surface. It is transferred to the
  current face if the integration point slides over an edge of a face, and an
  integration point which comes into contact within an increment is anchored at the
  (interpolated) point of first contact. The tangential traction acts on the integration
  point of the secondary surface and, equal and opposite, on the projected point of the
  primary surface. The consistent stiffness matrix contains the change of the normal
  vector, of the projected point and of the contact pressure. It is not symmetric, its
  symmetric part is used with `symmetric=True`.
- Added the arguments `elastic_slip` and `penalty_tangential` to `SolidBodyContact`.
  The tangential penalty stiffness is proportional to the contact pressure, i.e. a
  sticking integration point starts to slide at the allowable elastic slip (default is
  one percent of the characteristic size of the faces), and it is limited by the normal
  penalty stiffness. Hence, the tangential stiffness vanishes smoothly at the boundary
  of a contact zone and it is not overestimated for large penetrations in the first
  iterations of an increment. Alternatively, a constant tangential penalty stiffness is
  given by `penalty_tangential`. Both are evaluated by
  `SolidBodyContact.tangential_penalty()`.
- Added the state variables of the friction to `SolidBodyContact.results.statevars`
  (the anchors and the deformed coordinates of the last converged increment). The trial
  state variables are accepted after a converged increment by FElupe, see
  `felupe.mechanics.Results.update_statevars()`, and they are included in
  `SolidBodyContact.checkpoint()` and restored by `SolidBodyContact.restore()`, i.e. on
  a cutback of the `CutbackPlugin`. The tangential tractions and the mask of the sliding
  integration points are stored in `SolidBodyContact.results.shear` and
  `SolidBodyContact.results.slip`.
- Added `ContactSurfacePair.anchors()`, `ContactSurfacePair.relocate()`,
  `ContactSurfacePair.friction()`, `ContactSurfacePair.vector_friction()` and
  `ContactSurfacePair.matrix_friction()`, which evaluate the friction of the face-pairs.
  The anchors of sliding integration points are relocated exactly, i.e. the tangential
  tractions are reproduced at the beginning of the next increment.
- Added the tests `tests/test_friction.py` (incl. the tangent of the force vector for
  all boundary regions, gross sliding on a stiff base with a tangential force of the
  friction coefficient times the normal force and a cutback, which gives the same
  result as smaller increments), the how-to guide *Friction* and Ex. 7, a rubber block
  which is moved over a stiff base.
- Added `SolidBodyContactNitsche`, a frictionless contact between the surfaces of solid
  bodies, enforced by Nitsche's method (non-symmetric variant, θ = 0). The contact
  pressure `[λ_N - γ g]_+` contains the normal stress `λ_N = (P N) · n` of the solid
  bodies, evaluated by their material models at the integration points of the contact
  surfaces. Hence, the formulation is consistent: for a homogeneous stress state on
  matching meshes, the gap is zero and the contact pressure is equal to the normal
  stress, independent of the Nitsche parameter (patch test). The contact is created on
  solid bodies (`SolidBody` or `SolidBodyNearlyIncompressible` with the cell-wise
  constant pressure), it supports three-dimensional, plane strain and axisymmetric
  problems, all boundary regions of FElupe and material models with state variables,
  which are evaluated at the integration points of the contact surfaces and updated
  after a converged increment. The consistent stiffness matrix contains the elasticity
  tensors of the material models w.r.t. the displacements of all points of the cells of
  the faces. It is not symmetric.
- Added the weighted two-pass evaluation of `SolidBodyContactNitsche` (default). The
  contact is evaluated on both surfaces of a contact zone with the stress of each body
  on its own surface, and the contact tractions of an integration point of the body `b`,
  which is projected on the body `b̄`, are weighted by `k_b̄ / (k_b + k_b̄)`, where `k` is
  the stiffness of a body (the estimate of the penalty stiffness, evaluated per body).
  Both weights of a contact zone sum up to one, i.e. the formulation stays consistent.
  For equal bodies, both weights are one half; for a rubber body on a stiff body, the
  stress of the rubber body is used. The unweighted (unbiased) two-pass formulation
  failed to converge for a stiffness ratio of 1000, because the contact pressure of the
  stiff body requires a much higher Nitsche parameter. The Nitsche parameter is
  estimated from the softest body.
- Added the regularization of the contact pressure of `SolidBodyContactNitsche` w.r.t.
  the effective penetration `s = λ_N / γ - g`. The contact pressure is exact for
  `s ≥ δ`, the regularization causes a penetration of at most `4 δ / 27` within the
  transition zone, independent of the contact pressure.
- Added the argument `max_gap` to `ContactSurfacePair.kinematics()`, the maximum
  positive gap of a valid projection (default is 0.0, i.e. unchanged for
  `SolidBodyContact`). A Nitsche contact is also active for small positive gaps if the
  normal stress is compressive.
- Added `ContactSurfacePair.variations_projection()`, which returns the change of the
  natural element coordinates of the projected points and of the outward unit normal
  vectors. It is used by the friction and by the Nitsche contact.
- Added `felupe_contact._nitsche.SurfaceStress`, which evaluates the stress tensors of a
  solid body at the integration points of its contact surface and their change w.r.t.
  the displacements of the points of the cells of the faces, and the helpers
  `volume_cells()` (the cells of a mesh which contain the cells of a boundary region)
  and `select()`.
- Added the tests `tests/test_nitsche.py` (incl. the tangent of the force vector for
  all boundary regions with `SolidBody` and `SolidBodyNearlyIncompressible`, the patch
  test for one and for two materials with a stiffness ratio of 1000, the weights, a
  material model with state variables and a cutback) and Ex. 8, a rubber block which is
  pressed on a stiff base with two solid bodies.

### Fixed
- Fixed the connected bodies of a body with an enclosed cavity. The bodies were
  identified by the cells of the boundary regions, which are disconnected for a wall
  of more than two cells, i.e. the inner surface was detected as another body which
  penetrates the outer surface, even in the undeformed configuration. Now, the cells
  of the solid bodies are used if solid bodies are given.
- Fixed the contact search for self-contact if all faces on the outline of a mesh are
  used. The neighbours of a face occupied the candidates of the tree-query, i.e. some
  contact partners were not found. Now, the maximum number of neighbours is added to
  the number of candidates for self-contact.
- Fixed the package metadata: the project is named `felupe-contact` and its version is
  read from `felupe_contact.__about__`. Both were set to `felupe`, which made the build
  fail with `ModuleNotFoundError: No module named 'felupe'`.
- Fixed the option `:context:` of the `pyvista-plot` directive of the axisymmetric
  example in the docstring of `SolidBodyContact`, which takes no value (`close-figs` is
  a value of the `plot` directive of matplotlib). It raised an error in the build of the
  documentation.
- Fixed the `read-the-docs` dependency group: it now installs the documentation
  requirements and the optional dependencies of this package, `felupe-contact[all,docs]`.
  The group referenced `felupe[all,docs,examples]`, which was a self-reference to this
  package due to its wrong name. The examples of FElupe and their requirements (e.g.
  PyTorch from its own package index) are not required.
- Fixed the reference to `SolidBodyContact` in its docstring-example, which was given as
  `fem.SolidBodyContact`. The class is imported from `felupe_contact`.
- Fixed the *See Also*-reference of `SolidBodyContact`, which pointed to the
  non-existent `felupe.ContactRigidPlane`. It now refers to `felupe.MultiPointContact`
  and to `felupe.ContactRigidPlane` and `felupe.ContactRigidCylinder`, which are
  available since FElupe v11.0.0 and v11.2.0.
- Fixed the faces of the primary surface, which are assigned to the integration points
  of the secondary surface for the next evaluation (hysteresis of the contact search),
  if no integration point is in contact. They were kept from the evaluation before,
  i.e. after a complete loss of contact, an integration point got the face of an
  outdated evaluation. Now, no face is assigned, like for the integration points which
  are not in contact if other integration points are in contact.

### Changed
- Changed `copy_statevars()`, which copies any dict of arrays or lists of arrays, e.g.
  the state variables of the material models of a Nitsche contact.
- Changed the assembly of `SolidBodyContact` into the helpers `_assemble_vector()`,
  `_assemble_matrix()` and `_diagonal()`, which are shared with
  `SolidBodyContactNitsche` (the results are unchanged).
- Changed the contact search of `ContactSurfacePair.kinematics()`, which is about ten
  times faster for contact surfaces with all faces on the outline of the meshes and
  about 60 times faster for many bodies (the results are unchanged):
  - The bounds of the faces of the primary surface (a sphere, the heights w.r.t. the
    tangent plane at the center, a bounding box and the maximum angle of the normal
    vectors) are evaluated on the extended range of the natural element coordinates of
    valid projections. Cheap and rigorous criteria discard the candidate face-pairs of
    the tree-query which are not able to touch before the expensive closest-point
    projection. These criteria are evaluated one after another on the remaining
    face-pairs.
  - Only the integration points within the bounding box of a body (enlarged by the
    maximum distance) are queried for the faces of this body, i.e. the effort doesn't
    grow quadratically with the number of bodies. The tree-query is limited by the
    maximum reach of the faces.
  - The face of the previous evaluation is always a candidate (hysteresis), even if it
    is not among the nearest faces of the tree-query.
  - The neighbours of the faces are only checked for self-contact. Without
    self-contact, they belong to the same body and are skipped by the tree-query.
- Changed `closest_point_projection()` to iterate only the pairs which are not
  converged yet (active set) and to evaluate the shape functions and their derivatives
  at once by `BatchedElement.evaluate()`.
- Changed the assembly of the contact stiffness matrix: the stiffness matrix of a
  face-pair is evaluated by a single contraction of the stacked variations of both
  faces instead of three blocks and the matrices of all surface pairs are summed up at
  once by the sparsity pattern of FElupe (`felupe.assembly._sparsity.SparsityPattern`).
  The sparsity pattern is re-used as long as the face-pairs in contact don't change.
  The force vector is summed up by `numpy.bincount()`. Both replace the conversion of
  all values from the COO- to the CSR-format and the addition of sparse matrices,
  which is about three times faster.
- Changed `connected_bodies()` to support a list of point-connectivities, e.g. of
  cells with different numbers of points.
- Changed Ex. 2 and Ex. 3 to create the contact on the solid bodies and the README to
  the solid-body mode of `SolidBodyContact`.
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
- Changed `ContactSurfacePair.kinematics()` to return also the deformed coordinates of
  the integration points in contact and the natural element coordinates of their
  projections, which are required by the friction.
- Changed `ContactSurfacePair.kinematics()` to return C-contiguous arrays, see the new
  helper `contiguous()`. The arrays of the face-pairs in contact are selected from the
  candidate face-pairs by advanced indexing of their last axis, which results in arrays
  that are not contiguous in memory. This slowed down the contractions by
  `numpy.einsum()` of the assembly by up to a factor of 50 for a single contraction. The
  assembly of the stiffness matrix of the frictionless contact is about 1.3 to 2 times
  faster, the one of the frictional contact about 1.2 to 1.6 times (the results are
  unchanged, bit for bit).
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
- Changed the minimum version of Python to 3.11, like FElupe (v12). The tests run on
  Python 3.11 and 3.14.
- Changed the tests of tox to run in parallel by `pytest-xdist`. The environment `core`
  runs the import-test with the required dependencies only, i.e. without the optional
  dependencies of FElupe.
- Changed the license-headers of the source files to SPDX license identifiers, like in
  FElupe. The headers stated that the files are part of FElupe.
- Changed the version to `0.1.0-dev`: the version of an unreleased development state is
  marked by the suffix `-dev`, like in FElupe.
- Changed the development status of the package metadata from *Production/Stable* to
  *Alpha* and added `numpy` and `scipy` to the required dependencies, which are imported
  directly by this package.
- Changed the titles of Ex. 2 (all faces) and Ex. 3 (sub-meshes), which were equal to
  the title of Ex. 1, and moved all imports of the examples to their first cell.
- Changed the examples to print the contact object, which shows the estimated penalty
  stiffness, the smoothing and the number of integration points in contact. The text of
  the examples referred to the penalty stiffness, but only the number of integration
  points in contact was printed.

### Removed
- Removed the argument `order` of `BatchedElement.monomials()`. The partial
  derivatives of the shape functions are evaluated by the stacked coefficients of the
  derivatives instead of differentiating the monomials on each evaluation.
- Removed `ContactSurfacePair._assemble()` and `ContactSurfacePair._indices()` in favour
  of `ContactSurfacePair.dofs()` and the helpers `sparse_vector()` and
  `sparse_matrix()`.
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
- Removed the workflow `publish-to-pypi.yml`, which built the package on each push and
  published it on PyPI for each tag. The package is published on PyPI later.
- Removed the configuration of isort in `pyproject.toml`, which is replaced by ruff.
