.. _how-to-surfaces:

Define the Contact Surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A contact surface is a field container with a displacement field, which is created on a boundary region of FElupe, e.g. a :class:`~felupe.RegionHexahedronBoundary` or a :class:`~felupe.RegionQuadBoundary`. A :class:`~felupe_contact.SolidBodyContact` takes two contact surfaces: the integration points of the *secondary* surface are projected on the faces of the *primary* surface. The contact surfaces are created on solid bodies by the contact, or they are given as field containers on boundary regions.

One mesh for all bodies
-----------------------
The contact couples the degrees of freedom of the bodies which are in contact. Hence, the boundary regions have to be created on a mesh with the same points as the mesh of the solid bodies. Multiple bodies are meshed individually and are stacked to a single mesh. The cell types of both boundary regions have to be equal.

..  pyvista-plot::
    :context:

    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> from felupe_contact import SolidBodyContact
    >>>
    >>> bottom = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(4, 4, 3))
    >>> top = fem.Cube(a=(0.15, 0.15, 1.02), b=(0.85, 0.85, 1.62), n=(3, 3, 3))
    >>> mesh = fem.MeshContainer([bottom, top], merge=True).stack()
    >>>
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

The bodies of a mesh are identified by the connectivity of its cells: two faces belong to the same body if they are connected by a path of cells. Faces of the same body are never in contact, unless ``self_contact=True`` is given.

All faces on the outline
------------------------
The contact is created on the solid body. The boundary region of the region of the solid body is created with all faces on the outline of the mesh, i.e. the faces of the contact zone don't have to be known in advance. This contact surface is used as secondary and as primary surface. The penalty stiffness is estimated from the stiffness matrix of the solid body.

..  pyvista-plot::
    :context:

    >>> contact = SolidBodyContact(solid)
    >>> type(contact.field.region).__name__
    'RegionHexahedronBoundary'

The boundary region is looked up for the type of the region of the displacement field of the solid body. The displacement field of the contact surface is of the same type as the displacement field of the solid body and its values are linked to the solid body.

..  list-table:: The regions and boundary regions of FElupe
    :header-rows: 1
    :widths: 50 50

    *   - Region
        - Boundary region
    *   - :class:`~felupe.RegionQuad`
        - :class:`~felupe.RegionQuadBoundary`
    *   - :class:`~felupe.RegionQuadraticQuad`
        - :class:`~felupe.RegionQuadraticQuadBoundary`
    *   - :class:`~felupe.RegionBiQuadraticQuad`
        - :class:`~felupe.RegionBiQuadraticQuadBoundary`
    *   - :class:`~felupe.RegionHexahedron`
        - :class:`~felupe.RegionHexahedronBoundary`
    *   - :class:`~felupe.RegionQuadraticHexahedron`
        - :class:`~felupe.RegionQuadraticHexahedronBoundary`
    *   - :class:`~felupe.RegionTriQuadraticHexahedron`
        - :class:`~felupe.RegionTriQuadraticHexahedronBoundary`

The contact search discards all face-pairs which are not able to touch each other:

* faces which share at least one point (neighbours),
* faces which belong to the same body (unless ``self_contact=True``) and
* faces whose outward unit normal vectors are not opposed, e.g. faces which are located around a corner (see the argument ``facing``).

Both faces of a contact zone are used as secondary surfaces. Hence, twice the number of integration points are in contact, compared to the masked surfaces from below. This acts like a two-pass contact without its scale factors, i.e. ``two_pass=True`` is redundant in this case.

..  pyvista-plot::
    :context:

    >>> boundaries = {
    ...     "fixed": fem.Boundary(field[0], fz=0.0),
    ...     "clamped": fem.Boundary(field[0], fz=mesh.z.max(), skip=(0, 0, 1)),
    ...     "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0)),
    ... }
    >>> step = fem.Step(
    ...     items=[solid, contact],
    ...     ramp={boundaries["move"]: fem.math.linsteps([0, -0.3], num=3)},
    ...     boundaries=boundaries,
    ... )
    >>> job = fem.Job(steps=[step]).evaluate(verbose=0)
    >>>
    >>> contact.results.npoints_in_contact
    32

..  hint::

    The argument ``max_distance`` limits the maximum detectable penetration. It must be smaller than the distance between two opposed faces which are not able to touch, e.g. the top and the bottom face of a thin body, because such a face-pair is detected as a deeply penetrated one otherwise. The default value is five times the characteristic size of the faces. Decrease it for a coarse mesh of a small (thin) body.

Faces in the region of interest
-------------------------------
The faces of a boundary region are selected by masks of the points of the mesh. Here, the bottom faces of the upper block are the secondary surface and the top faces of the lower block are the primary surface.

..  pyvista-plot::
    :context:

    >>> contact = SolidBodyContact(
    ...     solid,
    ...     mask=np.logical_and(mesh.z > 1.0, mesh.z < 1.1),
    ...     mask_primary=np.isclose(mesh.z, 1.0),
    ... )

The choice of the secondary surface matters: the contact tractions are evaluated at the integration points of the secondary surface. Use the softer and finer meshed surface as the secondary surface, e.g. the rubber surface of a rubber-to-metal contact. Alternatively, a two-pass contact evaluates the contact twice with exchanged roles of the surfaces, each with half of the contact potential. This removes the bias of the choice of the secondary surface at the expense of a doubled evaluation time.

..  pyvista-plot::
    :context:

    >>> contact = SolidBodyContact(
    ...     solid,
    ...     mask=np.logical_and(mesh.z > 1.0, mesh.z < 1.1),
    ...     mask_primary=np.isclose(mesh.z, 1.0),
    ...     two_pass=True,
    ... )

Field containers on boundary regions
------------------------------------
Instead of solid bodies, field containers on boundary regions are accepted as contact surfaces. Then, the items for the estimation of the penalty stiffness have to be given. The bodies are identified by the cells of the boundary regions.

..  pyvista-plot::
    :context:

    >>> mask = np.logical_and(mesh.z > 1.0, mesh.z < 1.1)
    >>> secondary = fem.FieldContainer(
    ...     [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=mask), dim=3)]
    ... )
    >>> primary = fem.FieldContainer(
    ...     [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=np.isclose(mesh.z, 1.0)), dim=3)]
    ... )
    >>> contact = SolidBodyContact(secondary, primary, items=[solid])

..  hint::

    Prefer the solid bodies for contact surfaces with all faces on the outline of a mesh. The cells of a boundary region are not sufficient to identify a body with an enclosed cavity and a wall of more than two cells: its inner surface is detected as another body, which penetrates the outer surface.

Multiple solid bodies
---------------------
The bodies may also be given as individual solid bodies, e.g. with different materials. Their meshes have to share the same points, which is ensured by a :class:`~felupe.MeshContainer` with ``merge=True``. A contact surface is created for each solid body and both solid bodies are used for the estimation of the penalty stiffness.

..  pyvista-plot::
    :context:

    >>> container = fem.MeshContainer([bottom, top], merge=True)
    >>>
    >>> regions = [fem.RegionHexahedron(m) for m in container]
    >>> fields = [fem.FieldContainer([fem.Field(r, dim=3)]) for r in regions]
    >>> field = fem.field.merge(fields)
    >>>
    >>> solids = [
    ...     fem.SolidBody(umat=fem.NeoHooke(mu=5.0, bulk=250.0), field=fields[0]),
    ...     fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=fields[1]),
    ... ]
    >>> contact = SolidBodyContact(solids[1], solids[0])

Here, the softer upper block is the secondary surface. The boundary conditions are created on the merged field container and the contact is added to the items of the step, together with all solid bodies.

..  pyvista-plot::
    :context:

    >>> boundaries = {
    ...     "fixed": fem.Boundary(field[0], fz=0.0),
    ...     "clamped": fem.Boundary(field[0], fz=1.62, skip=(0, 0, 1)),
    ...     "move": fem.Boundary(field[0], fz=1.62, skip=(1, 1, 0)),
    ... }
    >>> step = fem.Step(
    ...     items=[*solids, contact],
    ...     ramp={boundaries["move"]: fem.math.linsteps([0, -0.3], num=3)},
    ...     boundaries=boundaries,
    ... )
    >>> job = fem.Job(steps=[step]).evaluate(verbose=0)

..  pyvista-plot::
    :context:
    :force_static:

    >>> solids[1].plot(
    ...     "Principal Values of Cauchy Stress",
    ...     plotter=solids[0].plot("Principal Values of Cauchy Stress", opacity=0.3),
    ... ).show()
