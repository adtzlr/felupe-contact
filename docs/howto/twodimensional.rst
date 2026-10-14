.. _how-to-2d:

Plane Strain and Axisymmetric Contact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contact surfaces of a two-dimensional mesh are created on boundary regions of quads, e.g. a :class:`~felupe.RegionQuadBoundary`. The faces of these boundary regions are the edges of the cells. Like for the solid bodies, the out-of-plane behaviour is defined by the type of the displacement field: a :class:`~felupe.FieldPlaneStrain` or a :class:`~felupe.FieldAxisymmetric` is required. The types of the fields of both contact surfaces have to be equal. For a contact on a solid body, the contact surfaces are created with the type of the displacement field of the solid body.

Plane strain
------------
A rubber block is pressed on a bigger block. The contact is created on the solid body with a plane strain field, i.e. the contact surfaces are created with plane strain fields.

..  pyvista-plot::
    :context:

    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> from felupe_contact import SolidBodyContact
    >>>
    >>> bottom = fem.Rectangle(a=(0, 0), b=(1, 1), n=(9, 5))
    >>> top = fem.Rectangle(a=(0.15, 1.02), b=(0.85, 1.62), n=(7, 5))
    >>> mesh = fem.MeshContainer([bottom, top], merge=True).stack()
    >>>
    >>> region = fem.RegionQuad(mesh)
    >>> field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
    >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)
    >>>
    >>> contact = SolidBodyContact(solid)
    >>> type(contact.field[0]).__name__
    'FieldPlaneStrain'

A (cartesian) :class:`~felupe.Field` is rejected for a boundary region of a two-dimensional mesh, because its out-of-plane behaviour is not defined.

..  code-block:: pycon

    >>> cartesian = fem.FieldContainer([fem.Field(fem.RegionQuadBoundary(mesh), dim=2)])
    >>> SolidBodyContact(cartesian, cartesian, items=[solid])
    Traceback (most recent call last):
    ...
    TypeError: A field of type `FieldPlaneStrain` or `FieldAxisymmetric` is required for a boundary region of a 2d mesh, got `Field`.

The model is completed by the boundary conditions, a step and a job, like for a three-dimensional model.

..  pyvista-plot::
    :context:

    >>> boundaries = {
    ...     "fixed": fem.Boundary(field[0], fy=0.0),
    ...     "clamped": fem.Boundary(field[0], fy=mesh.y.max(), skip=(0, 1)),
    ...     "move": fem.Boundary(field[0], fy=mesh.y.max(), skip=(1, 0)),
    ... }
    >>> step = fem.Step(
    ...     items=[solid, contact],
    ...     ramp={boundaries["move"]: fem.math.linsteps([0, -0.3], num=6)},
    ...     boundaries=boundaries,
    ... )
    >>> job = fem.Job(steps=[step]).evaluate(verbose=0)

..  pyvista-plot::
    :context:
    :force_static:

    >>> solid.plot("Principal Values of Cauchy Stress").show()

Axisymmetric
------------
The rotation axis of an axisymmetric field is the first axis of the mesh, i.e. :math:`(X, Y) \widehat{=} (Z, R)`. A rubber cylinder is pressed on a bigger cylinder, hence both cylinders are stacked along the first axis and their radial coordinates start at the rotation axis :math:`R = 0`.

..  pyvista-plot::
    :context:

    >>> bottom = fem.Rectangle(a=(0, 0), b=(1, 1), n=(5, 9))
    >>> top = fem.Rectangle(a=(1.02, 0), b=(1.62, 0.7), n=(5, 7))
    >>> mesh = fem.MeshContainer([bottom, top], merge=True).stack()
    >>>
    >>> region = fem.RegionQuad(mesh)
    >>> field = fem.FieldContainer([fem.FieldAxisymmetric(region, dim=2)])
    >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)
    >>>
    >>> contact = SolidBodyContact(solid)

The contact potential of an axisymmetric field is integrated on the surface of revolution, i.e. the differential area of a face is the product of its differential length and the circumference :math:`2 \pi R` of the circle at its radial coordinate. The contact is evaluated at the integration points of the faces, i.e. the faces which touch the rotation axis :math:`R = 0` are supported.

..  pyvista-plot::
    :context:

    >>> boundaries = {
    ...     "fixed": fem.Boundary(field[0], fx=0.0),
    ...     "clamped": fem.Boundary(field[0], fx=mesh.x.max(), skip=(1, 0)),
    ...     "move": fem.Boundary(field[0], fx=mesh.x.max(), skip=(0, 1)),
    ... }
    >>> step = fem.Step(
    ...     items=[solid, contact],
    ...     ramp={boundaries["move"]: fem.math.linsteps([0, -0.3], num=6)},
    ...     boundaries=boundaries,
    ... )
    >>> job = fem.Job(steps=[step]).evaluate(verbose=0)

The equivalent (von Mises) Cauchy stress of the axisymmetric solid body is plotted in the plane of the mesh.

..  pyvista-plot::
    :context:
    :force_static:

    >>> solid.plot("Equivalent of Cauchy Stress").show()

..  hint::

    The boundary regions of quadratic quads, i.e. a :class:`~felupe.RegionQuadraticQuadBoundary` or a :class:`~felupe.RegionBiQuadraticQuadBoundary`, are supported too. Their faces are quadratic lines, see Ex. 6 of the :ref:`examples`.
