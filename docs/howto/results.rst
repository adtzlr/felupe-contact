.. _how-to-results:

Contact Results
~~~~~~~~~~~~~~~

The results of a :class:`~felupe_contact.SolidBodyContact` are stored in its attribute ``results``, a :class:`felupe.mechanics.Results` object, like for the solid bodies of FElupe. A rubber block is pressed on a bigger block and the force-displacement curve of the moved boundary is tracked by a :class:`~felupe.CharacteristicCurve`.

..  plot::
    :context: reset
    :nofigs:

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
    >>>
    >>> mask = np.logical_and(mesh.z > 1.0, mesh.z < 1.1)
    >>> secondary = fem.FieldContainer(
    ...     [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=mask), dim=3)]
    ... )
    >>> primary = fem.FieldContainer(
    ...     [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=np.isclose(mesh.z, 1.0)), dim=3)]
    ... )
    >>> contact = SolidBodyContact(secondary, primary, items=[solid])
    >>>
    >>> boundaries = {
    ...     "fixed": fem.Boundary(field[0], fz=0.0),
    ...     "clamped": fem.Boundary(field[0], fz=mesh.z.max(), skip=(0, 0, 1)),
    ...     "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0)),
    ... }
    >>> step = fem.Step(
    ...     items=[solid, contact],
    ...     ramp={boundaries["move"]: fem.math.linsteps([0, -0.3], num=6)},
    ...     boundaries=boundaries,
    ... )
    >>> job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
    >>> job = job.evaluate(verbose=0)

Gap and contact pressure
------------------------
The results of the last evaluation are available at the integration points of the secondary surface which are in contact.

..  plot::
    :context:
    :nofigs:

    >>> contact.results.npoints_in_contact
    16

The gaps and the contact pressures are given as lists with an array for each surface pair, i.e. a list with one array (or two arrays for ``two_pass=True``). An entry is None if no integration point of the surface pair is in contact. A negative gap is a penetration.

..  plot::
    :context:
    :nofigs:

    >>> gap = contact.results.gap[0]
    >>> pressure = contact.results.pressure[0]
    >>>
    >>> gap.shape, pressure.shape
    ((16,), (16,))

    >>> bool(np.all(gap < 0) and np.all(pressure > 0))
    True

The contact pressure for given gaps is evaluated by the penalty law :meth:`~felupe_contact.SolidBodyContact.pressure`, which also returns the derivative of the contact pressure w.r.t. the gap.

..  plot::
    :context:
    :nofigs:

    >>> p, dpdg = contact.pressure(gap)
    >>> bool(np.allclose(p, pressure))
    True

Contact forces
--------------
The assembled contact force vector is stored in ``results.force``. It is a sparse vector with the internal forces of the contact at the degrees of freedom of the field, i.e. it has the same sign convention as the force vector of a solid body. The contact forces on the upper block are equal to the reaction force of the moved boundary, because the upper block is supported only by the contact.

..  plot::
    :context:
    :nofigs:

    >>> forces = contact.results.force.toarray().reshape(-1, 3)
    >>> force_on_upper_block = forces[mesh.z > 1.01].sum(axis=0)
    >>>
    >>> round(float(force_on_upper_block[2]), 3)
    -0.997

    >>> reaction_force = job.y[-1]
    >>> bool(np.allclose(force_on_upper_block, reaction_force))
    True

Force-displacement curve
------------------------
The force-displacement curve is plotted by the :class:`~felupe.CharacteristicCurve`. The force is zero until the gap between both blocks is closed.

..  plot::
    :context: close-figs

    >>> fig, ax = job.plot(
    ...     xaxis=2,
    ...     yaxis=2,
    ...     xscale=-1.0,
    ...     yscale=-1.0,
    ...     xlabel=r"Displacement $-u_z$ $\longrightarrow$",
    ...     ylabel=r"Force $-F_z$ $\longrightarrow$",
    ...     marker="o",
    ... )
