.. _how-to-friction:

Friction
~~~~~~~~

A :class:`~felupe_contact.SolidBodyContact` is frictionless by default. A positive friction coefficient ``friction`` adds Coulomb friction, i.e. the integration points of the secondary surface stick to the primary surface until the tangential contact traction reaches the friction coefficient times the contact pressure. Then, they slide. In this guide, a rubber block is pressed on a stiff base and is moved tangentially afterwards. The lower block is a thousand times stiffer than the rubber block, i.e. the material parameters are given for each cell of the mesh.

..  plot::
    :context: reset
    :nofigs:

    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> from felupe_contact import SolidBodyContact
    >>>
    >>> base = fem.Cube(a=(-1, 0, 0), b=(2, 1, 0.5), n=(7, 3, 2))
    >>> block = fem.Cube(a=(0.15, 0.15, 0.52), b=(0.85, 0.85, 1.12), n=(4, 4, 3))
    >>> mesh = fem.MeshContainer([base, block], merge=True).stack()
    >>>
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>>
    >>> mu = np.where(np.arange(mesh.ncells) < base.ncells, 1000.0, 1.0).reshape(1, -1)
    >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=mu, bulk=5 * mu), field=field)

Coulomb friction
----------------
The contact surfaces are the bottom face of the rubber block (secondary surface) and the top face of the base (primary surface). The friction coefficient is :math:`\mu = 0.3`.

..  plot::
    :context:
    :nofigs:

    >>> mask = np.logical_and(mesh.z > 0.5, mesh.z < 0.6)
    >>> secondary = fem.FieldContainer(
    ...     [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=mask), dim=3)]
    ... )
    >>> primary = fem.FieldContainer(
    ...     [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=np.isclose(mesh.z, 0.5)), dim=3)]
    ... )
    >>> contact = SolidBodyContact(secondary, primary, items=[solid], friction=0.3)

The bottom face of the base is fixed. The top face of the rubber block is pressed downwards in a first step and it is moved in direction :math:`X` in a second step. The pressed boundary keeps its last value in the second step.

..  plot::
    :context:
    :nofigs:

    >>> top = np.isclose(mesh.z, mesh.z.max())
    >>> boundaries = {
    ...     "fixed": fem.Boundary(field[0], fz=0.0),
    ...     "x": fem.Boundary(field[0], mask=top, skip=(0, 1, 1)),
    ...     "y": fem.Boundary(field[0], mask=top, skip=(1, 0, 1)),
    ...     "z": fem.Boundary(field[0], mask=top, skip=(1, 1, 0)),
    ... }
    >>> press = fem.Step(
    ...     items=[solid, contact],
    ...     ramp={boundaries["z"]: fem.math.linsteps([0, -0.1], num=2)},
    ...     boundaries=boundaries,
    ... )
    >>> move = fem.Step(
    ...     items=[solid, contact],
    ...     ramp={boundaries["x"]: fem.math.linsteps([0, 0.3], num=6)},
    ...     boundaries=boundaries,
    ... )
    >>> job = fem.CharacteristicCurve(steps=[press, move], boundary=boundaries["x"])
    >>> job = job.evaluate(verbose=0)

The tangential force of the moved boundary increases until the whole contact zone slides (gross sliding). Then, it is equal to the friction coefficient times the normal force.

..  plot::
    :context: close-figs

    >>> fig, ax = job.plot(
    ...     xaxis=0,
    ...     yaxis=0,
    ...     xlabel=r"Displacement $u_X$ $\longrightarrow$",
    ...     ylabel=r"Force $F_X$ $\longrightarrow$",
    ...     marker="o",
    ... )

..  plot::
    :context:
    :nofigs:

    >>> force = job.y[-1]
    >>> print(f"{force[0] / force[2]:.2f}")
    -0.30

Tangential traction and sliding
-------------------------------
The tangential contact tractions (shear) of the integration points in contact are given in ``results.shear``. These are the tractions which act on the secondary surface, i.e. they are opposed to the movement of the rubber block. The mask of the sliding integration points is stored in ``results.slip``. Both are lists with an entry for each surface pair, like the gaps and the contact pressures, see :ref:`how-to-results`. An entry is None for a frictionless contact.

..  plot::
    :context:
    :nofigs:

    >>> print(contact)
    <felupe SolidBodyContact object>
      Penalty stiffness: ...
      Smoothing: ...
      Friction coefficient: 0.3
      Integration points in contact: 36
      Integration points sliding: 36

    >>> shear = contact.results.shear[0]
    >>> pressure = contact.results.pressure[0]
    >>> shear.shape
    (3, 36)

The tangential tractions of the sliding integration points are located on the Coulomb limit.

..  plot::
    :context:
    :nofigs:

    >>> bool(np.allclose(np.linalg.norm(shear, axis=0), 0.3 * pressure))
    True

Tangential penalty stiffness
----------------------------
A sticking integration point is held by a tangential penalty stiffness, i.e. it moves by a small elastic slip before it starts to slide. By default, the tangential penalty stiffness is proportional to the contact pressure, :math:`\epsilon_t = \mu\ p / \gamma`, i.e. an integration point starts to slide at an elastic slip of :math:`\gamma`, independent of the contact pressure. The allowable elastic slip ``elastic_slip`` is one percent of the characteristic size of the faces of the contact surfaces by default.

..  plot::
    :context:
    :nofigs:

    >>> print(f"{contact.elastic_slip:.4f}")
    0.0023

Decrease the elastic slip to reduce the movement of the sticking integration points and increase it if the Newton-Raphson method does not converge. Alternatively, a constant tangential penalty stiffness is given by ``penalty_tangential``. An infinite friction coefficient ``friction=np.inf`` prevents sliding (rough contact).

..  plot::
    :context:
    :nofigs:

    >>> contact = SolidBodyContact(
    ...     secondary, primary, items=[solid], friction=0.3, elastic_slip=0.001
    ... )
    >>> contact = SolidBodyContact(
    ...     secondary, primary, items=[solid], friction=0.3, penalty_tangential=100.0
    ... )
    >>> contact = SolidBodyContact(secondary, primary, items=[solid], friction=np.inf)

State variables and cutbacks
----------------------------
A frictional contact is path-dependent. The anchors (stick points) of the integration points are the state variables of the contact. They are stored in ``results.statevars`` and they are only updated after a converged increment, like the state variables of the solid bodies of FElupe. The state variables are restored on a cutback of the :class:`~felupe.CutbackPlugin`, see :meth:`~felupe_contact.SolidBodyContact.checkpoint`.

..  hint::

    The transition from sticking to sliding of the whole contact zone (gross sliding) is the most demanding part of a frictional contact for the Newton-Raphson method. Use smaller increments there or add a :class:`~felupe.CutbackPlugin`, e.g. ``fem.CharacteristicCurve(steps, boundary, plugins=[fem.CutbackPlugin()])``.

The consistent stiffness matrix of the friction is not symmetric. A symmetric stiffness matrix is obtained by ``symmetric=True``, e.g. for a solver for symmetric matrices. This degrades the rate of convergence of the Newton-Raphson method.
