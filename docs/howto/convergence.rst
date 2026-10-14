.. _how-to-convergence:

Penalty Stiffness and Convergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contact constraints are enforced by a regularized penalty law. The penalty stiffness controls the penetration of the contact surfaces and it affects the convergence of the Newton-Raphson method of FElupe. This guide uses a model of two blocks, which is created by a function.

..  pyvista-plot::
    :context:

    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> from felupe_contact import SolidBodyContact
    >>>
    >>> def solve(move=-0.3, num=3, plugins=None, **kwargs):
    ...     "Press a rubber block on a bigger block and return the contact."
    ...
    ...     bottom = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(4, 4, 3))
    ...     top = fem.Cube(a=(0.15, 0.15, 1.02), b=(0.85, 0.85, 1.62), n=(3, 3, 3))
    ...     mesh = fem.MeshContainer([bottom, top], merge=True).stack()
    ...
    ...     region = fem.RegionHexahedron(mesh)
    ...     field = fem.FieldContainer([fem.Field(region, dim=3)])
    ...     solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)
    ...
    ...     mask = np.logical_and(mesh.z > 1.0, mesh.z < 1.1)
    ...     secondary = fem.FieldContainer(
    ...         [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=mask), dim=3)]
    ...     )
    ...     primary = fem.FieldContainer(
    ...         [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=np.isclose(mesh.z, 1.0)), dim=3)]
    ...     )
    ...     contact = SolidBodyContact(secondary, primary, items=[solid], **kwargs)
    ...
    ...     boundaries = {
    ...         "fixed": fem.Boundary(field[0], fz=0.0),
    ...         "clamped": fem.Boundary(field[0], fz=mesh.z.max(), skip=(0, 0, 1)),
    ...         "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0)),
    ...     }
    ...     step = fem.Step(
    ...         items=[solid, contact],
    ...         ramp={boundaries["move"]: fem.math.linsteps([0, move], num=num)},
    ...         boundaries=boundaries,
    ...     )
    ...     fem.Job(steps=[step], plugins=plugins or []).evaluate(verbose=0)
    ...
    ...     return contact

Penalty stiffness
-----------------
If no penalty stiffness is given, it is estimated from the mean of the diagonal entries of the stiffness matrices of the given ``items``, evaluated on the degrees of freedom of the points of a contact surface, and the mean area per point of this surface. The softer of both contact surfaces is decisive. This estimate scales with the stiffness of the material and the size of the faces, i.e. it requires no manual tuning, neither for rubber-to-rubber nor for rubber-to-metal contacts. The estimate is multiplied by ``penalty_scale`` (default is 10.0).

A higher penalty stiffness reduces the penetration of the contact surfaces, which is the negative gap of the integration points in contact.

..  pyvista-plot::
    :context:

    >>> for penalty_scale in [1.0, 10.0, 100.0]:
    ...     contact = solve(penalty_scale=penalty_scale)
    ...     penetration = -np.concatenate(contact.results.gap).min()
    ...     print(f"scale={penalty_scale:5.1f}, penetration={penetration:.3f}")
    scale=  1.0, penetration=0.033
    scale= 10.0, penetration=0.005
    scale=100.0, penetration=0.002

Increase ``penalty_scale`` to reduce the penetration and decrease it if the Newton-Raphson method doesn't converge. Alternatively, the penalty stiffness may be given as contact traction per unit penetration. Then, no items are required.

..  pyvista-plot::
    :context:

    >>> contact = solve(penalty=650.0)
    >>> contact.penalty
    650.0

..  note::

    The estimate of the penalty stiffness assumes an equal area per point of a contact surface. This is not the case for a boundary region of a mesh with quadratic cells, where the diagonal entries of the stiffness matrix differ between the points on the edges and the midpoints of a face. Hence, the estimate is stiffer for quadratic cells.

Regularized penalty law
-----------------------
The contact pressure is regularized by a smooth transition, i.e. both the contact pressure and its derivative w.r.t. the gap are continuous. This removes the jump of the tangent stiffness at the activation of a contact and hence improves the rate of convergence of the Newton-Raphson method. The length of the transition zone is given by ``smoothing`` (default is one percent of the characteristic size of the faces). A value of zero deactivates the regularization. The pressure law is evaluated by :meth:`SolidBodyContact.pressure() <felupe_contact.SolidBodyContact.pressure>`.

..  pyvista-plot::
    :context:

    >>> contact = solve(smoothing=0.0)
    >>> contact.smoothing
    0.0

The geometric part of the contact stiffness matrix, i.e. the part which is proportional to the contact pressure and which contains the change of the normal vectors and of the projected points, is required for a quadratic rate of convergence. It may be deactivated by ``geometric_stiffness=False``.

Large increments
----------------
A single, large increment fails to converge. Instead of decreasing the load increments of all substeps, add a :class:`~felupe.CutbackPlugin` to the job. It subdivides only the substeps which fail to converge. The contact provides the methods :meth:`~felupe_contact.SolidBodyContact.checkpoint` and :meth:`~felupe_contact.SolidBodyContact.restore`, i.e. the state of the contact search is restored to the last converged increment on a cutback.

..  pyvista-plot::
    :context:

    >>> cutback = fem.CutbackPlugin()
    >>> contact = solve(move=-0.45, num=1, plugins=[cutback])
    >>>
    >>> cutback.cutbacks
    [0, 1]

The second substep (the first substep is the initial state with zero displacements) is solved in two increments.

..  pyvista-plot::
    :context:

    >>> cutback.load_factors
    [[1.0], [0.5, 1.0]]

A :class:`~felupe.LinesearchPlugin` scales the Newton increments by a step length, which is halved until the norm of the residuals decreases sufficiently. It may be combined with a cutback.

..  pyvista-plot::
    :context:

    >>> linesearch = fem.LinesearchPlugin()
    >>> contact = solve(move=-0.45, num=1, plugins=[fem.CutbackPlugin(), linesearch])

..  hint::

    The contact search uses a tree-query for the nearest faces of the primary surface. The query is threaded if the job is evaluated with ``parallel=True``, e.g. ``fem.Job(steps).evaluate(parallel=True)``, which also activates the threaded assembly of the solid bodies.
