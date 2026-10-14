.. _getting-started:

Getting Started
===============

This guide adds a contact to a FElupe-script, step by step. A rubber block is pressed on a bigger rubber block. All building blocks of the model, except the contact, are taken from FElupe. If you are new to FElupe, start with its `Beginner's Guide <https://felupe.readthedocs.io/en/stable/tutorial/index.html>`_.

..  list-table:: The building blocks of a model with a contact
    :header-rows: 1
    :widths: 30 25 45

    *   - Building block
        - Package
        - Note
    *   - :class:`~felupe.Mesh`
        - FElupe
        - one mesh for all bodies, e.g. by :meth:`MeshContainer.stack() <felupe.MeshContainer.stack>`
    *   - :class:`~felupe.RegionHexahedron`, :class:`~felupe.Field`, :class:`~felupe.SolidBody`
        - FElupe
        - unchanged
    *   - :class:`~felupe_contact.SolidBodyContact`
        - felupe-contact
        - an item, created on the solid body
    *   - :class:`~felupe.RegionHexahedronBoundary`, :class:`~felupe.Field`
        - FElupe
        - the contact surfaces, created by the contact for the region of the solid body
    *   - :class:`~felupe.Boundary`, :class:`~felupe.Step`, :class:`~felupe.Job`
        - FElupe
        - the contact is added to the items of a step

Mesh
----
The contact couples the degrees of freedom of the bodies which are in contact. Hence, all bodies have to share one mesh and one field container. Both blocks are meshed individually and are stacked to a single mesh. The two blocks are not connected: they are separated by a small gap.

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
    >>> mesh.plot().show()

Solid body
----------
The region, the displacement field and the solid body are created as usual. Both blocks are made of the same rubber-like material.

..  pyvista-plot::
    :context:

    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

Contact
-------
The contact is created on the solid body. Its contact surfaces are field containers with a displacement field on the boundary region of the region of the solid body, here a :class:`~felupe.RegionHexahedronBoundary`, which are created by the contact. The integration points of the *secondary* surface are projected on the faces of the *primary* surface. Here, only the faces in the region of interest are used, selected by masks of the points of the mesh: the bottom faces of the upper block are the secondary surface and the top faces of the lower block are the primary surface.

..  pyvista-plot::
    :context:

    >>> contact = SolidBodyContact(
    ...     solid,
    ...     mask=np.logical_and(mesh.z > 1.0, mesh.z < 1.1),
    ...     mask_primary=np.isclose(mesh.z, 1.0),
    ... )

.. hint::

   The masks are optional. Without masks, all faces on the outline of the mesh are contact surfaces, i.e. ``SolidBodyContact(solid)``, see :ref:`how-to-surfaces`.

The penalty stiffness of the contact is estimated from the stiffness matrix of the solid body on the first assembly. The contact is an item of FElupe, see :class:`felupe.mechanics.Item`.

..  pyvista-plot::
    :context:

    >>> isinstance(contact, fem.mechanics.Item)
    True

Step and job
------------
The bottom face of the lower block is fixed and the top face of the upper block is moved downwards. The contact is added to the items of the step, like the solid body. The step is evaluated by a job, here by a :class:`~felupe.CharacteristicCurve`, which tracks the reaction force of the moved boundary.

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
    >>> job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
    >>> job = job.evaluate(verbose=0)

Results
-------
The estimated penalty stiffness and the number of integration points of the secondary surface which are in contact are shown by the contact object. Its results are explained in :ref:`how-to-results`.

..  pyvista-plot::
    :context:

    >>> print(contact)
    <felupe SolidBodyContact object>
      Penalty stiffness: ...
      Smoothing: ...
      Integration points in contact: 16

The results of the solid body, e.g. the principal values of the Cauchy stress, are plotted as usual.

..  pyvista-plot::
    :context:
    :force_static:

    >>> solid.plot("Principal Values of Cauchy Stress").show()

Next steps
----------

* :ref:`how-to-surfaces` explains how the contact surfaces are defined, e.g. without masks or by field containers on boundary regions.
* :ref:`how-to-2d` shows a contact for plane strain and axisymmetric problems.
* :ref:`how-to-convergence` shows how the penalty stiffness and the plugins of FElupe improve the convergence.
* The :ref:`examples` contain complete scripts.
