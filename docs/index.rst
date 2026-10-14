felupe-contact documentation
============================

felupe-contact is an extension package for `FElupe <https://felupe.readthedocs.io>`_ with a contact between the surfaces of deformable solid bodies, e.g. rubber components, which undergo large deformations and large relative displacements.

The contact is an item of FElupe, i.e. it is added to the items of a :class:`~felupe.Step` like a :class:`~felupe.SolidBody`. All other building blocks, like the mesh, the regions, the fields, the solid bodies, the boundary conditions, the jobs and their plugins, are taken from FElupe. Hence, an existing FElupe-script is extended by a contact with only a few lines of code.

.. grid::

   .. grid-item-card:: 🏃 Getting Started
      :link: getting-started
      :link-type: ref

      New to felupe-contact? This guide adds a contact to a FElupe-script, step by step. Requires an understanding of the key concepts of FElupe.

   .. grid-item-card:: 📖 API reference
      :link: felupe-contact-api
      :link-type: ref

      The reference guide contains a detailed description of the API of felupe-contact, including the formulation of the contact. It describes how the methods work and which parameters can be used.

.. grid::

   .. grid-item-card:: ☎ How-To
      :link: how-to
      :link-type: ref

      Step-by-step guides for specific tasks, like the definition of the contact surfaces, two-dimensional problems or the convergence of the Newton-Raphson method.

   .. grid-item-card:: 📚 Examples
      :link: examples
      :link-type: ref

      A gallery of examples.

.. admonition:: Highlights
   :class: admonition

   + a frictionless :class:`contact between the surfaces of deformable solid bodies <felupe_contact.SolidBodyContact>` (surface-to-surface, penalty-based)

   + an item of FElupe, i.e. it is used in a :class:`~felupe.Step`, a :class:`~felupe.Job` and a :class:`~felupe.CharacteristicCurve`, and with the :class:`~felupe.CutbackPlugin` and the :class:`~felupe.LinesearchPlugin`

   + three-dimensional, :class:`plane strain <felupe.FieldPlaneStrain>` and :class:`axisymmetric <felupe.FieldAxisymmetric>` problems

   + all boundary regions of FElupe, i.e. faces with linear or quadratic shape functions

   + contact surfaces given by all faces on the outline of a mesh, i.e. no manual assignment of the faces in the contact zone is required

   + an estimated penalty stiffness, which requires no manual tuning for rubber-to-rubber or for rubber-to-metal contacts

.. warning::

   felupe-contact is in an early stage of development. It is not available on PyPI yet and its API may change without a deprecation.

Contact in FElupe and in felupe-contact
---------------------------------------

FElupe contains contact formulations with rigid surfaces and a contact by a third medium. felupe-contact adds a contact between the surfaces of deformable solid bodies.

..  list-table:: Contact formulations of FElupe and felupe-contact
    :header-rows: 1
    :widths: 30 20 30 20

    *   - Class
        - Package
        - Contact partner
        - Friction
    *   - :class:`~felupe.MultiPointContact`
        - FElupe
        - rigid wall (point-to-wall)
        - no
    *   - :class:`~felupe.ContactRigidPlane`
        - FElupe
        - rigid plane (node-to-surface)
        - Coulomb
    *   - :class:`~felupe.ContactRigidCylinder`
        - FElupe
        - rigid cylinder (node-to-surface)
        - Coulomb
    *   - :class:`~felupe.ThirdMediumContactMixed`
        - FElupe
        - a third (soft) medium which fills the gap between the bodies
        - no
    *   - :class:`~felupe_contact.SolidBodyContact`
        - felupe-contact
        - deformable solid bodies (surface-to-surface)
        - no

Installation
------------
Install Python, open a terminal and run

.. code-block:: shell

   pip install git+https://github.com/adtzlr/felupe-contact.git@main

or clone the repository and install the package in editable mode.

.. code-block:: shell

   git clone https://github.com/adtzlr/felupe-contact.git
   cd felupe-contact
   pip install --editable .

felupe-contact has minimal requirements, all available at PyPI supporting all platforms.

* `felupe <https://github.com/adtzlr/felupe>`_ for the finite element infrastructure
* `numpy <https://github.com/numpy/numpy>`_ for array operations
* `scipy <https://github.com/scipy/scipy>`_ for sparse matrices and the contact search

The optional dependencies of FElupe, e.g. `pyvista <https://github.com/pyvista/pyvista>`_ for interactive visualizations and `matplotlib <https://github.com/matplotlib/matplotlib>`_ for plotting graphs, are installed by replacing the last line.

.. code-block:: shell

   pip install --editable ".[all]"

felupe-contact is tested with the latest release and with the development version of FElupe.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   start
   examples/index
   howto
   felupe_contact

License
-------

felupe-contact - Contact Extension-package for FElupe (C) 2026 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see `<https://www.gnu.org/licenses/>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
