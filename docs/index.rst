felupe-contact documentation
============================

felupe-contact is an extension package for `FElupe <https://felupe.readthedocs.io>`_ with contact formulations between the surfaces of (hyperelastic) solid bodies, which undergo large deformations and large relative displacements.

.. grid::

   .. grid-item-card:: 📖 API reference
      :link: felupe-contact-api
      :link-type: ref

      The reference guide contains a detailed description of the API of felupe-contact. It describes how the methods work and which parameters can be used. Requires an understanding of the key concepts of FElupe.

   .. grid-item-card:: 📚 Examples
      :link: examples
      :link-type: ref

      A gallery of examples.

.. admonition:: Highlights
   :class: admonition

   + a frictionless :class:`contact between the surfaces of solid bodies <felupe_contact.SolidBodyContact>` (surface-to-surface, penalty-based)

   + three-dimensional, plane strain and axisymmetric problems

   + all boundary regions of FElupe, i.e. faces with linear or quadratic shape functions

   + an item of FElupe, i.e. it is used in a :class:`~felupe.Step` and in a :class:`~felupe.Job` like a :class:`~felupe.SolidBody`

.. warning::

   felupe-contact is in an early stage of development. It is not available on PyPI yet and its API may change without a deprecation.

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

The optional dependencies of FElupe, e.g. `pyvista <https://github.com/pyvista/pyvista>`_ for interactive visualizations, are installed by replacing the last line.

.. code-block:: shell

   pip install --editable ".[all]"

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   examples/index
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
