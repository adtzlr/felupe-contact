.. _felupe-contact-api:

API Reference
=============

.. currentmodule:: felupe_contact

The contacts of felupe-contact are items of FElupe, see :class:`felupe.mechanics.Item`. They are created on the solid bodies (or on field containers on the boundary regions of FElupe) and they are used in a :class:`~felupe.Step` and in a :class:`~felupe.Job` together with the solid bodies.

**Contact**

.. autosummary::

   SolidBodyContact

**Helpers for the Contact Kinematics**

These classes and functions are used internally by :class:`~felupe_contact.SolidBodyContact`. They are not part of the public API, i.e. they may change without a deprecation.

.. autosummary::

   _solidbody_contact.ContactSurfacePair
   _solidbody_contact.closest_point_projection
   _solidbody_contact.unit_normals
   _solidbody_contact.connected_bodies
   _solidbody_contact.sparse_vector
   _solidbody_contact.sparsity_pattern
   _solidbody_contact.sparse_matrix
   _element.BatchedElement
   _element.face_element
   _boundary.boundary_region
   _boundary.contact_surface

**Related Classes of FElupe**

The building blocks of a model with a contact are taken from FElupe.

.. list-table::
   :widths: 40 60

   *   - Solid bodies and their regions
       - :class:`~felupe.SolidBody`, :class:`~felupe.SolidBodyNearlyIncompressible`, :class:`~felupe.RegionQuad`, :class:`~felupe.RegionQuadraticQuad`, :class:`~felupe.RegionBiQuadraticQuad`, :class:`~felupe.RegionHexahedron`, :class:`~felupe.RegionQuadraticHexahedron`, :class:`~felupe.RegionTriQuadraticHexahedron`
   *   - Contact surfaces
       - :class:`~felupe.RegionQuadBoundary`, :class:`~felupe.RegionQuadraticQuadBoundary`, :class:`~felupe.RegionBiQuadraticQuadBoundary`, :class:`~felupe.RegionHexahedronBoundary`, :class:`~felupe.RegionQuadraticHexahedronBoundary`, :class:`~felupe.RegionTriQuadraticHexahedronBoundary`
   *   - Displacement fields
       - :class:`~felupe.Field`, :class:`~felupe.FieldPlaneStrain`, :class:`~felupe.FieldAxisymmetric`, :class:`~felupe.FieldContainer`
   *   - Steps, jobs and plugins
       - :class:`~felupe.Step`, :class:`~felupe.Job`, :class:`~felupe.CharacteristicCurve`, :class:`~felupe.CutbackPlugin`, :class:`~felupe.LinesearchPlugin`
   *   - Other contacts
       - :class:`~felupe.MultiPointContact`, :class:`~felupe.ContactRigidPlane`, :class:`~felupe.ContactRigidCylinder`, :class:`~felupe.ThirdMediumContactMixed`

**Detailed API Reference**

.. autoclass:: felupe_contact.SolidBodyContact
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe_contact._solidbody_contact.ContactSurfacePair
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: felupe_contact._solidbody_contact.closest_point_projection

.. autofunction:: felupe_contact._solidbody_contact.unit_normals

.. autofunction:: felupe_contact._solidbody_contact.connected_bodies

.. autofunction:: felupe_contact._solidbody_contact.sparse_vector

.. autofunction:: felupe_contact._solidbody_contact.sparsity_pattern

.. autofunction:: felupe_contact._solidbody_contact.sparse_matrix

.. autoclass:: felupe_contact._element.BatchedElement
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: felupe_contact._element.face_element

.. autofunction:: felupe_contact._boundary.boundary_region

.. autofunction:: felupe_contact._boundary.contact_surface
