.. _felupe-contact-api:

API Reference
=============

.. currentmodule:: felupe_contact

The contact formulations of felupe-contact are items of FElupe, i.e. they are used in a :class:`~felupe.Step` and in a :class:`~felupe.Job` together with the solid bodies.

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
   _element.BatchedElement
   _element.face_element

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

.. autoclass:: felupe_contact._element.BatchedElement
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: felupe_contact._element.face_element
