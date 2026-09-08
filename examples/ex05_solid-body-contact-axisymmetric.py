r"""
Solid-body contact (axisymmetric)
---------------------------------
A rubber cylinder is pressed on a (stiffer) cylinder. Both cylinders are meshed
individually and are combined to a single mesh.

Compared to Ex. 4, axisymmetric fields are used.
"""

# sphinx_gallery_thumbnail_number = -1
import felupe as fem
import numpy as np

from felupe_contact import SolidBodyContact

# %%
# The rotation axis of an axisymmetric field is the first axis of the mesh, i.e.
# :math:`(X, Y) \widehat{=} (Z, R)`. Hence both cylinders are stacked along the first
# axis and their radial coordinates start at the rotation axis :math:`R = 0`.
bottom = fem.Rectangle(a=(0, 0), b=(1, 1), n=(5, 9))
top = fem.Rectangle(a=(1.02, 0), b=(1.62, 0.7), n=(5, 7))
container = fem.MeshContainer([bottom, top], merge=True)
mesh = container.stack()

region = fem.RegionQuad(mesh)
field = fem.FieldContainer([fem.FieldAxisymmetric(region, dim=2)])
solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

# %%
# The contact surfaces are created as boundary regions on the same mesh. The contact
# potential of an axisymmetric field is integrated on the surface of revolution, i.e.
# the differential area of a face is the product of its differential length and the
# circumference :math:`2 \pi R` of the circle at its radial coordinate.
mask = np.logical_and(mesh.x > 1.0, mesh.x < 1.1)
secondary = fem.FieldContainer(
    [fem.FieldAxisymmetric(fem.RegionQuadBoundary(mesh, mask=mask), dim=2)]
)
primary = fem.FieldContainer(
    [
        fem.FieldAxisymmetric(
            fem.RegionQuadBoundary(mesh, mask=np.isclose(mesh.x, 1.0)),
            dim=2,
        )
    ]
)
contact = SolidBodyContact(secondary, primary, items=[solid])

# %%
# The bottom face of the lower cylinder is fixed and the top face of the upper cylinder
# is moved downwards.
boundaries = {
    "fixed": fem.Boundary(field[0], fx=0.0),
    "clamped": fem.Boundary(field[0], fx=mesh.x.max(), skip=(1, 0)),
    "move": fem.Boundary(field[0], fx=mesh.x.max(), skip=(0, 1)),
}
move = fem.math.linsteps([0, -0.3], num=6)
step = fem.Step(
    items=[solid, contact],
    ramp={boundaries["move"]: move},
    boundaries=boundaries,
)
job = fem.Job(steps=[step]).evaluate(verbose=0)

# %%
# The number of integration points which are in contact and the penalty stiffness,
# which is estimated from the given items, are available in the contact object.
print(contact.results.npoints_in_contact)

# %%
# The principal values of the Cauchy stress are available in the solid body
# and can be plotted.
solid.plot("Principal Values of Cauchy Stress").show()
