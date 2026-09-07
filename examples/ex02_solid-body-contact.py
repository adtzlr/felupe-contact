r"""
Solid-body contact
------------------
A rubber block is pressed on a (stiffer) block. Both blocks are meshed
individually and are combined to a single mesh.

Compared to Ex. 1, all faces of the mesh are used as contact surfaces.
"""

# sphinx_gallery_thumbnail_number = -1
import felupe as fem

from felupe_contact import SolidBodyContact

# %%
#
bottom = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(4, 4, 3))
top = fem.Cube(a=(0.15, 0.15, 1.02), b=(0.85, 0.85, 1.62), n=(3, 3, 3))
container = fem.MeshContainer([bottom, top], merge=True)
mesh = container.stack()

region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])
solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

# %%
# The contact surfaces are created as boundary regions on the same mesh. In contrast to
# Ex. 1, no mask is used: all faces on the outline of the mesh are contact surfaces. The
# contact search discards the face-pairs which are not able to touch each other, i.e.
# the faces which belong to the same body and the faces which are located around a
# corner.
secondary = fem.FieldContainer([fem.Field(fem.RegionHexahedronBoundary(mesh), dim=3)])
primary = fem.FieldContainer([fem.Field(fem.RegionHexahedronBoundary(mesh), dim=3)])
contact = SolidBodyContact(secondary, primary, items=[solid])

# %%
# The bottom face of the lower block is fixed and the top face of the upper block is
# moved downwards.
boundaries = {
    "fixed": fem.Boundary(field[0], fz=0.0),
    "clamped": fem.Boundary(field[0], fz=mesh.z.max(), skip=(0, 0, 1)),
    "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0)),
}
move = fem.math.linsteps([0, -0.3], num=3)
step = fem.Step(
    items=[solid, contact],
    ramp={boundaries["move"]: move},
    boundaries=boundaries,
)
job = fem.Job(steps=[step]).evaluate(verbose=0)

# %%
# The number of integration points which are in contact and the penalty stiffness,
# which is estimated from the given items, are available in the contact object. Both
# faces of the contact zone are used as secondary surfaces here, hence twice the number
# of integration points of Ex. 1 are in contact.
print(contact.results.npoints_in_contact)

# %%
# The principal values of the Cauchy stress are available in the solid body
# and can be plotted.
solid.plot("Principal Values of Cauchy Stress").show()
