r"""
Solid-body contact
------------------
A rubber block is pressed on a (stiffer) block. Both blocks are meshed
individually and are combined to a single mesh.

Compared to Ex. 1 / 2, individual sub-meshes are used.
"""

# sphinx_gallery_thumbnail_number = -1
import felupe as fem
from felupe_contact import SolidBodyContact

# %%
#
import felupe as fem
import numpy as np

bottom = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(4, 4, 3))
top = fem.Cube(a=(0.15, 0.15, 1.02), b=(0.85, 0.85, 1.62), n=(3, 3, 3))
container = fem.MeshContainer([bottom, top], merge=True)

regions = [fem.RegionHexahedron(m) for m in container]
fields = [fem.FieldContainer([fem.Field(r, dim=3)]) for r in regions]

field = fem.field.merge(fields)
solids = [fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=f) for f in fields]

# %%
# The contact surfaces are created as boundary regions on the same mesh. In contrast to
# Ex. 1, no mask is used: all faces on the outline of the mesh are contact surfaces. The
# contact search discards the face-pairs which are not able to touch each other, i.e.
# the faces which belong to the same body and the faces which are located around a
# corner.
fields_boundary = [
    fem.FieldContainer([fem.Field(fem.RegionHexahedronBoundary(m), dim=3)])
    for m in container
]
contact = SolidBodyContact(*fields_boundary, items=solids)

# %%
# The bottom face of the lower block is fixed and the top face of the upper block is
# moved downwards.
boundaries = {
    "fixed": fem.Boundary(field[0], fz=0.0),
    "clamped": fem.Boundary(field[0], fz=field.region.mesh.z.max(), skip=(0, 0, 1)),
    "move": fem.Boundary(field[0], fz=field.region.mesh.z.max(), skip=(1, 1, 0)),
}
move = fem.math.linsteps([0, -0.3], num=3)
step = fem.Step(
    items=[*solids, contact],
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
solids[1].plot(
    "Principal Values of Cauchy Stress",
    plotter=solids[0].plot("Principal Values of Cauchy Stress", opacity=0.2),
).show()
