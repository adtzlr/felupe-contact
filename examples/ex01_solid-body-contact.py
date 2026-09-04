r"""
Solid-body contact
------------------
A rubber block is pressed on a (stiffer) block. Both blocks are meshed
individually and are combined to a single mesh.
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
mesh = container.stack()

region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])
solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

# %%
# The contact surfaces are created as boundary regions on the same mesh. Only the
# faces on the outline of the mesh which are located in the region of interest are
# used.
mask = np.logical_and(mesh.z > 1.0, mesh.z < 1.1)
secondary = fem.FieldContainer(
    [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=mask), dim=3)]
)
mask_primary = np.isclose(mesh.z, 1.0)
primary = fem.FieldContainer(
    [
        fem.Field(
            fem.RegionHexahedronBoundary(mesh, mask=mask_primary),
            dim=3,
        )
    ]
)
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
# which is estimated from the given items, are available in the contact object.
print(contact.results.npoints_in_contact)

# %%
# The principal values of the Cauchy stress are available in the solid body
# and can be plotted.
solid.plot("Principal Values of Cauchy Stress").show()
