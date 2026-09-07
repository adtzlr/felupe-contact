r"""
Solid-body contact (quadratic faces)
------------------------------------
A rubber block is pressed on a (stiffer) block. Both blocks are meshed
individually and are combined to a single mesh.

Compared to Ex. 4, a mesh with bi-quadratic cells is used, i.e. the faces of the contact
surfaces are quadratic lines instead of linear ones.
"""

# sphinx_gallery_thumbnail_number = -1
import felupe as fem
import numpy as np

from felupe_contact import SolidBodyContact

# %%
# All boundary regions of FElupe are supported. A linear mesh is converted into a
# bi-quadratic one, where the faces of its boundary region are quadratic lines.
bottom = fem.Rectangle(a=(0, 0), b=(1, 1), n=(9, 5))
top = fem.Rectangle(a=(0.15, 1.02), b=(0.85, 1.62), n=(7, 5))
container = fem.MeshContainer([bottom, top], merge=True)
mesh = fem.mesh.convert(container.stack(), order=2, calc_midfaces=True)

region = fem.RegionBiQuadraticQuad(mesh)
field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

# %%
# The contact surfaces are created as boundary regions on the same mesh. The estimate of
# the penalty stiffness assumes an equal area per point of a contact surface, which is
# not the case for quadratic cells. Hence the estimate is stiffer than for linear cells
# and ``penalty_scale`` is decreased here.
mask = np.logical_and(mesh.y > 1.0, mesh.y < 1.1)
secondary = fem.FieldContainer(
    [fem.FieldPlaneStrain(fem.RegionBiQuadraticQuadBoundary(mesh, mask=mask), dim=2)]
)
primary = fem.FieldContainer(
    [
        fem.FieldPlaneStrain(
            fem.RegionBiQuadraticQuadBoundary(mesh, mask=np.isclose(mesh.y, 1.0)),
            dim=2,
        )
    ]
)
contact = SolidBodyContact(secondary, primary, items=[solid], penalty_scale=3.0)

# %%
# The bottom edge of the lower block is fixed and the top edge of the upper block is
# moved downwards.
boundaries = {
    "fixed": fem.Boundary(field[0], fy=0.0),
    "clamped": fem.Boundary(field[0], fy=mesh.y.max(), skip=(0, 1)),
    "move": fem.Boundary(field[0], fy=mesh.y.max(), skip=(1, 0)),
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
# which is estimated from the given items, are available in the contact object. A
# quadratic face has three integration points instead of two.
print(contact.results.npoints_in_contact)

# %%
# The principal values of the Cauchy stress are available in the solid body
# and can be plotted.
solid.plot("Principal Values of Cauchy Stress", nonlinear_subdivision=2).show()
