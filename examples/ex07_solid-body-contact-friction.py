r"""
Solid-body contact with friction
--------------------------------
A rubber block is pressed on a stiff base and is moved tangentially afterwards. The
contact between the block and the base is frictional (Coulomb friction).

Compared to Ex. 1, a friction coefficient is given and the base is a thousand times
stiffer than the rubber block.
"""

# sphinx_gallery_thumbnail_number = -1
import felupe as fem
import numpy as np

from felupe_contact import SolidBodyContact

# %%
# The base and the rubber block are meshed individually and are combined to a single
# mesh. The material parameters are given for each cell of the mesh, i.e. the cells of
# the base are a thousand times stiffer than the cells of the rubber block.
base = fem.Cube(a=(-1, 0, 0), b=(2, 1, 0.5), n=(13, 5, 3))
block = fem.Cube(a=(0.15, 0.15, 0.52), b=(0.85, 0.85, 1.12), n=(6, 6, 5))
container = fem.MeshContainer([base, block], merge=True)
mesh = container.stack()

region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])

mu = np.where(np.arange(mesh.ncells) < base.ncells, 1000.0, 1.0).reshape(1, -1)
solid = fem.SolidBody(umat=fem.NeoHooke(mu=mu, bulk=5 * mu), field=field)

# %%
# The secondary surface is the bottom face of the rubber block and the primary surface
# is the top face of the base. The friction coefficient is :math:`\mu = 0.3`.
contact = SolidBodyContact(
    solid,
    mask=np.logical_and(mesh.z > 0.5, mesh.z < 0.6),
    mask_primary=np.isclose(mesh.z, 0.5),
    friction=0.3,
)

# %%
# The bottom face of the base is fixed. The top face of the rubber block is pressed
# downwards in a first step and it is moved in direction :math:`X` in a second step.
# The pressed boundary keeps its last value in the second step.
top = np.isclose(mesh.z, mesh.z.max())
boundaries = {
    "fixed": fem.Boundary(field[0], fz=0.0),
    "x": fem.Boundary(field[0], mask=top, skip=(0, 1, 1)),
    "y": fem.Boundary(field[0], mask=top, skip=(1, 0, 1)),
    "z": fem.Boundary(field[0], mask=top, skip=(1, 1, 0)),
}
press = fem.Step(
    items=[solid, contact],
    ramp={boundaries["z"]: fem.math.linsteps([0, -0.1], num=2)},
    boundaries=boundaries,
)
move = fem.Step(
    items=[solid, contact],
    ramp={boundaries["x"]: fem.math.linsteps([0, 0.4], num=8)},
    boundaries=boundaries,
)

# %%
# The transition from sticking to sliding of the whole contact zone (gross sliding) is
# the most demanding part for the Newton-Raphson method. A cutback subdivides the
# substeps which fail to converge.
job = fem.CharacteristicCurve(
    steps=[press, move], boundary=boundaries["x"], plugins=[fem.CutbackPlugin()]
)
job.evaluate(verbose=0)

# %%
# The tangential force increases until the whole contact zone slides. Then, it is
# equal to the friction coefficient times the normal force.
fig, ax = job.plot(
    xaxis=0,
    yaxis=0,
    xlabel=r"Displacement $u_X$ $\longrightarrow$",
    ylabel=r"Force $F_X$ $\longrightarrow$",
    marker="o",
)

force = job.y[-1]
print(f"Tangential force / normal force: {-force[0] / force[2]:.3f}")

# %%
# The tangential contact tractions (shear) and the sliding integration points are
# available in the results of the contact.
print(contact)

# %%
# The displacements in direction :math:`X` show the sliding of the rubber block, which
# is sheared by the friction.
solid.plot("Displacement", component=0).show()
