r"""
Nitsche contact between two solid bodies
----------------------------------------
A rubber block is pressed on a stiff base. Both blocks are solid bodies with their own
material models and the contact between them is enforced by Nitsche's method, see
:class:`~felupe_contact.SolidBodyContactNitsche`.

Compared to the penalty contact of Ex. 1, the contact pressure of a Nitsche contact
contains the normal stress of the solid bodies. Hence, the penetration doesn't grow with
the contact pressure.
"""

# sphinx_gallery_thumbnail_number = -1
import felupe as fem

from felupe_contact import SolidBodyContact, SolidBodyContactNitsche

# %%
# The base and the rubber block are meshed individually and are combined to a single
# mesh.
base = fem.Cube(a=(-0.25, -0.25, 0), b=(1.25, 1.25, 0.5), n=(7, 7, 3))
block = fem.Cube(a=(0.15, 0.15, 0.52), b=(0.85, 0.85, 1.12), n=(5, 5, 4))
container = fem.MeshContainer([base, block], merge=True)
mesh = container.stack()


# %%
# The model is created by a function, which takes the class of the contact. Each block
# is a solid body with its own field container, which is linked to the global field
# container. The rubber block is nearly incompressible and the base is a thousand times
# stiffer. The contact surfaces are created on the outlines of the meshes of both solid
# bodies, where the rubber surface is the secondary surface.
#
# The bottom face of the base is fixed and the top face of the rubber block is moved
# downwards. A job with several solid bodies requires the global field container.
def solve(Contact):
    "Press the rubber block on the base and return the contact and the solid bodies."

    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    fields = [
        fem.FieldContainer([fem.Field(fem.RegionHexahedron(m), dim=3)])
        for m in container.meshes
    ]
    for f in fields:
        f.link(field)

    metal = fem.SolidBody(fem.NeoHooke(mu=1000.0, bulk=5000.0), fields[0])
    rubber = fem.SolidBodyNearlyIncompressible(
        fem.NeoHooke(mu=1.0), fields[1], bulk=100.0
    )
    contact = Contact(rubber, metal)

    boundaries = {
        "fixed": fem.Boundary(field[0], fz=0.0),
        "clamped": fem.Boundary(field[0], fz=mesh.z.max(), skip=(0, 0, 1)),
        "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0)),
    }
    step = fem.Step(
        items=[metal, rubber, contact],
        ramp={boundaries["move"]: fem.math.linsteps([0, -0.15], num=3)},
        boundaries=boundaries,
    )
    fem.Job(steps=[step]).evaluate(x0=field, verbose=0)

    return contact, rubber, metal


# %%
# The contact is evaluated on both surfaces and the contact tractions are weighted by
# the stiffness of the bodies, i.e. mainly the stress of the rubber block is used.
contact, rubber, metal = solve(SolidBodyContactNitsche)
print(contact)

# %%
# The gap, the contact pressure and the normal stress of the rubber block at the
# integration points in contact are available in the results of the contact. The first
# list item belongs to the integration points of the rubber surface. The same model
# with a penalty contact penetrates about ten times more, because its penetration is
# proportional to the contact pressure.
penalty = solve(SolidBodyContact)[0]

print(f"Max. penetration (Nitsche): {-contact.results.gap[0].min():.2e}")
print(f"Max. penetration (penalty): {-penalty.results.gap[0].min():.2e}")

# %%
# The rubber block is compressed and bulges at its free side faces.
rubber.plot("Principal Values of Cauchy Stress").show()
