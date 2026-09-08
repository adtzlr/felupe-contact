import felupe as fem
import numpy as np
import pytest

from felupe_contact import SolidBodyContact

# the boundary regions of FElupe with their cell-region, the cell type of the faces and
# the keyword arguments which convert a linear mesh into the mesh of the region
BOUNDARY_REGIONS = [
    (fem.RegionQuad, fem.RegionQuadBoundary, "line", {}),
    (
        fem.RegionQuadraticQuad,
        fem.RegionQuadraticQuadBoundary,
        "VTK_LAGRANGE_LINE",
        dict(order=2, calc_midfaces=False),
    ),
    (
        fem.RegionBiQuadraticQuad,
        fem.RegionBiQuadraticQuadBoundary,
        "VTK_LAGRANGE_LINE",
        dict(order=2, calc_midfaces=True),
    ),
    (fem.RegionHexahedron, fem.RegionHexahedronBoundary, "quad", {}),
    (
        fem.RegionQuadraticHexahedron,
        fem.RegionQuadraticHexahedronBoundary,
        "quad8",
        dict(order=2, calc_midfaces=False, calc_midvolumes=False),
    ),
    (
        fem.RegionTriQuadraticHexahedron,
        fem.RegionTriQuadraticHexahedronBoundary,
        "quad9",
        dict(order=2, calc_midfaces=True, calc_midvolumes=True),
    ),
]


def cases():
    "Return the boundary regions of FElupe, combined with their valid field types."

    parameters = []
    for cell_region, boundary_region, face_type, convert in BOUNDARY_REGIONS:
        dim = 3 if "Hexahedron" in boundary_region.__name__ else 2

        # in 2d, the out-of-plane behaviour is defined by the type of the field
        types = (
            [fem.Field] if dim == 3 else [fem.FieldPlaneStrain, fem.FieldAxisymmetric]
        )

        for field_type in types:
            parameters.append(
                pytest.param(
                    cell_region,
                    boundary_region,
                    face_type,
                    convert,
                    field_type,
                    id=f"{boundary_region.__name__}-{field_type.__name__}",
                )
            )

    return parameters


def rectangle(a, b, n, axis):
    "Return a rectangle- or a cube-mesh, oriented along ``axis``."

    if len(a) == 3:
        return fem.Cube(a=a, b=b, n=n)

    # the rotation axis of an axisymmetric field is the first axis of the mesh, hence
    # the blocks are stacked along the first axis instead of the second one
    if axis == 0:
        a, b, n = [tuple(reversed(x)) for x in [a, b, n]]

    return fem.Rectangle(a=a, b=b, n=n)


def two_blocks(dim, convert=None, axis=None):
    """Return the mesh of a block which is placed on top of a bigger block, with an
    initial gap of 0.02 between both blocks. The blocks are stacked along ``axis``."""

    if axis is None:
        axis = dim - 1

    a, b = (0.0,) * dim, (1.0,) * dim
    c, d = (0.15,) * (dim - 1) + (1.02,), (0.85,) * (dim - 1) + (1.62,)
    n = (4,) * (dim - 1) + (3,)

    mesh = fem.MeshContainer(
        [rectangle(a, b, n, axis), rectangle(c, d, n, axis)], merge=True
    ).stack()

    return fem.mesh.convert(mesh, **convert) if convert else mesh


def penetrating(mesh, field_type, cell_region, boundary_region, **kwargs):
    """Return a contact and its field, evaluated for a configuration with a penetration
    of the upper block into the lower one."""

    axis = mesh.dim - 1

    surface = fem.FieldContainer([field_type(boundary_region(mesh), dim=mesh.dim)])
    contact = SolidBodyContact(surface, surface, **kwargs)

    field = fem.FieldContainer([field_type(cell_region(mesh), dim=mesh.dim)])

    # translate the upper block downwards, the initial gap between both blocks is 0.02
    field[0].values[mesh.points[:, axis] > 1.0, axis] = -0.06

    return contact, field


@pytest.mark.parametrize("cell_region, region, face_type, convert, field_type", cases())
def test_stiffness_is_the_tangent_of_the_force(
    cell_region, region, face_type, convert, field_type
):
    "The contact stiffness matrix is the tangent of the contact force vector."

    dim = 3 if "Hexahedron" in region.__name__ else 2
    mesh = two_blocks(dim, convert)
    contact, field = penetrating(
        mesh, field_type, cell_region, region, penalty=100.0, max_distance=0.4
    )
    values = field[0].values.copy()

    assert contact.pairs[0].element.element.cell_type == face_type

    force = contact.assemble.vector(field=field).toarray().ravel()
    stiffness = contact.assemble.matrix(field=field).toarray()

    assert contact.results.npoints_in_contact > 0
    assert np.linalg.norm(force) > 0

    # the stiffness matrix of the contact is symmetric
    scale = np.abs(stiffness).max()
    assert np.abs(stiffness - stiffness.T).max() < 1e-10 * scale

    # central differences of the contact force vector for a subset of the dofs
    rng = np.random.default_rng(seed=0)
    dofs = rng.choice(len(force), size=25, replace=False)
    tangent = np.zeros((len(force), len(dofs)))

    eps = 1e-7
    for j, dof in enumerate(dofs):
        for sign in (1, -1):
            perturbed = values.copy()
            perturbed.ravel()[dof] += sign * eps

            field[0].values[:] = perturbed
            contact._values = None
            contact._kinematics = None

            vector = contact.assemble.vector(field=field).toarray().ravel()
            tangent[:, j] += sign * vector / (2 * eps)

    assert np.abs(stiffness[:, dofs] - tangent).max() < 1e-6 * scale


@pytest.mark.parametrize("field_type", [fem.FieldPlaneStrain, fem.FieldAxisymmetric])
def test_2d(field_type):
    "Two blocks of a two-dimensional mesh are compressed, see Ex. 4 and Ex. 5."

    # the rotation axis of an axisymmetric field is the first axis of the mesh
    axis = 0 if field_type is fem.FieldAxisymmetric else 1
    mesh = two_blocks(dim=2, axis=axis)

    field = fem.FieldContainer([field_type(fem.RegionQuad(mesh), dim=2)])
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

    z = mesh.points[:, axis]
    surface = fem.FieldContainer(
        [field_type(fem.RegionQuadBoundary(mesh, mask=np.abs(z - 1.01) < 0.1), dim=2)]
    )
    contact = SolidBodyContact(surface, surface, items=[solid], max_distance=0.4)

    # the transverse dofs of the moved face are clamped, its axial dof is prescribed
    axial = tuple(int(i == axis) for i in range(2))
    transverse = tuple(1 - i for i in axial)

    key = ["fx", "fy"][axis]
    boundaries = {
        "fixed": fem.Boundary(field[0], **{key: 0.0}),
        "clamped": fem.Boundary(field[0], **{key: z.max()}, skip=axial),
        "move": fem.Boundary(field[0], **{key: z.max()}, skip=transverse),
    }
    step = fem.Step(
        items=[solid, contact],
        ramp={boundaries["move"]: fem.math.linsteps([0, -0.2], num=4)},
        boundaries=boundaries,
    )
    fem.Job(steps=[step]).evaluate(verbose=0)

    assert contact.results.npoints_in_contact > 0
    assert np.isclose(np.abs(field[0].values[:, axis]).max(), 0.2)

    for gap, pressure in zip(contact.results.gap, contact.results.pressure):
        assert np.all(gap < 0)
        assert np.all(pressure > 0)


def test_differential_area_of_a_plane_strain_field():
    "The contact of a plane strain field is integrated on the boundary region."

    mesh = two_blocks(dim=2)
    surface = fem.FieldContainer(
        [fem.FieldPlaneStrain(fem.RegionQuadBoundary(mesh), dim=2)]
    )
    pair = SolidBodyContact(surface, surface, penalty=1.0).pairs[0]

    assert np.allclose(pair.dA, pair.dV)


def test_differential_area_of_an_axisymmetric_field():
    "The contact of an axisymmetric field is integrated on the surface of revolution."

    mesh = two_blocks(dim=2)
    surface = fem.FieldContainer(
        [fem.FieldAxisymmetric(fem.RegionQuadBoundary(mesh), dim=2)]
    )
    pair = SolidBodyContact(surface, surface, penalty=1.0).pairs[0]
    radius = surface[0].radius

    assert np.allclose(pair.dA, 2 * np.pi * radius * pair.dV)

    # the differential area vanishes on the rotation axis, i.e. at a zero radius
    assert np.any(np.isclose(radius, 0.0))
    assert np.all(pair.dA[np.isclose(radius, 0.0)] == 0.0)


@pytest.mark.parametrize("field_type", [fem.FieldPlaneStrain, fem.FieldAxisymmetric])
def test_contact_force_of_a_uniform_penetration(field_type):
    """The total contact force of a uniform penetration is the product of the contact
    pressure and the area of the contact zone. This is the width of the contact zone
    for a plane strain field and the area of a circle for an axisymmetric field."""

    penalty, penetration, gap, width = 100.0, 0.04, 0.02, 0.7

    # the rotation axis of an axisymmetric field is the first axis of the mesh
    axis = 0 if field_type is fem.FieldAxisymmetric else 1

    bottom = rectangle((0.0, 0.0), (1.0, 1.0), (5, 5), axis)
    top = rectangle((0.0, 1.0 + gap), (width, 1.6), (4, 4), axis)
    mesh = fem.MeshContainer([bottom, top], merge=True).stack()

    z = mesh.points[:, axis]

    def surface(mask):
        region = fem.RegionQuadBoundary(mesh, mask=mask)
        return fem.FieldContainer([field_type(region, dim=2)])

    # the contact zone is the face of the upper block, which is the secondary surface
    secondary = surface(np.isclose(z, 1.0 + gap))
    contact = SolidBodyContact(
        secondary, surface(np.isclose(z, 1.0)), penalty=penalty, smoothing=0.0
    )

    field = fem.FieldContainer([field_type(fem.RegionQuad(mesh), dim=2)])
    field[0].values[z > 1.0, axis] = -(gap + penetration)

    force = contact.assemble.vector(field=field).toarray().reshape(-1, 2)

    # the gap is constant for a rigid translation of the upper block
    assert np.allclose(contact.results.gap[0], -penetration)

    # the total contact force on the points of the upper block
    total = np.abs(force[z > 1.0, axis].sum())
    area = np.pi * width**2 if axis == 0 else width

    assert np.isclose(total, penalty * penetration * area)
