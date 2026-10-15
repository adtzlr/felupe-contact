# SPDX-License-Identifier: GPL-3.0-or-later

import warnings

import felupe as fem
import numpy as np
import pytest

from felupe_contact import SolidBodyContact, SolidBodyContactNitsche
from felupe_contact._nitsche import select, volume_cells
from felupe_contact._solidbody_contact import copy_statevars

# the boundary regions of FElupe with their cell-region and the keyword arguments which
# convert a linear mesh into the mesh of the region
BOUNDARY_REGIONS = [
    (fem.RegionQuad, {}),
    (fem.RegionQuadraticQuad, dict(order=2, calc_midfaces=False)),
    (fem.RegionBiQuadraticQuad, dict(order=2, calc_midfaces=True)),
    (fem.RegionHexahedron, {}),
    (
        fem.RegionQuadraticHexahedron,
        dict(order=2, calc_midfaces=False, calc_midvolumes=False),
    ),
    (
        fem.RegionTriQuadraticHexahedron,
        dict(order=2, calc_midfaces=True, calc_midvolumes=True),
    ),
]

HEXAHEDRON = BOUNDARY_REGIONS[3]
QUAD = BOUNDARY_REGIONS[0]


def cases():
    "Return the regions of FElupe, combined with their valid field types."

    parameters = []
    for cell_region, convert in BOUNDARY_REGIONS:
        dim = 3 if "Hexahedron" in cell_region.__name__ else 2
        types = (
            [fem.Field] if dim == 3 else [fem.FieldPlaneStrain, fem.FieldAxisymmetric]
        )

        for field_type in types:
            parameters.append(
                pytest.param(
                    (cell_region, convert),
                    field_type,
                    id=f"{cell_region.__name__}-{field_type.__name__}",
                )
            )

    return parameters


def two_blocks(dim, convert=None, top=1.02):
    """Return the mesh of a block which is placed on top of a bigger block and the mask
    of the points of the upper block. The lower face of the upper block is located at
    ``top``, i.e. the initial gap is 0.02 by default."""

    a, b = (0.0,) * dim, (1.0,) * dim
    c, d = (0.15,) * (dim - 1) + (top,), (0.85,) * (dim - 1) + (top + 0.6,)
    n = (4,) * (dim - 1) + (3,)

    blocks = [fem.Rectangle, fem.Cube][dim - 2]
    lower, upper = blocks(a, b, n=n), blocks(c, d, n=n)
    mesh = fem.MeshContainer([lower, upper], merge=True).stack()

    if convert:
        mesh = fem.mesh.convert(mesh, **convert)

    # the cells of the upper block follow the cells of the lower block
    mask = np.zeros(mesh.npoints, dtype=bool)
    mask[np.unique(mesh.cells[lower.ncells :])] = True

    return mesh, mask


def solid_body(mesh, field_type=fem.Field, region=fem.RegionHexahedron, kind="solid"):
    "Return the field and a solid body of a mesh."

    dim = mesh.dim

    # the rotation axis of an axisymmetric field is the first axis of the mesh, i.e. the
    # mesh is moved away from the rotation axis
    if field_type is fem.FieldAxisymmetric:
        mesh = mesh.copy()
        mesh.points[:, 1] += 0.5

    field = fem.FieldContainer([field_type(region(mesh), dim=dim)])

    if kind == "solid":
        solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=5.0), field=field)

    elif kind == "incompressible":
        solid = fem.SolidBodyNearlyIncompressible(
            umat=fem.NeoHooke(mu=1.0), field=field, bulk=20.0
        )

    else:
        umat = fem.Hyperelastic(
            fem.finite_strain_viscoelastic, mu=1.0, eta=1.0, dtime=1.0, nstatevars=6
        )
        solid = fem.SolidBodyNearlyIncompressible(umat=umat, field=field, bulk=20.0)

    return field, solid


def nitsche(regions, field_type, kind="solid", top=1.02, **kwargs):
    """Return a Nitsche contact on all faces of two blocks, its field and the mask of
    the points of the upper block."""

    cell_region, convert = regions
    dim = 2 if field_type in (fem.FieldPlaneStrain, fem.FieldAxisymmetric) else 3
    mesh, upper = two_blocks(dim, convert, top=top)

    field, solid = solid_body(mesh, field_type, cell_region, kind=kind)
    contact = SolidBodyContactNitsche(solid, max_distance=0.4, **kwargs)

    return contact, field, solid, upper


def configure(field, upper, seed=0):
    "Press the upper block into the lower block and distort the faces."

    rng = np.random.default_rng(seed)
    values = field[0].values
    dim = values.shape[1]

    values[upper, dim - 1] = -0.06
    values[upper, : dim - 1] += 0.03
    values += 2e-3 * rng.standard_normal(values.shape)


def central_differences(contact, field, dofs, eps=1e-7):
    """Return the central differences of the contact force vector w.r.t. the given
    degrees of freedom, evaluated for fixed faces and fixed state variables."""

    values = field[0].values.copy()
    faces = [pair.face.copy() for pair in contact.pairs]
    columns = []

    for dof in dofs:
        column = 0.0
        for sign in (1, -1):
            field[0].values[:] = values
            field[0].values.ravel()[dof] += sign * eps

            for pair, face in zip(contact.pairs, faces):
                pair.face[:] = face

            contact._values = None
            vector = contact.assemble.vector(field=field).toarray().ravel()
            column = column + sign * vector / (2 * eps)

        columns.append(column)

    field[0].values[:] = values
    for pair, face in zip(contact.pairs, faces):
        pair.face[:] = face

    contact._values = None

    return np.stack(columns, axis=1)


def check_tangent(contact, field, ndofs=24):
    "Check that the stiffness matrix is the tangent of the contact force vector."

    force = contact.assemble.vector(field=field).toarray().ravel()
    stiffness = contact.assemble.matrix(field=field).toarray()

    assert contact.results.npoints_in_contact > 0
    assert np.linalg.norm(force) > 0

    rng = np.random.default_rng(seed=0)
    active = np.flatnonzero(np.abs(stiffness).sum(axis=0) > 0)
    dofs = rng.choice(active, size=min(ndofs, len(active)), replace=False)

    tangent = central_differences(contact, field, dofs)
    scale = np.abs(stiffness).max()

    assert np.abs(stiffness[:, dofs] - tangent).max() < 1e-6 * scale

    return stiffness


@pytest.mark.parametrize("regions, field_type", cases())
@pytest.mark.parametrize("kind", ["solid", "incompressible"])
def test_stiffness_is_the_tangent_of_the_force(regions, field_type, kind):
    """The stiffness matrix of the Nitsche contact is the tangent of the force vector,
    including the change of the stress tensor of the solid bodies."""

    contact, field, solid, upper = nitsche(regions, field_type, kind=kind)
    configure(field, upper)
    stiffness = check_tangent(contact, field)

    # the stiffness matrix of a Nitsche contact is not symmetric
    scale = np.abs(stiffness).max()
    assert np.abs(stiffness - stiffness.T).max() > 1e-3 * scale

    # the normal stress of the solid bodies contributes to the contact pressure
    traction = np.concatenate([t for t in contact.results.traction if t is not None])
    assert np.abs(traction).max() > 0


def test_stiffness_is_the_tangent_with_state_variables():
    """The stiffness matrix is the tangent of the force vector for a material model
    with state variables, which are evaluated at the integration points of the contact
    surfaces."""

    contact, field, solid, upper = nitsche(HEXAHEDRON, fem.Field, kind="viscoelastic")
    statevars = contact.results.statevars["material"][0]

    assert statevars.shape[0] == 6
    assert np.allclose(statevars, 0.0)

    # a converged increment updates the state variables of all integration points
    field[0].values[upper, 2] = -0.05
    contact.assemble.vector(field=field)
    contact.results.update_statevars()

    statevars = contact.results.statevars["material"][0]
    assert not np.allclose(statevars, 0.0)

    configure(field, upper)
    check_tangent(contact, field)


def test_geometric_stiffness():
    """Without the geometric part, the stiffness matrix differs from the tangent of the
    force vector."""

    contact, field, solid, upper = nitsche(HEXAHEDRON, fem.Field)
    configure(field, upper)

    stiffness = contact.assemble.matrix(field=field).toarray()
    contact.geometric_stiffness = False
    contact._values = None
    material = contact.assemble.matrix(field=field).toarray()

    assert np.abs(stiffness - material).max() > 1e-3 * np.abs(stiffness).max()


def patch_test(cls, dim=3, kind="solid", move=-0.2, **kwargs):
    """Compress two stacked blocks with matching meshes uniaxially, i.e. the stress
    state is homogeneous. Return the contact and the normal component of the first
    Piola-Kirchhoff stress tensor."""

    if dim == 3:
        blocks, region, field_type = fem.Cube, fem.RegionHexahedron, fem.Field
    else:
        blocks, region, field_type = fem.Rectangle, fem.RegionQuad, fem.FieldPlaneStrain

    lower = blocks(a=(0,) * dim, b=(1,) * dim, n=3)
    upper = blocks(a=(0,) * (dim - 1) + (1,), b=(1,) * (dim - 1) + (2,), n=3)
    mesh = fem.MeshContainer([lower, upper], merge=False).stack()

    field, solid = solid_body(mesh, field_type, region, kind=kind)
    contact = cls(solid, max_distance=0.4, **kwargs)

    # symmetry conditions, a fixed bottom face and a moved top face
    axis = dim - 1
    skip = lambda i: tuple(int(j != i) for j in range(dim))
    coord = ["fx", "fy", "fz"][:dim]

    boundaries = {
        f"symmetry-{i}": fem.Boundary(field[0], **{coord[i]: 0.0}, skip=skip(i))
        for i in range(axis)
    }
    boundaries["bottom"] = fem.Boundary(field[0], **{coord[axis]: 0.0}, skip=skip(axis))
    boundaries["move"] = fem.Boundary(field[0], **{coord[axis]: 2.0}, skip=skip(axis))

    step = fem.Step(
        items=[solid, contact],
        ramp={boundaries["move"]: fem.math.linsteps([0, move], num=2)},
        boundaries=boundaries,
    )
    fem.Job(steps=[step]).evaluate(verbose=0)

    stress = solid.evaluate.gradient(field)[0][axis, axis]

    return contact, stress


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", ["solid", "incompressible"])
def test_patch_test(dim, kind):
    """The contact pressure of a homogeneous stress state is equal to the normal stress
    of the solid bodies and the gap is zero, independent of the Nitsche parameter. A
    penalty contact penetrates."""

    for penalty_scale in [0.5, 10.0]:
        contact, stress = patch_test(
            SolidBodyContactNitsche,
            dim=dim,
            kind=kind,
            smoothing=0.0,
            penalty_scale=penalty_scale,
        )

        gap = np.concatenate([g for g in contact.results.gap if g is not None])
        pressure = np.concatenate(
            [p for p in contact.results.pressure if p is not None]
        )

        assert np.allclose(stress, stress.mean())
        assert np.allclose(pressure, -stress.mean())
        assert np.abs(gap).max() < 1e-10

    contact, stress = patch_test(SolidBodyContact, dim=dim, kind=kind)
    gap = np.concatenate([g for g in contact.results.gap if g is not None])

    assert gap.max() < -1e-3


def test_patch_test_with_smoothing():
    "The regularization of the contact pressure leads to a small penetration only."

    contact, stress = patch_test(SolidBodyContactNitsche, kind="incompressible")
    gap = np.concatenate([g for g in contact.results.gap if g is not None])

    assert gap.max() < 0
    assert gap.min() > -0.15 * contact.smoothing


def solve(kind="solid", move=-0.3, num=3, plugins=None, **kwargs):
    """Compress a rubber block on top of a bigger block, i.e. the example of the
    docstring, and return the contact, the solid body and the job."""

    mesh, upper = two_blocks(3)
    field, solid = solid_body(mesh, kind=kind)
    contact = SolidBodyContactNitsche(solid, **kwargs)

    boundaries = {
        "fixed": fem.Boundary(field[0], fz=0.0),
        "clamped": fem.Boundary(field[0], fz=mesh.z.max(), skip=(0, 0, 1)),
        "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0)),
    }
    step = fem.Step(
        items=[solid, contact],
        ramp={boundaries["move"]: fem.math.linsteps([0, move], num=num)},
        boundaries=boundaries,
    )
    job = fem.Job(steps=[step], plugins=plugins or []).evaluate(verbose=0)

    return contact, solid, job


@pytest.mark.parametrize("kind", ["solid", "incompressible", "viscoelastic"])
def test_job(kind):
    """The Nitsche contact converges in a job, the contact pressure is positive and the
    state variables of the material model are updated."""

    contact, solid, job = solve(kind=kind)

    assert contact.results.npoints_in_contact > 0
    assert all([len(norms) <= 8 for norms in job.fnorms])

    pressure = np.concatenate([p for p in contact.results.pressure if p is not None])
    assert np.all(pressure > 0)

    statevars = contact.results.statevars["material"][0]
    assert statevars.shape[0] == (6 if kind == "viscoelastic" else 0)


def test_job_with_cutback():
    """A large increment is solved with the cutback plugin. The checkpoint contains the
    normal stress and the state variables of the material model."""

    cutback = fem.CutbackPlugin()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        contact, solid, job = solve(kind="viscoelastic", num=1, plugins=[cutback])

    assert contact.results.npoints_in_contact > 0

    checkpoint = contact.checkpoint()
    traction = [t.copy() for t in contact.results.traction]
    statevars = copy_statevars(contact.results.statevars)

    field = solid.field
    field[0].values[:] *= 0.5
    contact.assemble.vector(field=field)
    contact.results.update_statevars()

    contact.restore(checkpoint)

    assert contact._traction is None
    assert contact.results._statevars is None

    for a, b in zip(contact.results.traction, traction):
        assert np.allclose(a, b)

    for a, b in zip(contact.results.statevars["material"], statevars["material"]):
        assert np.allclose(a, b)


def test_surface_pairs():
    """A contact surface which is also the primary surface is evaluated once, distinct
    surfaces are evaluated twice. Both are weighted by the stiffness of the bodies. A
    single-pass contact is not weighted."""

    mesh, upper = two_blocks(3)
    field, solid = solid_body(mesh)

    for contact in [
        SolidBodyContactNitsche(solid),
        SolidBodyContactNitsche(solid, solid),
        SolidBodyContactNitsche(solid, mask=mesh.z >= 0, two_pass=False),
    ]:
        assert len(contact.pairs) == 1
        assert contact.pairs[0].weight == 1.0
        assert contact.weighted

    mask = np.logical_and(mesh.z > 1.0, mesh.z < 1.1)
    mask_primary = np.isclose(mesh.z, 1.0)

    contact = SolidBodyContactNitsche(solid, mask=mask, mask_primary=mask_primary)
    assert [pair.weight for pair in contact.pairs] == [1.0, 1.0]
    assert contact.pairs[1].field is contact.field_primary
    assert contact.weighted

    contact = SolidBodyContactNitsche(
        solid, mask=mask, mask_primary=mask_primary, two_pass=False
    )
    assert [pair.weight for pair in contact.pairs] == [1.0]
    assert not contact.weighted

    # two solid bodies with their own material models on a merged mesh
    field, solids, upper = two_solid_bodies(mu=(1.0, 2.0))
    contact = SolidBodyContactNitsche(*solids)

    assert [pair.weight for pair in contact.pairs] == [1.0, 1.0]
    assert [surface.umat for surface in contact.surfaces] == [
        solids[0].umat,
        solids[1].umat,
    ]


def two_solid_bodies(mu=(1.0, 1000.0), gap=0.02, n=3):
    """Return the global field, two solid bodies with their own material models on the
    meshes of two stacked blocks and the mask of the points of the upper block. The
    meshes of both blocks are equal, i.e. their stiffness differs by the ratio of the
    shear moduli. The initial gap between the blocks is ``gap``."""

    # the points of the meshes are stacked, i.e. coincident points are not merged
    lower = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=n)
    other = fem.Cube(a=(0, 0, 1 + gap), b=(1, 1, 2 + gap), n=n)
    container = fem.MeshContainer([lower, other], merge=False)
    mesh = container.stack()

    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    solids = []
    for m, modulus in zip(container.meshes, mu):
        f = fem.FieldContainer([fem.Field(fem.RegionHexahedron(m), dim=3)])
        f.link(field)
        umat = fem.NeoHooke(mu=modulus, bulk=5.0 * modulus)
        solids.append(fem.SolidBody(umat, f))

    upper = np.zeros(mesh.npoints, dtype=bool)
    upper[np.unique(container.meshes[1].cells)] = True

    return field, solids, upper


@pytest.mark.parametrize("two_pass", [True, False])
def test_weights(two_pass):
    """The contact tractions are weighted by the stiffness of the bodies, i.e. the
    softer body dominates. The weights of both sides of a contact zone sum up to one,
    the Nitsche parameter is estimated from the softest body."""

    field, solids, upper = two_solid_bodies(mu=(1.0, 1000.0))
    contact = SolidBodyContactNitsche(*solids, two_pass=two_pass, max_distance=0.4)

    configure(field, upper)
    contact.assemble.vector(field=field)

    # equal meshes, i.e. the stiffness of the bodies differs by the ratio of the moduli
    stiffness = contact.stiffness[np.isfinite(contact.stiffness)]
    assert len(stiffness) == 2
    assert np.isclose(stiffness.max() / stiffness.min(), 1000.0)
    assert np.isclose(contact.penalty, 10.0 * stiffness.min())

    weights = [w for w in contact._weights if w is not None]

    if not two_pass:
        assert weights == [1.0]
        return

    # the integration points on the soft body are weighted by 1000 / 1001, those on
    # the stiff body by 1 / 1001
    assert len(weights) == 2
    assert np.allclose(weights[0], 1000 / 1001)
    assert np.allclose(weights[1], 1 / 1001)

    # the weights don't change the tangent
    check_tangent(contact, field)


def test_stiffness_given_penalty():
    "The stiffness of the bodies is estimated also for a given Nitsche parameter."

    field, solids, upper = two_solid_bodies(mu=(1.0, 3.0))
    contact = SolidBodyContactNitsche(*solids, penalty=5.0, max_distance=0.4)

    configure(field, upper)
    contact.assemble.vector(field=field)

    assert contact.penalty == 5.0
    assert np.sum(np.isfinite(contact.stiffness)) == 2

    weights = np.concatenate([w for w in contact._weights if w is not None])
    assert np.allclose(np.unique(np.round(weights, 12)), [0.25, 0.75])

    # the items must contain all solid bodies of the contact surfaces
    contact = SolidBodyContactNitsche(*solids, items=solids[:1], max_distance=0.4)

    with pytest.raises(ValueError, match="stiffness of a body"):
        contact.assemble.vector(field=field)


@pytest.mark.parametrize("ratio", [1.0, 1000.0])
def test_patch_test_two_materials(ratio):
    """The contact pressures of both passes are equal to the normal stress of the solid
    bodies and the gap is zero for solid bodies of different stiffness. The lateral
    displacements are constrained (uniaxial strain), i.e. the stress state of both
    bodies is homogeneous although their lateral contraction differs."""

    field, solids, upper = two_solid_bodies(mu=(1.0, ratio), gap=0.0)
    contact = SolidBodyContactNitsche(*solids, smoothing=0.0, max_distance=0.4)
    mesh = field.region.mesh

    boundaries = {
        "x": fem.Boundary(field[0], mask=np.isin(mesh.x, [0, 1]), skip=(0, 1, 1)),
        "y": fem.Boundary(field[0], mask=np.isin(mesh.y, [0, 1]), skip=(1, 0, 1)),
        "bottom": fem.Boundary(field[0], fz=0.0, skip=(1, 1, 0)),
        "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0)),
    }
    step = fem.Step(
        items=[*solids, contact],
        ramp={boundaries["move"]: fem.math.linsteps([0, -0.1], num=2)},
        boundaries=boundaries,
    )
    job = fem.Job(steps=[step]).evaluate(x0=field, verbose=0)

    stress = np.concatenate(
        [solid.evaluate.gradient()[0][2, 2].ravel() for solid in solids]
    )
    gap = np.concatenate([g for g in contact.results.gap if g is not None])
    pressure = np.concatenate([p for p in contact.results.pressure if p is not None])

    assert len(contact.pairs) == 2
    assert all([g is not None for g in contact.results.gap])
    assert np.allclose(stress, stress.mean())
    assert np.allclose(pressure, -stress.mean())
    assert np.abs(gap).max() < 1e-10


def test_job_two_materials():
    """A rubber block is pressed on a thousand times stiffer block. The weighted
    two-pass contact converges and the penetration is small."""

    field, solids, upper = two_solid_bodies(mu=(1000.0, 1.0), n=(5, 5, 3))
    contact = SolidBodyContactNitsche(*solids)
    mesh = field.region.mesh

    boundaries = {
        "fixed": fem.Boundary(field[0], fz=0.0),
        "clamped": fem.Boundary(field[0], fz=mesh.z.max(), skip=(0, 0, 1)),
        "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0)),
    }
    step = fem.Step(
        items=[*solids, contact],
        ramp={boundaries["move"]: fem.math.linsteps([0, -0.2], num=3)},
        boundaries=boundaries,
    )
    job = fem.Job(steps=[step]).evaluate(x0=field, verbose=0)

    assert all([len(norms) <= 8 for norms in job.fnorms])

    gap = np.concatenate([g for g in contact.results.gap if g is not None])
    assert -gap.min() < 1e-2 * contact.size


def test_overlapping_surfaces():
    "A warning is raised if the contact surfaces share some but not all faces."

    mesh, upper = two_blocks(3)
    field, solid = solid_body(mesh)

    with pytest.warns(UserWarning, match="share"):
        SolidBodyContactNitsche(
            solid, mask=mesh.z >= 0.99, mask_primary=np.isclose(mesh.z, 1.0)
        )


def test_invalid_solid_bodies():
    "The stress tensor of a solid body with a displacement field is required."

    mesh, upper = two_blocks(3)
    field, solid = solid_body(mesh)

    with pytest.raises(TypeError, match="material model"):
        SolidBodyContactNitsche(field)

    with pytest.raises(TypeError, match="material model"):
        SolidBodyContactNitsche(solid, field)

    region = fem.RegionHexahedron(mesh)
    mixed = fem.FieldsMixed(region, n=3)
    umat = fem.ThreeFieldVariation(fem.NeoHooke(mu=1.0, bulk=5.0))

    with pytest.raises(NotImplementedError, match="mixed-field"):
        SolidBodyContactNitsche(fem.SolidBody(umat, mixed))

    boundary = fem.FieldContainer(
        [fem.Field(fem.RegionHexahedronBoundary(mesh), dim=3)]
    )
    surface = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=5.0), boundary)

    with pytest.raises(TypeError, match="boundary region"):
        SolidBodyContactNitsche(surface)


def test_no_contact():
    """Integration points in front of the primary surface are considered by the contact
    search, but they are not in contact without a compressive normal stress."""

    contact, field, solid, upper = nitsche(HEXAHEDRON, fem.Field)

    assert contact.max_gap == 0.5 * contact.size

    # the faces are close to each other, but the solid bodies are stress-free
    pair = contact.pairs[0]
    x = pair.field.region.mesh.points
    kinematics = pair.kinematics(
        x,
        max_distance=0.4,
        candidates=8,
        tolerance=0.1,
        facing=0.1,
        self_contact=False,
        max_gap=contact.max_gap,
    )
    assert np.all(kinematics["gap"] > 0)

    force = contact.assemble.vector(field=field)
    stiffness = contact.assemble.matrix(field=field)

    assert contact.results.npoints_in_contact == 0
    assert force.nnz == 0
    assert stiffness.nnz == 0
    assert contact.results.pressure == [None]

    # the blocks are far away from each other
    contact, field, solid, upper = nitsche(HEXAHEDRON, fem.Field, top=2.0)
    contact.assemble.vector(field=field)

    assert contact.results.npoints_in_contact == 0


def test_pressure():
    """The regularized contact pressure and its derivative are continuous, the contact
    pressure is exact beyond the transition zone."""

    contact, field, solid, upper = nitsche(HEXAHEDRON, fem.Field, penalty=10.0)
    delta = contact.smoothing
    eps = 1e-9

    # the gap at the transitions, for a given normal stress. the second derivative of
    # the contact pressure is bounded by 4 penalty / delta, i.e. the derivatives on both
    # sides of a transition differ by less than 8 eps penalty / delta
    traction = 0.3
    for s in [0.0, delta]:
        gap = traction / contact.penalty - s + np.array([eps, -eps])
        pressure, dpressure = contact.pressure(gap, traction)

        assert np.isclose(pressure[0], pressure[1], atol=1e-8)
        assert np.isclose(
            dpressure[0], dpressure[1], atol=10 * eps * contact.penalty / delta
        )

    gap = np.linspace(-2 * delta, 2 * delta, 101)
    pressure, dpressure = contact.pressure(gap)
    difference = (contact.pressure(gap + eps)[0] - contact.pressure(gap - eps)[0]) / (
        2 * eps
    )

    assert np.allclose(pressure[gap <= -delta], -contact.penalty * gap[gap <= -delta])
    assert np.allclose(pressure[gap >= 0], 0.0)
    assert np.all(np.diff(pressure) <= 0)
    assert np.allclose(dpressure, difference, atol=1e-4)

    contact.smoothing = 0.0
    pressure, dpressure = contact.pressure(gap)

    assert np.allclose(pressure, contact.penalty * np.maximum(-gap, 0.0))
    assert np.allclose(dpressure, np.where(gap < 0, -contact.penalty, 0.0))


@pytest.mark.parametrize("regions", [QUAD, HEXAHEDRON, BOUNDARY_REGIONS[5]])
def test_volume_cells(regions):
    "The cells of a boundary region are rotated cells of the mesh."

    cell_region, convert = regions
    dim = 3 if "Hexahedron" in cell_region.__name__ else 2
    mesh = two_blocks(dim, convert)[0]

    field_type = fem.Field if dim == 3 else fem.FieldPlaneStrain
    field, solid = solid_body(mesh, field_type, region=cell_region)
    contact = SolidBodyContactNitsche(solid)

    cells = contact.pairs[0].field.region.mesh.cells
    cell, permutation = volume_cells(cells, mesh.cells)

    assert np.all(cell >= 0)
    assert np.array_equal(cells, np.take_along_axis(mesh.cells[cell], permutation, 1))


def test_select():
    "The arrays are selected along their last axis."

    arrays = {"a": np.arange(4), "b": np.arange(12).reshape(3, 4)}
    selected = select(arrays, np.array([True, False, True, False]))

    assert np.array_equal(selected["a"], [0, 2])
    assert np.array_equal(selected["b"], arrays["b"][:, [0, 2]])
    assert selected["b"].flags["C_CONTIGUOUS"]


def test_repr():
    "The representation shows the Nitsche parameter."

    contact, solid, job = solve(num=3)

    assert "SolidBodyContactNitsche" in repr(contact)
    assert "Nitsche parameter" in repr(contact)
    assert f"{contact.results.npoints_in_contact}" in repr(contact)
