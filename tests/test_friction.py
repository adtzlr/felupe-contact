# SPDX-License-Identifier: GPL-3.0-or-later

import felupe as fem
import numpy as np
import pytest

from felupe_contact import SolidBodyContact
from felupe_contact._solidbody_contact import contiguous, copy_statevars

# the boundary regions of FElupe with their cell-region and the keyword arguments which
# convert a linear mesh into the mesh of the region
BOUNDARY_REGIONS = [
    (fem.RegionQuad, fem.RegionQuadBoundary, {}),
    (
        fem.RegionQuadraticQuad,
        fem.RegionQuadraticQuadBoundary,
        dict(order=2, calc_midfaces=False),
    ),
    (
        fem.RegionBiQuadraticQuad,
        fem.RegionBiQuadraticQuadBoundary,
        dict(order=2, calc_midfaces=True),
    ),
    (fem.RegionHexahedron, fem.RegionHexahedronBoundary, {}),
    (
        fem.RegionQuadraticHexahedron,
        fem.RegionQuadraticHexahedronBoundary,
        dict(order=2, calc_midfaces=False, calc_midvolumes=False),
    ),
    (
        fem.RegionTriQuadraticHexahedron,
        fem.RegionTriQuadraticHexahedronBoundary,
        dict(order=2, calc_midfaces=True, calc_midvolumes=True),
    ),
]

HEXAHEDRON = BOUNDARY_REGIONS[3]
QUAD = BOUNDARY_REGIONS[0]


def cases():
    "Return the boundary regions of FElupe, combined with their valid field types."

    parameters = []
    for cell_region, boundary_region, convert in BOUNDARY_REGIONS:
        dim = 3 if "Hexahedron" in boundary_region.__name__ else 2
        types = (
            [fem.Field] if dim == 3 else [fem.FieldPlaneStrain, fem.FieldAxisymmetric]
        )

        for field_type in types:
            parameters.append(
                pytest.param(
                    (cell_region, boundary_region, convert),
                    field_type,
                    id=f"{boundary_region.__name__}-{field_type.__name__}",
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


def frictional(regions, field_type, top=1.02, **kwargs):
    """Return a frictional contact on all faces of two blocks, its field and the mask of
    the points of the upper block."""

    cell_region, boundary_region, convert = regions
    dim = 2 if field_type in (fem.FieldPlaneStrain, fem.FieldAxisymmetric) else 3
    mesh, upper = two_blocks(dim, convert, top=top)

    surface = fem.FieldContainer([field_type(boundary_region(mesh), dim=dim)])
    kwargs = {"penalty": 100.0, "max_distance": 0.4, "friction": 0.3, **kwargs}
    contact = SolidBodyContact(surface, surface, **kwargs)
    field = fem.FieldContainer([field_type(cell_region(mesh), dim=dim)])

    return contact, field, upper


def configure(contact, field, upper, case, seed=0):
    """Accept a converged (penetrating) configuration of two blocks and move the upper
    block tangentially. The tangential displacement leads to sticking (``"stick"``) or
    sliding (``"slide"``) integration points or the integration points slide over the
    edges of the faces of the primary surface (``"transfer"``). For ``"new"``, the
    converged configuration is not accepted, i.e. all integration points come into
    contact within the increment."""

    rng = np.random.default_rng(seed)
    values = field[0].values
    dim = values.shape[1]

    # the converged configuration with a penetration and with distorted faces
    values[upper, dim - 1] = -0.05
    values[~upper] += 0.003 * rng.standard_normal(values[~upper].shape)

    if case != "new":
        contact.assemble.vector(field=field)
        contact.results.update_statevars()

    shift = {"stick": 1e-4, "slide": 0.05, "transfer": 0.15, "new": 0.05}[case]
    values[upper, : dim - 1] += shift * np.linspace(1.0, 0.5, dim - 1)
    values[upper, dim - 1] -= 0.01
    values += 1e-4 * rng.standard_normal(values.shape)


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


def slides(contact):
    "Return the mask of the sliding integration points of all surface pairs."

    return np.concatenate([s for s in contact.results.slip if s is not None])


@pytest.mark.parametrize("regions, field_type", cases())
@pytest.mark.parametrize("case", ["stick", "slide"])
def test_stiffness_is_the_tangent_of_the_force(regions, field_type, case):
    "The stiffness matrix of the friction is the tangent of the contact force vector."

    contact, field, upper = frictional(regions, field_type)
    configure(contact, field, upper, case)
    stiffness = check_tangent(contact, field)

    # the integration points stick or slide
    assert np.all(slides(contact) == (case == "slide"))

    # the stiffness matrix of a frictional contact is not symmetric
    scale = np.abs(stiffness).max()
    assert np.abs(stiffness - stiffness.T).max() > 1e-3 * scale


LAWS = {
    "limited": {},
    "proportional": {"elastic_slip": 0.02},
    "constant": {"penalty_tangential": 50.0},
    "rough": {"friction": np.inf},
}


@pytest.mark.parametrize("regions, field_type", [(HEXAHEDRON, fem.Field)])
@pytest.mark.parametrize("case", ["stick", "slide", "transfer", "new"])
@pytest.mark.parametrize("law", LAWS.keys())
def test_stiffness_is_the_tangent_for_all_friction_laws(regions, field_type, case, law):
    """The stiffness matrix is the tangent of the force vector for all tangential
    penalty laws, for anchors on other faces and for new integration points."""

    contact, field, upper = frictional(regions, field_type, **LAWS[law])
    configure(contact, field, upper, case)
    check_tangent(contact, field)

    if law == "rough" or case == "stick":
        assert not np.any(slides(contact))
    else:
        assert np.all(slides(contact))

    statevars = contact.results.statevars
    for kin, anchors in zip(contact._kinematics, statevars["face"]):
        anchor = anchors[kin["point"]]

        if case == "new":
            # no integration point is anchored, i.e. all are scaled by the fraction of
            # the increment after the first contact
            assert np.all(anchor < 0)

        if case == "transfer":
            # some anchors are located on other faces than the projected points
            assert np.any((anchor >= 0) & (anchor != kin["face"]))


@pytest.mark.parametrize("field_type", [fem.FieldPlaneStrain, fem.FieldAxisymmetric])
@pytest.mark.parametrize("case", ["transfer", "new"])
def test_stiffness_is_the_tangent_in_2d(field_type, case):
    "The stiffness matrix of the friction is the tangent of the force vector in 2d."

    contact, field, upper = frictional(QUAD, field_type)
    configure(contact, field, upper, case)
    check_tangent(contact, field)


def test_stiffness_is_the_tangent_for_initially_penetrating_bodies():
    """Integration points which are already penetrating in the converged configuration
    but not in contact (e.g. initially) are not scaled."""

    contact, field, upper = frictional(HEXAHEDRON, fem.Field, top=0.98)
    field[0].values[upper, 0] += 1e-3

    check_tangent(contact, field)

    # all integration points are new, i.e. they are not anchored
    statevars = contact.results.statevars
    statevars = {key: value[0] for key, value in statevars.items() if key != "points"}
    statevars["points"] = contact.results.statevars["points"]

    pair, kin = contact.pairs[0], contact._kinematics[0]
    scale, dscale = pair.anchors(kin, statevars)[2:]

    assert np.all(statevars["face"] < 0)
    assert np.allclose(scale, 1.0) and np.allclose(dscale, 0.0)


def test_symmetric_stiffness_matrix():
    "The symmetric part of the stiffness matrix is used with ``symmetric=True``."

    contact, field, upper = frictional(HEXAHEDRON, fem.Field, symmetric=True)
    configure(contact, field, upper, "slide")
    stiffness = contact.assemble.matrix(field=field).toarray()

    assert np.allclose(stiffness, stiffness.T)


def test_geometric_stiffness():
    "The geometric part of the stiffness matrix of the friction may be deactivated."

    matrices = []
    for geometric_stiffness in [True, False]:
        contact, field, upper = frictional(
            HEXAHEDRON, fem.Field, geometric_stiffness=geometric_stiffness
        )
        configure(contact, field, upper, "slide")
        matrices.append(contact.assemble.matrix(field=field).toarray())

    assert not np.allclose(*matrices)


def test_relocated_anchors_reproduce_the_elastic_slip():
    """The anchors of sliding integration points are relocated, i.e. the force vector
    is reproduced for the next increment with unchanged displacements."""

    contact, field, upper = frictional(HEXAHEDRON, fem.Field)
    configure(contact, field, upper, "slide")

    force = contact.assemble.vector(field=field).toarray()
    assert np.all(slides(contact))

    contact.results.update_statevars()
    assert np.allclose(contact.assemble.vector(field=field).toarray(), force)

    # an integration point, which is located on the limit, sticks
    assert not np.any(slides(contact))


@pytest.mark.parametrize("regions", [HEXAHEDRON, BOUNDARY_REGIONS[-1]])
def test_relocate(regions):
    "The tangential part of the relative position of a relocated anchor is zero."

    contact, field, upper = frictional(regions, fem.Field)
    configure(contact, field, upper, "stick")
    contact.assemble.vector(field=field)

    pair, kin = contact.pairs[0], contact._kinematics[0]
    x = contact.field.region.mesh.points + field[0].values
    vertices = pair.vertices(x, pair.cells_faces_primary[kin["face"]])

    rng = np.random.default_rng(seed=1)
    points = kin["points"] + 0.05 * rng.standard_normal(kin["points"].shape)
    coordinates = pair.relocate(
        points, vertices, kin["tangents"], np.zeros_like(kin["coordinates"])
    )

    h = pair.element_primary.function(coordinates)
    x_ref = np.einsum("ap,aip->ip", h, vertices)

    assert np.allclose(np.einsum("Jip,ip->Jp", kin["tangents"], points - x_ref), 0)


def stiff_base(dim=3, field_type=fem.Field, **kwargs):
    """Return a frictional contact of a rubber block on a stiff base, the solid body,
    the field and the boundary conditions."""

    if dim == 3:
        base = fem.Cube(a=(-1, 0, 0), b=(2, 1, 0.5), n=(7, 3, 2))
        block = fem.Cube(a=(0.15, 0.15, 0.52), b=(0.85, 0.85, 1.12), n=(4, 4, 3))
        cell_region, boundary_region = (
            fem.RegionHexahedron,
            fem.RegionHexahedronBoundary,
        )
    else:
        base = fem.Rectangle(a=(-1, 0), b=(2, 0.5), n=(7, 2))
        block = fem.Rectangle(a=(0.15, 0.52), b=(0.85, 1.12), n=(4, 3))
        cell_region, boundary_region = fem.RegionQuad, fem.RegionQuadBoundary

    mesh = fem.MeshContainer([base, block], merge=True).stack()
    field = fem.FieldContainer([field_type(cell_region(mesh), dim=dim)])

    # the base is a thousand times stiffer than the rubber block
    mu = np.where(np.arange(mesh.ncells) < base.ncells, 1000.0, 1.0).reshape(1, -1)
    solid = fem.SolidBody(fem.NeoHooke(mu=mu, bulk=5 * mu), field)

    y = mesh.points[:, dim - 1]
    secondary = fem.FieldContainer(
        [field_type(boundary_region(mesh, mask=(y > 0.51) & (y < 0.6)), dim=dim)]
    )
    primary = fem.FieldContainer(
        [field_type(boundary_region(mesh, mask=np.isclose(y, 0.5)), dim=dim)]
    )
    contact = SolidBodyContact(secondary, primary, items=[solid], **kwargs)

    top = np.isclose(y, y.max())
    boundaries = {"fixed": fem.Boundary(field[0], mask=np.isclose(y, 0.0))}
    for i in range(dim):
        skip = np.ones(dim, dtype=int)
        skip[i] = 0
        boundaries[i] = fem.Boundary(field[0], mask=top, skip=tuple(skip))

    return contact, solid, field, boundaries, mesh.points[:, dim - 1] > 0.51


def solve(dim=3, field_type=fem.Field, slide=0.3, num=6, plugins=None, **kwargs):
    """Press a rubber block on a stiff base, move it tangentially and return the contact,
    the contact forces on the rubber block of all substeps and the field."""

    contact, solid, field, boundaries, upper = stiff_base(dim, field_type, **kwargs)
    axis = dim - 1
    forces = []

    class Forces(fem.Plugin):
        "Track the contact force on the rubber block after each converged substep."

        def after_substep(self, context, state):
            force = -contact.results.force.toarray().reshape(-1, dim)
            forces.append(force[upper].sum(axis=0))

    # the pressed boundary keeps its last value in the second step
    press = fem.Step(
        items=[solid, contact],
        ramp={boundaries[axis]: fem.math.linsteps([0, -0.1], num=2)},
        boundaries=boundaries,
    )
    move = fem.Step(
        items=[solid, contact],
        ramp={boundaries[0]: fem.math.linsteps([0, slide], num=num)},
        boundaries=boundaries,
    )
    job = fem.Job(steps=[press, move], plugins=[Forces(), *(plugins or [])])
    job.evaluate(verbose=0)

    return contact, np.array(forces), field


def test_gross_sliding_on_a_stiff_base():
    """The tangential force is the friction coefficient times the normal force if all
    integration points slide."""

    contact, forces, field = solve(friction=0.3)

    # sticking at the beginning of the tangential movement
    assert -forces[-6, 0] / forces[-6, 2] < 0.29

    # gross sliding: the tangential force acts against the movement
    assert np.isclose(-forces[-1, 0] / forces[-1, 2], 0.3, atol=1e-3)
    assert np.all(slides(contact))

    # the tangential traction is equal to the limit of the sliding integration points
    shear = contact.results.shear[0]
    pressure = contact.results.pressure[0]

    assert shear.shape == (3, contact.results.npoints_in_contact)
    assert np.allclose(np.linalg.norm(shear, axis=0), 0.3 * pressure)
    assert np.all(shear[0] < 0)


def test_partial_slip():
    """Some integration points slide at the boundary of the contact zone, if the
    tangential movement is small. The Coulomb law is fulfilled by all integration
    points."""

    contact, forces, field = solve(friction=0.3, slide=0.01, num=1)

    slide = slides(contact)
    shear = np.linalg.norm(contact.results.shear[0], axis=0)
    limit = 0.3 * contact.results.pressure[0]

    assert np.any(slide) and not np.all(slide)
    assert np.allclose(shear[slide], limit[slide])
    assert np.all(shear[~slide] <= limit[~slide])
    assert 0 < -forces[-1, 0] / forces[-1, 2] < 0.3


def test_rough_contact_never_slides():
    "An infinite friction coefficient prevents sliding."

    contact, forces, field = solve(friction=np.inf, slide=0.1, num=2)

    assert not np.any(slides(contact))
    assert np.all(np.linalg.norm(contact.results.shear[0], axis=0) > 0)
    assert -forces[-1, 0] > -forces[-2, 0] > 0


def test_frictionless_contact():
    "A frictionless contact has no tangential tractions and no state variables."

    contact, forces, field = solve(friction=0.0, slide=0.1, num=2)

    assert contact.results.statevars is None
    assert contact.results.shear == [None]
    assert contact.results.slip == [None]
    assert np.abs(forces[-1, 0] / forces[-1, 2]) < 0.05


@pytest.mark.parametrize("field_type", [fem.FieldPlaneStrain, fem.FieldAxisymmetric])
def test_gross_sliding_2d(field_type):
    "The friction is formulated for plane strain and axisymmetric problems."

    contact, forces, field = solve(2, field_type, friction=0.3, slide=0.2, num=4)

    shear = contact.results.shear[0]
    pressure = contact.results.pressure[0]

    assert shear.shape == (2, contact.results.npoints_in_contact)
    assert np.all(slides(contact))
    assert np.allclose(np.linalg.norm(shear, axis=0), 0.3 * pressure)


def test_two_pass():
    "Each surface pair of a two-pass contact has its own anchors."

    contact, forces, field = solve(friction=0.3, slide=0.1, num=2, two_pass=True)

    statevars = contact.results.statevars
    assert len(statevars["face"]) == len(statevars["coordinates"]) == 2
    assert all(np.any(face >= 0) for face in statevars["face"])
    assert all(np.any(slip) for slip in contact.results.slip)


def test_state_variables_of_the_converged_increment():
    """The state variables are the deformed coordinates and the anchors of the last
    converged increment."""

    contact, forces, field = solve(friction=0.3, slide=0.1, num=2)

    statevars = contact.results.statevars
    x = contact.field.region.mesh.points + field[0].values

    assert np.allclose(statevars["points"], x)

    # all integration points in contact (and only those) are anchored
    kin = contact._kinematics[0]
    anchored = np.flatnonzero(statevars["face"][0] >= 0)

    assert np.array_equal(anchored, kin["point"])
    assert np.array_equal(statevars["face"][0][anchored], kin["face"])


def test_checkpoint_and_restore():
    "The state variables of the friction are restored from a checkpoint."

    contact, forces, field = solve(friction=0.3, slide=0.1, num=2)
    checkpoint = contact.checkpoint()
    force = contact.assemble.vector(field=field).toarray()

    # a modified state of the friction
    field[0].values[:, 0] += 0.05
    contact.assemble.vector(field=field)
    contact.results.update_statevars()
    field[0].values[:, 0] -= 0.05

    contact.restore(checkpoint)

    statevars = contact.results.statevars
    assert contact.results._statevars is None
    assert np.allclose(statevars["points"], checkpoint["statevars"]["points"])
    assert np.array_equal(statevars["face"][0], checkpoint["statevars"]["face"][0])
    assert np.allclose(contact.assemble.vector(field=field).toarray(), force)

    # the checkpoint is not modified by the restored state
    statevars["face"][0][:] = -1
    assert np.any(checkpoint["statevars"]["face"][0] >= 0)


class RejectOnce(fem.Plugin):
    """Reject the first converged solution of the tangential movement, i.e. after the
    state variables are updated. The first substep of the second step is the initial
    state of the tangential movement."""

    def __init__(self):
        self.steps = 0
        self.substeps = 0

    def before_step(self, context, state):
        self.steps += 1

    def after_newton(self, context, state):
        if self.steps == 2:
            self.substeps += 1

            if self.substeps == 2:
                raise fem.NewtonConvergenceError("The converged solution is rejected.")


def test_cutback_is_equal_to_smaller_increments():
    """A cutback of the tangential movement gives the same (path-dependent) result as
    smaller increments, i.e. the state variables are restored from the checkpoint."""

    cutback = fem.CutbackPlugin()
    field = solve(friction=0.3, slide=0.2, num=1, plugins=[RejectOnce(), cutback])[2]
    reference = solve(friction=0.3, slide=0.2, num=2)[2]

    assert cutback.load_factors[-1] == [0.5, 1.0]
    assert np.allclose(field[0].values, reference[0].values)


def test_tangential_penalty():
    "The tangential penalty stiffness is proportional to the pressure and limited."

    contact = stiff_base(friction=0.3, penalty=100.0, elastic_slip=0.01)[0]
    pressure = np.array([0.0, 1.0, 10.0])
    dpressure = np.array([0.0, -100.0, -100.0])

    stiffness, dstiffness = contact.tangential_penalty(pressure, dpressure)

    assert np.allclose(stiffness, [0.0, 30.0, 100.0])
    assert np.allclose(dstiffness, [0.0, -3000.0, 0.0])

    contact.friction = np.inf
    assert np.allclose(contact.tangential_penalty(pressure, dpressure)[0][:2], [0, 100])

    contact = stiff_base(friction=0.3, penalty_tangential=50.0)[0]
    stiffness, dstiffness = contact.tangential_penalty(pressure, dpressure)

    assert np.allclose(stiffness, 50.0) and np.allclose(dstiffness, 0.0)
    assert contact.elastic_slip is None


def test_default_elastic_slip():
    "The default elastic slip is one percent of the characteristic size of the faces."

    contact = stiff_base(friction=0.3)[0]

    assert np.isclose(contact.elastic_slip, 0.01 * contact.size)
    assert contact.penalty_tangential is None


def test_copy_statevars():
    "The state variables are copied, i.e. they don't share memory with the original."

    assert copy_statevars(None) is None

    statevars = {
        "points": np.zeros((2, 3)),
        "face": [np.zeros(4, dtype=int)],
        "coordinates": [np.zeros((2, 4))],
    }
    copied = copy_statevars(statevars)

    for key in ["face", "coordinates"]:
        assert not np.shares_memory(copied[key][0], statevars[key][0])

    assert not np.shares_memory(copied["points"], statevars["points"])


def test_contiguous():
    "The arrays of the face-pairs are converted to C-contiguous arrays."

    arrays = {"normal": np.ones((3, 8))[:, ::2], "point": np.arange(4)}
    converted = contiguous(arrays)

    assert not arrays["normal"].flags["C_CONTIGUOUS"]
    assert all(value.flags["C_CONTIGUOUS"] for value in converted.values())
    assert np.array_equal(converted["normal"], arrays["normal"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"friction": -0.1},
        {"friction": 0.3, "elastic_slip": 0.0},
        {"friction": 0.3, "penalty_tangential": -1.0},
        {"friction": 0.3, "elastic_slip": 0.01, "penalty_tangential": 1.0},
    ],
)
def test_invalid_parameters(kwargs):
    "Invalid parameters of the friction are rejected."

    with pytest.raises(ValueError):
        stiff_base(**kwargs)


def test_repr():
    "The friction coefficient and the sliding integration points are printed."

    contact, forces, field = solve(friction=0.3, slide=0.1, num=2)
    text = repr(contact)

    assert "Friction coefficient: 0.3" in text
    assert f"Integration points sliding: {np.sum(slides(contact))}" in text
    assert "Friction" not in repr(solve(slide=0.0, num=1)[0])
