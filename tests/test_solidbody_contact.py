import warnings

import felupe as fem
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from felupe_contact import SolidBodyContact


def two_blocks():
    "Return the mesh of a block which is placed on top of a bigger block."

    bottom = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(4, 4, 3))
    top = fem.Cube(a=(0.15, 0.15, 1.02), b=(0.85, 0.85, 1.62), n=(3, 3, 3))

    return fem.MeshContainer([bottom, top], merge=True).stack()


def solid_body(mesh):
    "Return the field and the solid body of a mesh."

    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

    return field, solid


def surface(mesh, mask=None):
    "Return a field container on a boundary region of a mesh."

    region = fem.RegionHexahedronBoundary(mesh, mask=mask)

    return fem.FieldContainer([fem.Field(region, dim=3)])


def surfaces(mesh, masked):
    "Return the secondary and the primary contact surface of the two blocks."

    if not masked:
        return surface(mesh), surface(mesh)

    # only the faces in the region of interest, i.e. near the contact zone
    return (
        surface(mesh, mask=np.logical_and(mesh.z > 1.0, mesh.z < 1.1)),
        surface(mesh, mask=np.isclose(mesh.z, 1.0)),
    )


def compress(mesh, field, solid, contact, move=-0.3, num=3):
    "Move the top face of the upper block downwards and return the solved job."

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

    return fem.Job(steps=[step]).evaluate(verbose=0)


def solve(masked, move=-0.3, num=3, **kwargs):
    "Compress the two blocks and return the contact and the displacements."

    mesh = two_blocks()
    field, solid = solid_body(mesh)
    secondary, primary = surfaces(mesh, masked=masked)

    contact = SolidBodyContact(secondary, primary, items=[solid], **kwargs)
    compress(mesh, field, solid, contact, move=move, num=num)

    return contact, field[0].values


def test_masked_surfaces():
    "The faces in the region of interest are used as contact surfaces, see Ex. 1."

    contact, displacement = solve(masked=True)

    assert contact.results.npoints_in_contact == 16
    assert np.isclose(np.abs(displacement).max(), 0.3)


def test_all_faces():
    "All faces on the outline of the mesh are used as contact surfaces, see Ex. 2."

    contact, displacement = solve(masked=False)

    # both faces of the contact zone are secondary surfaces, hence the number of
    # integration points in contact is twice the number of Ex. 1
    assert contact.results.npoints_in_contact == 32
    assert np.isclose(np.abs(displacement).max(), 0.3)


def test_all_faces_agrees_with_masked_surfaces():
    "Both choices of the contact surfaces give the same displacement field."

    displacement_masked = solve(masked=True)[1]
    displacement_all = solve(masked=False)[1]

    assert np.abs(displacement_masked - displacement_all).max() < 5e-3


def test_gap_is_negative():
    "The gap is negative for all integration points which are in contact."

    contact = solve(masked=False)[0]

    for gap in contact.results.gap:
        assert gap is None or np.all(gap < 0)


def test_pressure_is_positive():
    "The contact pressure is positive for all integration points in contact."

    contact = solve(masked=False)[0]

    for pressure in contact.results.pressure:
        assert pressure is None or np.all(pressure > 0)


@pytest.mark.parametrize("masked", [True, False])
def test_two_pass(masked):
    "A two-pass contact evaluates both surface pairs with half of the potential."

    contact = solve(masked=masked, num=1, two_pass=True)[0]

    assert len(contact.pairs) == 2
    assert contact.pairs[0].weight == 0.5
    assert contact.results.npoints_in_contact > 0


def test_without_geometric_stiffness():
    "The contact converges without the geometric part of the stiffness matrix."

    contact = solve(masked=False, num=1, geometric_stiffness=False)[0]

    assert contact.results.npoints_in_contact == 32


def test_without_smoothing():
    "The contact converges with a non-regularized penalty law."

    contact = solve(masked=False, num=1, smoothing=0.0)[0]

    assert contact.smoothing == 0.0
    assert contact.results.npoints_in_contact == 32


def test_parallel():
    "The contact is assembled with a threaded assembly."

    mesh = two_blocks()
    field, solid = solid_body(mesh)
    secondary, primary = surfaces(mesh, masked=False)
    contact = SolidBodyContact(secondary, primary, items=[solid], penalty=500.0)

    force = contact.assemble.vector(parallel=True)
    stiffness = contact.assemble.matrix(parallel=True)

    assert force.shape == (field[0].indices.dof.size, 1)
    assert stiffness.shape[0] == force.shape[0]


def test_given_penalty():
    "A given penalty stiffness is used instead of the estimated one."

    mesh = two_blocks()
    secondary, primary = surfaces(mesh, masked=False)
    contact = SolidBodyContact(secondary, primary, penalty=1234.0)

    assert contact.penalty == 1234.0


def test_estimated_penalty():
    "The penalty stiffness is estimated from the stiffness of the given items."

    mesh = two_blocks()
    field, solid = solid_body(mesh)
    secondary, primary = surfaces(mesh, masked=False)
    contact = SolidBodyContact(secondary, primary, items=[solid])

    assert contact.penalty is None
    assert contact.update_penalty() > 0
    assert contact.penalty > 0


def test_estimated_penalty_from_assembled_stiffness():
    "The already assembled stiffness matrix of an item is re-used."

    mesh = two_blocks()
    field, solid = solid_body(mesh)
    secondary, primary = surfaces(mesh, masked=False)

    solid.assemble.matrix()
    contact = SolidBodyContact(secondary, primary, items=[solid])

    assert solid.results.stiffness is not None
    assert contact.update_penalty() > 0


def test_default_smoothing_and_max_distance():
    "The smoothing length and the search distance are derived from the face size."

    mesh = two_blocks()
    secondary, primary = surfaces(mesh, masked=False)
    contact = SolidBodyContact(secondary, primary, penalty=1.0)

    assert contact.smoothing == 1e-2 * contact.size
    assert contact.max_distance == 5 * contact.size


def test_repr():
    "The representation contains the penalty stiffness and the contact state."

    mesh = two_blocks()
    secondary, primary = surfaces(mesh, masked=False)
    contact = SolidBodyContact(secondary, primary, penalty=100.0)

    assert "SolidBodyContact" in repr(contact)
    assert "100.0" in repr(contact)


def test_pressure_law():
    "The regularized penalty law is continuous at the end of the transition zone."

    mesh = two_blocks()
    secondary, primary = surfaces(mesh, masked=False)
    contact = SolidBodyContact(secondary, primary, penalty=100.0, smoothing=0.1)

    gap = np.array([0.5, 0.0, -0.05, -0.1, -0.5])
    pressure, dpressure = contact.pressure(gap)

    # open, transition (quadratic) and closed (linear) states of the contact
    assert np.allclose(pressure, [0.0, 0.0, 1.25, 5.0, 45.0])
    assert np.allclose(dpressure, [0.0, 0.0, -50.0, -100.0, -100.0])


def test_pressure_law_without_smoothing():
    "The non-regularized penalty law is linear for a closed contact."

    mesh = two_blocks()
    secondary, primary = surfaces(mesh, masked=False)
    contact = SolidBodyContact(secondary, primary, penalty=100.0, smoothing=0.0)

    pressure, dpressure = contact.pressure(np.array([0.5, -0.5]))

    assert np.allclose(pressure, [0.0, 50.0])
    assert np.allclose(dpressure, [0.0, -100.0])


def test_kinematics_are_cached():
    "The contact kinematics are only evaluated once per displacement field."

    mesh = two_blocks()
    field, solid = solid_body(mesh)
    secondary, primary = surfaces(mesh, masked=False)
    contact = SolidBodyContact(secondary, primary, items=[solid])

    kinematics = contact._extract()

    assert contact._extract() is kinematics


def test_normals_point_outwards():
    "The unit normal vectors of the faces of the bottom face of a block point down."

    mesh = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(3, 3, 3))
    secondary = surface(mesh)
    contact = SolidBodyContact(secondary, surface(mesh), penalty=1.0, self_contact=True)

    pair = contact.pairs[0]

    # the unit normal vectors are given in the array-layout of FElupe, i.e. with the
    # components on the leading and the faces on the trailing axis
    normals = pair.normals(mesh.points)
    center = mesh.points[pair.cells_faces].mean(axis=1)

    assert np.allclose(fem.math.norm(normals, axis=0), 1.0)
    assert np.allclose(normals[:, np.isclose(center[:, 2], 0.0)].T, [0.0, 0.0, -1.0])
    assert np.allclose(normals[:, np.isclose(center[:, 2], 1.0)].T, [0.0, 0.0, 1.0])


def test_separated_bodies_are_not_in_contact():
    "Two bodies which are far apart from each other are not in contact."

    bottom = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(3, 3, 3))
    top = fem.Cube(a=(0.2, 0.2, 3.0), b=(0.8, 0.8, 3.6), n=(3, 3, 3))
    mesh = fem.MeshContainer([bottom, top], merge=True).stack()

    field, solid = solid_body(mesh)
    contact = SolidBodyContact(surface(mesh), surface(mesh), items=[solid])

    force = contact.assemble.vector()
    stiffness = contact.assemble.matrix()

    assert contact.results.npoints_in_contact == 0
    assert force.nnz == 0
    assert stiffness.nnz == 0


def test_resize_of_vector_and_matrix():
    "The assembled contact vector and matrix are resized on demand."

    mesh = two_blocks()
    field, solid = solid_body(mesh)
    contact = SolidBodyContact(surface(mesh), surface(mesh), items=[solid])

    ndof = field[0].indices.dof.size
    force = contact.assemble.vector(resize=csr_matrix((ndof + 3, 1)))
    stiffness = contact.assemble.matrix(resize=csr_matrix((ndof + 3, ndof + 3)))

    assert force.shape == (ndof + 3, 1)
    assert stiffness.shape == (ndof + 3, ndof + 3)


def test_requires_items_or_penalty():
    "Either the items or the penalty stiffness must be given."

    mesh = two_blocks()

    with pytest.raises(ValueError):
        SolidBodyContact(surface(mesh), surface(mesh))


def test_requires_a_boundary_region():
    "A field on a boundary region is required for both contact surfaces."

    mesh = two_blocks()
    volume = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])

    with pytest.raises(TypeError):
        SolidBodyContact(volume, surface(mesh), penalty=1.0)


def test_requires_a_hexahedron_mesh():
    "Only boundary regions of hexahedron meshes are supported."

    mesh = fem.Rectangle(n=3)
    quad = fem.FieldContainer([fem.Field(fem.RegionQuadBoundary(mesh), dim=2)])

    with pytest.raises(NotImplementedError):
        SolidBodyContact(quad, quad, penalty=1.0)


def test_warns_for_a_single_body():
    "No contact is detected if all faces belong to the same body."

    mesh = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(3, 3, 3))

    with pytest.warns(UserWarning, match="same body"):
        SolidBodyContact(surface(mesh), surface(mesh), penalty=1.0)


def test_self_contact_does_not_warn():
    "A single body is a valid contact surface if self-contact is considered."

    mesh = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(3, 3, 3))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SolidBodyContact(surface(mesh), surface(mesh), penalty=1.0, self_contact=True)


def test_bodies_of_the_two_blocks():
    "The two blocks of the mesh are identified as two separate bodies."

    mesh = two_blocks()
    contact = SolidBodyContact(surface(mesh), surface(mesh), penalty=1.0)

    pair = contact.pairs[0]

    assert len(np.unique(pair.body)) == 2
    assert len(np.unique(pair.body_primary)) == 2


def test_single_body_without_self_contact():
    "A single body is never in contact with itself by default."

    mesh = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(3, 3, 3))
    field, solid = solid_body(mesh)

    with pytest.warns(UserWarning, match="same body"):
        contact = SolidBodyContact(surface(mesh), surface(mesh), items=[solid])

    assert contact.assemble.vector().nnz == 0
    assert contact.results.npoints_in_contact == 0


def test_single_body_with_self_contact():
    "The faces of a single body are searched if self-contact is considered."

    mesh = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(3, 3, 3))
    field, solid = solid_body(mesh)
    contact = SolidBodyContact(
        surface(mesh), surface(mesh), items=[solid], self_contact=True
    )

    # a block at rest does not penetrate itself
    assert contact.assemble.vector().nnz == 0
    assert contact.results.npoints_in_contact == 0


def test_estimated_penalty_with_a_multiplier():
    "The multiplier of an item is applied on its stiffness matrix."

    mesh = two_blocks()
    field, solid = solid_body(mesh)
    secondary, primary = surfaces(mesh, masked=False)

    pressure = fem.SolidBodyPressure(field=surface(mesh))
    contact = SolidBodyContact(secondary, primary, items=[solid, pressure])

    assert pressure.assemble.multiplier is not None
    assert contact.update_penalty() > 0
