# SPDX-License-Identifier: GPL-3.0-or-later

import felupe as fem
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from felupe_contact import SolidBodyContact
from felupe_contact._solidbody_contact import (
    ContactSurfacePair,
    closest_point_projection,
    connected_bodies,
    unit_normals,
)

# meshes of FElupe with all supported cell types and the keyword arguments which
# convert a linear mesh into the mesh of the cell type
MESHES = [
    ("quad", fem.Rectangle, {}),
    ("quad8", fem.Rectangle, dict(order=2, calc_midfaces=False)),
    ("quad9", fem.Rectangle, dict(order=2, calc_midfaces=True)),
    ("hexahedron", fem.Cube, {}),
    (
        "hexahedron20",
        fem.Cube,
        dict(order=2, calc_midfaces=False, calc_midvolumes=False),
    ),
    (
        "hexahedron27",
        fem.Cube,
        dict(order=2, calc_midfaces=True, calc_midvolumes=True),
    ),
]

BOUNDARY_REGIONS = {
    "quad": fem.RegionQuadBoundary,
    "quad8": fem.RegionQuadraticQuadBoundary,
    "quad9": fem.RegionBiQuadraticQuadBoundary,
    "hexahedron": fem.RegionHexahedronBoundary,
    "hexahedron20": fem.RegionQuadraticHexahedronBoundary,
    "hexahedron27": fem.RegionTriQuadraticHexahedronBoundary,
}


def surface_pair(cell_type, mesh_type, convert, n=4):
    "Return a surface pair of all faces on the outline of a mesh."

    mesh = mesh_type(n=n)
    if convert:
        mesh = fem.mesh.convert(mesh, **convert)

    field_type = fem.Field if mesh.dim == 3 else fem.FieldPlaneStrain
    region = BOUNDARY_REGIONS[cell_type](mesh)
    surface = fem.FieldContainer([field_type(region, dim=mesh.dim)])

    return mesh, ContactSurfacePair(surface, surface)


def two_blocks():
    "Return the mesh of a block which is placed on top of a bigger block."

    bottom = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(5, 5, 4))
    top = fem.Cube(a=(0.15, 0.15, 1.02), b=(0.85, 0.85, 1.62), n=(4, 4, 3))

    return fem.MeshContainer([bottom, top], merge=True).stack()


def c_shape():
    """Return the mesh of a C-shaped body, where the tip of the upper arm penetrates
    the lower arm, and the deformed coordinates of its points."""

    mesh = fem.Rectangle(a=(0, 0), b=(4, 2.1), n=(65, 35))
    centers = mesh.points[mesh.cells].mean(axis=1)
    slot = (centers[:, 0] > 1) & (centers[:, 1] > 1.0) & (centers[:, 1] < 1.1)
    mesh.update(cells=mesh.cells[~slot])
    mesh = mesh.expand(n=5, z=0.5)

    x = mesh.points.copy()
    upper = (x[:, 1] > 1.05) & (x[:, 0] > 1.5)
    x[upper, 1] -= 0.2 * (x[upper, 0] - 1.5) / 2.5

    return mesh, x


def kinematics(contact, x):
    "Evaluate the contact kinematics of the first surface pair."

    return contact.pairs[0].kinematics(
        x=x,
        max_distance=contact.max_distance,
        candidates=contact.candidates,
        tolerance=contact.tolerance,
        facing=contact.facing,
        self_contact=contact.self_contact,
    )


@pytest.mark.parametrize("cell_type, mesh_type, convert", MESHES)
def test_bounds_of_the_faces(cell_type, mesh_type, convert):
    """The bounds of the faces contain the faces on the extended range of the natural
    element coordinates."""

    mesh, pair = surface_pair(cell_type, mesh_type, convert)

    # distorted (curved and warped) faces
    rng = np.random.default_rng(seed=5)
    x = mesh.points + rng.uniform(-0.02, 0.02, size=mesh.points.shape)

    extent = 1.2
    vertices = pair.vertices(x, pair.cells_faces_primary)
    bounds = pair.bounds(vertices, extent=extent)

    # sample the faces on a fine grid of the extended range
    element = pair.element_primary
    axes = [np.linspace(-extent, extent, 9)] * element.dim
    grid = np.array([a.ravel() for a in np.meshgrid(*axes, indexing="ij")])

    h, dhdr, _ = element.evaluate(grid)
    points = np.einsum("as,aip->isp", h, vertices)

    offset = points - bounds["center"][:, None]
    height = np.einsum("isp,ip->sp", offset, bounds["normal"])

    eps = 1e-10
    assert np.all(points >= bounds["min"][:, None] - eps)
    assert np.all(points <= bounds["max"][:, None] + eps)
    assert np.all(np.linalg.norm(offset, axis=0) <= bounds["radius"] + eps)
    assert np.all(height <= bounds["upper"] + eps)
    assert np.all(height >= bounds["lower"] - eps)

    # the normal vector at the center is a unit normal vector
    assert np.allclose(np.linalg.norm(bounds["normal"], axis=0), 1.0)

    # the angle of the normal vectors w.r.t. the normal vector at the center
    tangents = np.einsum("aJs,aip->Jisp", dhdr, vertices)
    normals = unit_normals(tangents, pair.orientation_primary)
    cosine = np.einsum("isp,ip->sp", normals, bounds["normal"])
    angle = np.arccos(np.clip(cosine, -1, 1))

    assert np.all(angle <= bounds["angle"] + 1e-8)


def test_bounds_of_bent_faces():
    """The normal vectors of strongly distorted faces are turned by 90 degrees or more,
    then the angle is not bounded."""

    mesh, pair = surface_pair("hexahedron27", fem.Cube, MESHES[-1][-1])

    rng = np.random.default_rng(seed=5)
    x = mesh.points + rng.uniform(-0.1, 0.1, size=mesh.points.shape)

    vertices = pair.vertices(x, pair.cells_faces_primary)
    bounds = pair.bounds(vertices, extent=1.2)

    bent = bounds["angle"] >= np.pi / 2

    assert np.any(bent)
    assert np.all(bounds["angle"][bent] == np.pi)


def test_discarded_face_pairs_are_never_in_contact():
    """The cheap criteria of the contact search discard only face-pairs which are not
    in contact, i.e. the contact kinematics are equal without these criteria."""

    mesh = two_blocks()
    rng = np.random.default_rng(seed=3)

    x = mesh.points.copy()
    x[mesh.z > 1.0, 2] -= 0.08
    x += rng.uniform(-0.01, 0.01, size=x.shape)

    surface = fem.FieldContainer([fem.Field(fem.RegionHexahedronBoundary(mesh), dim=3)])
    contact = SolidBodyContact(surface, penalty=1.0)
    result = kinematics(contact, x)

    contact.pairs[0].face[:] = -1
    discard = ContactSurfacePair.discard

    try:
        ContactSurfacePair.discard = lambda self, point, face, *args: (point, face)
        reference = kinematics(contact, x)
    finally:
        ContactSurfacePair.discard = discard

    assert len(result["point"]) > 0
    for key in ["point", "face"]:
        np.testing.assert_array_equal(result[key], reference[key])

    assert np.allclose(result["gap"], reference["gap"])


def test_self_contact_with_neighbours():
    """The neighbours of a face occupy candidates of the tree-query for self-contact.
    Hence, their number is added to the number of candidates, i.e. the result is equal
    to the result of an (almost) exhaustive search."""

    mesh, x = c_shape()
    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

    contact = SolidBodyContact(solid, self_contact=True, max_distance=0.3)
    exhaustive = SolidBodyContact(
        solid, self_contact=True, max_distance=0.3, candidates=200
    )

    # the faces of a structured surface share points with up to 9 faces (incl. itself)
    assert contact.pairs[0].nneighbours >= 9

    result = kinematics(contact, x)
    reference = kinematics(exhaustive, x)

    assert len(result["point"]) > 0
    np.testing.assert_array_equal(result["point"], reference["point"])
    np.testing.assert_array_equal(result["face"], reference["face"])


def test_number_of_neighbours():
    "The maximum number of faces of the primary surface which share a point."

    mesh, pair = surface_pair("hexahedron", fem.Cube, {}, n=4)

    faces = pair.cells_faces
    shared = [
        np.sum(np.any(np.isin(pair.cells_faces_primary, face), axis=1))
        for face in faces
    ]

    assert pair.nneighbours == max(shared)


def test_previous_face_is_a_candidate():
    """The face of the previous evaluation is a candidate, even if it is not among the
    nearest faces of the tree-query."""

    # the lower body has a big and a small face on its top. the upper body is located
    # above the big face, but its integration points are closer to the center of the
    # small face
    lower = fem.Mesh(
        points=np.array(
            [[0, 0], [2, 0], [2.2, 0], [0, 1], [2, 1], [2.2, 1]], dtype=float
        ),
        cells=np.array([[0, 1, 4, 3], [1, 2, 5, 4]]),
        cell_type="quad",
    )
    upper = fem.Rectangle(a=(1.75, 1.02), b=(1.95, 1.2), n=2)
    mesh = fem.MeshContainer([lower, upper], merge=True).stack()

    def surface(mask):
        region = fem.RegionQuadBoundary(mesh, mask=mask)
        return fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])

    secondary = surface((mesh.y > 1.01) & (mesh.y < 1.03))
    primary = surface(np.isclose(mesh.y, 1.0))

    contact = SolidBodyContact(
        secondary, primary, penalty=1.0, candidates=1, max_distance=0.3
    )
    pair = contact.pairs[0]

    # the upper body penetrates the big face of the lower body
    x = mesh.points.copy()
    x[mesh.y > 1.01, 1] -= 0.07

    # the nearest face (by its center) is the small face, i.e. no contact is found
    assert kinematics(contact, x) is None

    # the big face is kept as the face of the previous evaluation
    left = mesh.points[pair.cells_faces_primary, 0].min(axis=1)
    big = np.flatnonzero(np.isclose(left, 0.0))
    pair.face[:] = big[0]

    result = kinematics(contact, x)

    assert len(result["point"]) == pair.dV.size
    assert np.all(result["face"] == big[0])
    assert np.allclose(result["gap"], -0.05)


def test_kinematics_are_contiguous():
    """The arrays of the contact kinematics are C-contiguous, which speeds up the
    assembly of the stiffness matrix."""

    mesh = two_blocks()
    surface = fem.FieldContainer([fem.Field(fem.RegionHexahedronBoundary(mesh), dim=3)])
    contact = SolidBodyContact(surface, surface, penalty=1.0, max_distance=0.4)

    x = mesh.points.copy()
    x[mesh.z > 1.01, 2] -= 0.05
    result = kinematics(contact, x)

    assert len(result["point"]) > 0
    assert all(value.flags["C_CONTIGUOUS"] for value in result.values())


def test_closest_point_projection_active_set():
    """The pairs of a batch, which converge at different rates, are projected like
    single pairs."""

    from felupe_contact._element import BatchedElement

    element = BatchedElement(fem.Quad())
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [1.0, 1.0, -0.1], [0.0, 1.0, 0.3]]
    )

    rng = np.random.default_rng(seed=4)
    points = np.vstack(
        [rng.uniform(-0.5, 1.5, size=(2, 6)), rng.uniform(-0.5, 2.0, size=(1, 6))]
    )
    faces = np.broadcast_to(vertices[:, :, None], (*vertices.shape, points.shape[1]))

    coordinates, converged = closest_point_projection(points, faces, element)

    for p in range(points.shape[1]):
        single = closest_point_projection(points[:, [p]], faces[..., [p]], element)
        assert np.allclose(coordinates[:, p], single[0][:, 0])
        assert converged[p] == single[1][0]


def test_connected_bodies_of_cells_with_different_numbers_of_points():
    "A list of point-connectivities with different numbers of points per cell."

    quads = np.array([[0, 1, 2, 3]])
    lines = np.array([[3, 4], [5, 6]])

    labels = connected_bodies([quads, lines], npoints=7)

    assert len(np.unique(labels[:5])) == 1
    assert labels[5] == labels[6]
    assert labels[5] != labels[0]


def test_assembly_by_the_sparsity_pattern():
    """The vector and the matrix of a surface pair are equal to an assembly in the
    coordinate format (COO)."""

    mesh = two_blocks()
    x = mesh.points.copy()
    x[mesh.z > 1.0, 2] -= 0.06

    surface = fem.FieldContainer([fem.Field(fem.RegionHexahedronBoundary(mesh), dim=3)])
    contact = SolidBodyContact(surface, penalty=1.0)
    pair = contact.pairs[0]
    kin = kinematics(contact, x)

    pressure, dpressure = contact.pressure(kin["gap"])
    dofs = pair.dofs(kin)
    size = (pair.ndof, pair.ndof)

    values = pair.vector(kin, -pressure)
    reference = csr_matrix(
        (values.T.ravel(), (dofs.ravel(), np.zeros(dofs.size, dtype=int))),
        shape=(pair.ndof, 1),
    )
    vector = pair.assemble_vector(kin, -pressure)

    assert np.allclose(vector.toarray(), reference.toarray())

    values = pair.matrix(kin, -pressure, -dpressure)
    rows = np.broadcast_to(dofs.T[:, None, :], values.shape)
    cols = np.broadcast_to(dofs.T[None, :, :], values.shape)
    reference = csr_matrix((values.ravel(), (rows.ravel(), cols.ravel())), shape=size)
    matrix = pair.assemble_matrix(kin, -pressure, -dpressure)

    assert np.allclose(matrix.toarray(), reference.toarray())
    assert np.allclose(matrix.toarray(), matrix.toarray().T)


def test_sparsity_pattern_is_reused():
    "The sparsity pattern is re-used as long as the face-pairs don't change."

    mesh = two_blocks()
    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)
    contact = SolidBodyContact(solid)

    field[0].values[mesh.z > 1.0, 2] = -0.06
    contact.assemble.matrix()
    dofs, pattern = contact._pattern

    # a slightly different penetration, the same face-pairs are in contact
    field[0].values[mesh.z > 1.0, 2] = -0.061
    contact.assemble.matrix()

    assert contact._pattern[1] is pattern

    # a lateral displacement, other face-pairs are in contact
    field[0].values[mesh.z > 1.0, 0] = 0.1
    contact.assemble.matrix()

    assert contact._pattern[1] is not pattern
