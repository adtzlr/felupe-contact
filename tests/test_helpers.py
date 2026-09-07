import felupe as fem
import numpy as np

from felupe_contact._solidbody_contact import (
    D2HDRDS,
    QUAD,
    closest_point_projection,
    connected_bodies,
    unit_normals,
)

# the natural element coordinates of the vertices of a bilinear quad, given in the
# array-layout of FElupe, i.e. with the coordinates on the leading axis
VERTICES = QUAD.points.T

# a unit quad in the x-y-plane, located at z = 0
QUAD_XY = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])


def faces(vertices, npairs=1):
    "Return a batch of quad faces with the given vertices."

    return np.broadcast_to(vertices[:, :, None], (4, 3, npairs))


def test_shape_function_quad_at_vertices():
    "Each shape function is one at its own vertex and zero at the other vertices."

    assert np.allclose(QUAD.function(VERTICES), np.eye(4))


def test_shape_function_quad_partition_of_unity():
    "The shape functions sum up to one and their gradients sum up to zero."

    coordinates = np.random.default_rng(seed=0).uniform(-1, 1, size=(2, 20))

    assert np.allclose(QUAD.function(coordinates).sum(axis=0), 1.0)
    assert np.allclose(QUAD.gradient(coordinates).sum(axis=0), 0.0)


def test_shape_function_quad_gradient():
    "The gradients match central differences of the shape functions."

    coordinates = np.random.default_rng(seed=1).uniform(-0.8, 0.8, size=(2, 5))
    dhdr = QUAD.gradient(coordinates)

    eps = 1e-6
    for direction in range(2):
        step = np.zeros((2, 1))
        step[direction] = eps

        forward = QUAD.function(coordinates + step)
        backward = QUAD.function(coordinates - step)

        assert np.allclose(dhdr[:, direction], (forward - backward) / (2 * eps))


def test_shape_function_quad_mixed_second_derivative():
    "Only the mixed second derivative of the shape functions is non-zero."

    hessian = QUAD.hessian(np.zeros(2))

    assert np.allclose(D2HDRDS, hessian[:, 0, 1])
    assert np.allclose(D2HDRDS, hessian[:, 1, 0])
    assert np.allclose(hessian[:, 0, 0], 0.0)
    assert np.allclose(hessian[:, 1, 1], 0.0)


def test_unit_normals():
    "The unit normal vectors are normalized and orthogonal to the in-plane vectors."

    rng = np.random.default_rng(seed=3)
    u = rng.uniform(-1, 1, size=(3, 8))
    v = rng.uniform(-1, 1, size=(3, 8))

    normals = unit_normals(u, v)

    assert np.allclose(fem.math.norm(normals, axis=0), 1.0)
    assert np.allclose(fem.math.dot(normals, u, mode=(1, 1)), 0.0)
    assert np.allclose(fem.math.dot(normals, v, mode=(1, 1)), 0.0)


def test_unit_normals_orientation():
    "A negative orientation flips the unit normal vectors."

    u = np.array([[1.0], [0.0], [0.0]])
    v = np.array([[0.0], [1.0], [0.0]])

    assert np.allclose(unit_normals(u, v), [[0.0], [0.0], [1.0]])
    assert np.allclose(unit_normals(u, v, orientation=-1.0), [[0.0], [0.0], [-1.0]])


def test_connected_bodies_single():
    "All points of a single cell belong to the same body."

    cells = np.array([[0, 1, 2, 3]])
    labels = connected_bodies(cells, npoints=4)

    assert len(np.unique(labels)) == 1


def test_connected_bodies_two():
    "Two cells without shared points belong to two different bodies."

    cells = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    labels = connected_bodies(cells, npoints=8)

    assert len(np.unique(labels[:4])) == 1
    assert len(np.unique(labels[4:])) == 1
    assert labels[0] != labels[4]


def test_connected_bodies_shared_point():
    "Two cells which share one point belong to the same body."

    cells = np.array([[0, 1, 2, 3], [3, 4, 5, 6]])
    labels = connected_bodies(cells, npoints=7)

    assert len(np.unique(labels)) == 1


def test_closest_point_projection_inside():
    "A point above the center of a quad is projected on its center."

    points = np.array([[0.5], [0.5], [1.0]])
    coordinates, converged = closest_point_projection(points, faces(QUAD_XY))

    assert np.allclose(coordinates, 0.0, atol=1e-6)
    assert np.all(converged)


def test_closest_point_projection_on_vertex():
    "A point above a vertex of a quad is projected on this vertex."

    points = np.array([[1.0], [1.0], [0.5]])
    coordinates, converged = closest_point_projection(points, faces(QUAD_XY))

    assert np.allclose(coordinates, 1.0, atol=1e-6)
    assert np.all(converged)


def test_closest_point_projection_outside():
    "A point far away from a quad is projected outside of its natural coordinates."

    points = np.array([[5.0], [0.5], [0.0]])
    coordinates = closest_point_projection(points, faces(QUAD_XY))[0]

    assert coordinates[0, 0] > 1.0


def test_closest_point_projection_batch():
    "The projections of a batch of points are evaluated independently."

    points = np.array([[0.5, 0.0, 1.0], [0.5, 0.0, 0.0], [1.0, 1.0, 1.0]])
    coordinates, converged = closest_point_projection(points, faces(QUAD_XY, npairs=3))

    assert np.all(converged)
    assert np.allclose(coordinates[:, 0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(coordinates[:, 1], [-1.0, -1.0], atol=1e-6)
    assert np.allclose(coordinates[:, 2], [1.0, -1.0], atol=1e-6)


def test_closest_point_projection_degenerate():
    "A singular local equation system is regularized instead of raising an error."

    degenerate = np.zeros((4, 3, 1))
    coordinates = closest_point_projection(np.ones((3, 1)), degenerate)[0]

    assert np.all(np.isfinite(coordinates))
