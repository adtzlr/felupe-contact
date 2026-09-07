import numpy as np

from felupe_contact._solidbody_contact import (
    closest_point_projection,
    connected_bodies,
    invert_2x2,
    shape_function_quad,
    solve_2x2,
)

# the natural element coordinates of the vertices of a bilinear quad
VERTICES = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

# a unit quad in the x-y-plane, located at z = 0
QUAD = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])


def test_shape_function_quad_at_vertices():
    "Each shape function is one at its own vertex and zero at the other vertices."

    h = shape_function_quad(VERTICES)[0]

    assert np.allclose(h, np.eye(4))


def test_shape_function_quad_partition_of_unity():
    "The shape functions sum up to one and their gradients sum up to zero."

    coordinates = np.random.default_rng(seed=0).uniform(-1, 1, size=(20, 2))
    h, dhdr = shape_function_quad(coordinates)

    assert np.allclose(h.sum(axis=-1), 1.0)
    assert np.allclose(dhdr.sum(axis=-2), 0.0)


def test_shape_function_quad_gradient():
    "The gradients match central differences of the shape functions."

    coordinates = np.random.default_rng(seed=1).uniform(-0.8, 0.8, size=(5, 2))
    dhdr = shape_function_quad(coordinates)[1]

    eps = 1e-6
    for direction in range(2):
        step = np.zeros(2)
        step[direction] = eps

        forward = shape_function_quad(coordinates + step)[0]
        backward = shape_function_quad(coordinates - step)[0]

        assert np.allclose(dhdr[..., direction], (forward - backward) / (2 * eps))


def test_solve_2x2():
    "The solution of a batch of 2x2 systems satisfies the equation system."

    rng = np.random.default_rng(seed=2)
    matrix = rng.uniform(1, 2, size=(10, 2, 2))
    vector = rng.uniform(-1, 1, size=(10, 2))

    solution = solve_2x2(matrix, vector)

    assert np.allclose(matrix @ solution[:, :, None], vector[:, :, None])


def test_solve_2x2_singular():
    "A singular system is regularized instead of raising a division by zero."

    matrix = np.zeros((1, 2, 2))
    vector = np.ones((1, 2))

    assert np.all(np.isfinite(solve_2x2(matrix, vector)))


def test_invert_2x2():
    "The inverse of a batch of symmetric 2x2 matrices gives the identity."

    matrix = np.array([[[4.0, 1.0], [1.0, 3.0]], [[2.0, 0.0], [0.0, 5.0]]])

    assert np.allclose(matrix @ invert_2x2(matrix), np.eye(2))


def test_invert_2x2_singular():
    "A singular matrix is regularized instead of raising a division by zero."

    assert np.all(np.isfinite(invert_2x2(np.zeros((1, 2, 2)))))


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

    points = np.array([[0.5, 0.5, 1.0]])
    coordinates, converged = closest_point_projection(points, QUAD[None])

    assert np.allclose(coordinates, 0.0, atol=1e-6)
    assert np.all(converged)


def test_closest_point_projection_on_vertex():
    "A point above a vertex of a quad is projected on this vertex."

    points = np.array([[1.0, 1.0, 0.5]])
    coordinates, converged = closest_point_projection(points, QUAD[None])

    assert np.allclose(coordinates, 1.0, atol=1e-6)
    assert np.all(converged)


def test_closest_point_projection_outside():
    "A point far away from a quad is projected outside of its natural coordinates."

    points = np.array([[5.0, 0.5, 0.0]])
    coordinates = closest_point_projection(points, QUAD[None])[0]

    assert coordinates[0, 0] > 1.0


def test_closest_point_projection_batch():
    "The projections of a batch of points are evaluated independently."

    points = np.array([[0.5, 0.5, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    vertices = np.broadcast_to(QUAD, (3, 4, 3))
    coordinates, converged = closest_point_projection(points, vertices)

    assert np.all(converged)
    assert np.allclose(coordinates[0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(coordinates[1], [-1.0, -1.0], atol=1e-6)
    assert np.allclose(coordinates[2], [1.0, -1.0], atol=1e-6)
