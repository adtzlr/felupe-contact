import felupe as fem
import numpy as np
import pytest

from felupe_contact._element import (
    FACE_ELEMENTS,
    BatchedElement,
    face_element,
    quadratic_line,
)
from felupe_contact._solidbody_contact import (
    closest_point_projection,
    connected_bodies,
    unit_normals,
)

# the element formulation of the faces of a boundary region of a hexahedron mesh
QUAD = BatchedElement(fem.Quad())

# the element formulation of the faces of a boundary region of a quad mesh
LINE = BatchedElement(fem.element.Line())

# a unit quad in the x-y-plane, located at z = 0
QUAD_XY = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

# a unit line on the x-axis, located at y = 0
LINE_X = np.array([[0.0, 0.0], [1.0, 0.0]])

# the boundary regions of FElupe and the keyword arguments which convert a linear mesh
# into the mesh of the boundary region
BOUNDARY_REGIONS = [
    (fem.RegionQuadBoundary, "quad", {}),
    (fem.RegionQuadraticQuadBoundary, "quad8", dict(order=2, calc_midfaces=False)),
    (fem.RegionBiQuadraticQuadBoundary, "quad9", dict(order=2, calc_midfaces=True)),
    (fem.RegionHexahedronBoundary, "hexahedron", {}),
    (
        fem.RegionQuadraticHexahedronBoundary,
        "hexahedron20",
        dict(order=2, calc_midfaces=False, calc_midvolumes=False),
    ),
    (
        fem.RegionTriQuadraticHexahedronBoundary,
        "hexahedron27",
        dict(order=2, calc_midfaces=True, calc_midvolumes=True),
    ),
]


def faces(vertices, npairs=1):
    "Return a batch of faces with the given points."

    return np.broadcast_to(vertices[:, :, None], (*vertices.shape, npairs))


def boundary_region(cls, convert):
    "Return a boundary region of a rectangle- or a cube-mesh."

    mesh = fem.Cube(n=3) if "Hexahedron" in cls.__name__ else fem.Rectangle(n=3)

    if convert:
        mesh = fem.mesh.convert(mesh, **convert)

    return cls(mesh)


@pytest.mark.parametrize("cell_type", list(FACE_ELEMENTS.keys()))
def test_batched_element_matches_felupe(cell_type):
    "The batched shape functions and their derivatives match those of FElupe."

    element = FACE_ELEMENTS[cell_type]()
    batched = BatchedElement(element)

    rng = np.random.default_rng(seed=7)
    coordinates = rng.uniform(-1, 1, size=(batched.dim, 12))

    npoints, dim = batched.npoints, batched.dim
    shapes = [(npoints,), (npoints, dim), (npoints, dim, dim)]
    methods = ["function", "gradient", "hessian"]

    for method, shape in zip(methods, shapes):
        values = getattr(batched, method)(coordinates)
        reference = np.array(
            [
                np.asarray(getattr(element, method)(point)).reshape(shape)
                for point in coordinates.T
            ]
        )

        assert values.shape == (*shape, coordinates.shape[-1])
        assert np.allclose(values, np.moveaxis(reference, 0, -1))


@pytest.mark.parametrize("cell_type", list(FACE_ELEMENTS.keys()))
def test_batched_element_partition_of_unity(cell_type):
    "The shape functions sum up to one and their derivatives sum up to zero."

    element = BatchedElement(FACE_ELEMENTS[cell_type]())

    rng = np.random.default_rng(seed=1)
    coordinates = rng.uniform(-1, 1, size=(element.dim, 12))

    assert np.allclose(element.function(coordinates).sum(axis=0), 1.0)
    assert np.allclose(element.gradient(coordinates).sum(axis=0), 0.0)
    assert np.allclose(element.hessian(coordinates).sum(axis=0), 0.0)


@pytest.mark.parametrize("cell_type", list(FACE_ELEMENTS.keys()))
def test_batched_element_at_points(cell_type):
    "Each shape function is one at its own point and zero at the other points."

    element = BatchedElement(FACE_ELEMENTS[cell_type]())
    values = element.function(element.points.T)

    assert np.allclose(values, np.eye(element.npoints))


@pytest.mark.parametrize("cell_type", list(FACE_ELEMENTS.keys()))
def test_batched_element_hessian_by_central_differences(cell_type):
    "The hessians match central differences of the gradients."

    element = BatchedElement(FACE_ELEMENTS[cell_type]())

    rng = np.random.default_rng(seed=2)
    coordinates = rng.uniform(-0.8, 0.8, size=(element.dim, 5))
    hessian = element.hessian(coordinates)

    eps = 1e-6
    for direction in range(element.dim):
        step = np.zeros((element.dim, 1))
        step[direction] = eps

        forward = element.gradient(coordinates + step)
        backward = element.gradient(coordinates - step)

        assert np.allclose(hessian[:, direction], (forward - backward) / (2 * eps))


def test_batched_element_requires_polynomial_shape_functions():
    "An element with non-polynomial shape functions is not supported."

    class NonPolynomial(fem.element.Line):
        def function(self, rv):
            (r,) = rv
            return np.array([np.exp(r), np.exp(-r)])

    with pytest.raises(NotImplementedError):
        BatchedElement(NonPolynomial())


@pytest.mark.parametrize("cls, cell_type, convert", BOUNDARY_REGIONS)
def test_face_element(cls, cell_type, convert):
    "The shape functions of a face are the shape functions of a boundary region."

    region = boundary_region(cls, convert)
    element, index = face_element(region)

    assert region.mesh.cell_type == cell_type
    assert element.npoints == region.mesh.cells_faces.shape[1]
    assert element.dim == region.mesh.dim - 1

    # the points of a face are located at the same positions within the points of a
    # cell of the boundary region for all cells
    assert np.array_equal(region.mesh.cells[:, index], region.mesh.cells_faces)

    # the boundary quadrature points are located on the first face of a cell, i.e. the
    # in-plane coordinates are their leading components
    coordinates = region.quadrature.points[:, : element.dim].T

    assert np.allclose(region.h[index, :, 0], element.function(coordinates))
    assert np.allclose(
        region.dhdr[index, : element.dim, :, 0], element.gradient(coordinates)
    )


def test_face_element_requires_a_supported_cell_type():
    "A boundary region of an unsupported cell type is rejected."

    region = fem.RegionQuadBoundary(fem.Rectangle(n=3))
    region.mesh.cell_type = "triangle"

    with pytest.raises(NotImplementedError):
        face_element(region)


def test_face_element_requires_a_consistent_point_ordering():
    "The points of a face must be located at the same positions for all cells."

    region = fem.RegionHexahedronBoundary(fem.Cube(n=3))

    # swap two points of the faces of all cells but the first one
    cells_faces = region.mesh.cells_faces.copy()
    cells_faces[1:] = cells_faces[1:][:, [1, 0, 2, 3]]
    region.mesh.cells_faces = cells_faces

    with pytest.raises(ValueError):
        face_element(region)


def test_quadratic_line():
    "The quadratic line element is ordered as a `line3` cell, i.e. ends first."

    element = quadratic_line()

    assert np.allclose(element.points.ravel(), [-1.0, 1.0, 0.0])


def test_unit_normals():
    "The unit normal vectors are normalized and orthogonal to the in-plane vectors."

    rng = np.random.default_rng(seed=3)
    tangents = rng.uniform(-1, 1, size=(2, 3, 8))

    normals = unit_normals(tangents)

    assert np.allclose(fem.math.norm(normals, axis=0), 1.0)
    for tangent in tangents:
        assert np.allclose(fem.math.dot(normals, tangent, mode=(1, 1)), 0.0)


def test_unit_normals_2d():
    "The unit normal vector of an edge is orthogonal to its in-plane vector."

    rng = np.random.default_rng(seed=4)
    tangents = rng.uniform(-1, 1, size=(1, 2, 8))

    normals = unit_normals(tangents)

    assert np.allclose(fem.math.norm(normals, axis=0), 1.0)
    assert np.allclose(fem.math.dot(normals, tangents[0], mode=(1, 1)), 0.0)


def test_unit_normals_orientation():
    "A negative orientation flips the unit normal vectors."

    tangents = np.array([[[1.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]]])

    assert np.allclose(unit_normals(tangents), [[0.0], [0.0], [1.0]])
    assert np.allclose(unit_normals(tangents, orientation=-1.0), [[0.0], [0.0], [-1.0]])


def test_unit_normals_orientation_2d():
    "The normal vector of an edge along the x-axis points in negative y-direction."

    tangents = np.array([[[1.0], [0.0]]])

    assert np.allclose(unit_normals(tangents), [[0.0], [-1.0]])
    assert np.allclose(unit_normals(tangents, orientation=-1.0), [[0.0], [1.0]])


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
    coordinates, converged = closest_point_projection(points, faces(QUAD_XY), QUAD)

    assert np.allclose(coordinates, 0.0, atol=1e-6)
    assert np.all(converged)


def test_closest_point_projection_on_vertex():
    "A point above a vertex of a quad is projected on this vertex."

    points = np.array([[1.0], [1.0], [0.5]])
    coordinates, converged = closest_point_projection(points, faces(QUAD_XY), QUAD)

    assert np.allclose(coordinates, 1.0, atol=1e-6)
    assert np.all(converged)


def test_closest_point_projection_outside():
    "A point far away from a quad is projected outside of its natural coordinates."

    points = np.array([[5.0], [0.5], [0.0]])
    coordinates = closest_point_projection(points, faces(QUAD_XY), QUAD)[0]

    assert coordinates[0, 0] > 1.0


def test_closest_point_projection_batch():
    "The projections of a batch of points are evaluated independently."

    points = np.array([[0.5, 0.0, 1.0], [0.5, 0.0, 0.0], [1.0, 1.0, 1.0]])
    coordinates, converged = closest_point_projection(
        points, faces(QUAD_XY, npairs=3), QUAD
    )

    assert np.all(converged)
    assert np.allclose(coordinates[:, 0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(coordinates[:, 1], [-1.0, -1.0], atol=1e-6)
    assert np.allclose(coordinates[:, 2], [1.0, -1.0], atol=1e-6)


def test_closest_point_projection_degenerate():
    "A singular local equation system is regularized instead of raising an error."

    degenerate = np.zeros((4, 3, 1))
    coordinates = closest_point_projection(np.ones((3, 1)), degenerate, QUAD)[0]

    assert np.all(np.isfinite(coordinates))


def test_closest_point_projection_edge():
    "A point above an edge is projected on the edge, i.e. on a face of a quad mesh."

    points = np.array([[0.5, 0.0], [1.0, 1.0]])
    coordinates, converged = closest_point_projection(
        points, faces(LINE_X, npairs=2), LINE
    )

    assert np.all(converged)
    assert coordinates.shape == (1, 2)
    assert np.allclose(coordinates[0], [0.0, -1.0], atol=1e-6)


def test_closest_point_projection_edge_outside():
    "A point beside an edge is projected outside of its natural coordinates."

    points = np.array([[2.0], [1.0]])
    coordinates = closest_point_projection(points, faces(LINE_X), LINE)[0]

    assert coordinates[0, 0] > 1.0


def test_closest_point_projection_curved_edge():
    "A point above the center of a curved edge is projected on its center."

    # a quadratic line with an offset midpoint, i.e. a curved edge
    vertices = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.3]])
    points = np.array([[0.0], [1.0]])

    coordinates, converged = closest_point_projection(
        points, faces(vertices), BatchedElement(quadratic_line())
    )

    assert np.all(converged)
    assert np.allclose(coordinates, 0.0, atol=1e-6)
