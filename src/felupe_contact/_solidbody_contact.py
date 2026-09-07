# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""

import warnings

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from felupe import Field, FieldAxisymmetric, FieldPlaneStrain
from felupe.math import cross, det, dot, dya, inv, norm, transpose
from felupe.mechanics import Assemble, Results

from ._element import face_element

# the field types which define the out-of-plane behaviour of a contact surface, given
# for the dimension of the mesh of a boundary region. in 2d, a field must state whether
# the out-of-plane direction is a plane strain or an axisymmetric one
FIELD_TYPES = {2: (FieldPlaneStrain, FieldAxisymmetric), 3: (Field,)}


def unit_normals(tangents, orientation=1.0):
    r"""Return the outward unit normal vectors of faces, spanned by their in-plane
    (tangent) vectors.

    Parameters
    ----------
    tangents : ndarray of shape (dim - 1, dim, ...)
        The in-plane vectors of the faces. A face of a three-dimensional mesh is
        spanned by two in-plane vectors, a face (edge) of a two-dimensional mesh by one.
    orientation : float or ndarray of shape (...), optional
        The orientation :math:`\pm 1` of the faces, i.e. the sign which turns the normal
        vectors outwards (default is 1.0).

    Returns
    -------
    ndarray of shape (dim, ...)
        The outward unit normal vectors of the faces.

    Notes
    -----
    The normal vector of a face of a three-dimensional mesh is the cross product
    :math:`\boldsymbol{a}_1 \times \boldsymbol{a}_2` of its two in-plane vectors. The
    normal vector of a face (edge) of a two-dimensional mesh is its in-plane vector,
    rotated by :math:`-90` degrees.
    """

    if len(tangents) == 1:
        (u,) = tangents
        normals = np.stack([u[1], -u[0]]) * orientation
    else:
        normals = cross(*tangents) * orientation

    return normals / norm(normals, axis=0)


def connected_bodies(cells, npoints):
    """Return the labels of the connected bodies for the points of a mesh. Two points
    belong to the same body if they are connected by a path of cells.

    Parameters
    ----------
    cells : ndarray of shape (ncells, npoints_per_cell)
        The point-connectivity of the cells of a mesh.
    npoints : int
        The number of points of the mesh.

    Returns
    -------
    ndarray of shape (npoints,)
        The label of the body of each point of the mesh. Points which are not connected
        to any cell are labelled individually.
    """

    # it is sufficient to connect the points of a cell to its first point: this results
    # in the same connected components as a fully-connected cell
    rows = np.repeat(cells[:, 0], cells.shape[1] - 1)
    cols = cells[:, 1:].ravel()

    graph = coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)), shape=(npoints, npoints)
    )

    return connected_components(graph, directed=False, return_labels=True)[1]


def closest_point_projection(points, vertices, element, maxiter=12, tol=1e-10):
    r"""Return the natural element coordinates of the closest-point projections of
    points onto the faces of a surface.

    Parameters
    ----------
    points : ndarray of shape (dim, npairs)
        The coordinates of the points to be projected.
    vertices : ndarray of shape (npoints_per_face, dim, npairs)
        The coordinates of the points of the faces.
    element : BatchedElement
        The element formulation of the faces.
    maxiter : int, optional
        The maximum number of local Newton iterations (default is 12).
    tol : float, optional
        The tolerance for the local Newton iterations (default is 1e-10).

    Returns
    -------
    coordinates : ndarray of shape (dim - 1, npairs)
        The natural element coordinates of the projected points.
    converged : ndarray of shape (npairs,)
        A mask with the pairs for which the projection converged.

    Notes
    -----
    The projection minimizes the squared distance between a point and a face, see Eq.
    :eq:`closest-point`.

    ..  math::
        :label: closest-point

        f(\boldsymbol{\xi}) = \frac{1}{2} \left(
            \boldsymbol{x} - \hat{\boldsymbol{x}}(\boldsymbol{\xi})
        \right) \cdot \left(
            \boldsymbol{x} - \hat{\boldsymbol{x}}(\boldsymbol{\xi})
        \right) \rightarrow \min

    The stationary condition is the orthogonality of the distance vector and the
    tangent vectors
    :math:`\boldsymbol{a}_\alpha = \partial \hat{\boldsymbol{x}} / \partial \xi^\alpha`.
    The Hessian :math:`a_{\alpha\beta} - \boldsymbol{d} \cdot \hat{\boldsymbol{x}},
    _{\alpha\beta}` of the objective function is not positive definite if a point is
    located far away from a face. In this case, the Gauss-Newton approximation
    :math:`a_{\alpha\beta}` is used instead, which is always positive definite. This
    ensures a descent direction and hence a robust iteration for all pairs.
    """

    coordinates = np.zeros((element.dim, points.shape[-1]))
    dcoordinates = np.zeros_like(coordinates)

    for _ in range(maxiter):
        h = element.function(coordinates)
        dhdr = element.gradient(coordinates)
        d2hdrdr = element.hessian(coordinates)

        x = np.einsum("ap,aip->ip", h, vertices)
        a = np.einsum("aJp,aip->Jip", dhdr, vertices)
        dadr = np.einsum("aJKp,aip->JKip", d2hdrdr, vertices)
        d = points - x

        # negative gradient of the objective function
        fun = dot(a, d, mode=(2, 1))

        # hessian of the objective function and its Gauss-Newton approximation
        metric = dot(a, transpose(a))
        hessian = metric - np.einsum("JKip,ip->JKp", dadr, d)

        # use the Gauss-Newton approximation if the hessian is not positive definite
        definite = (det(hessian) > 0) & (hessian[0, 0] > 0)
        hessian = np.where(definite, hessian, metric)

        # the determinant of a singular system is regularized, the affected pairs are
        # filtered out by the convergence-check
        determinant = det(hessian)
        np.copyto(determinant, 1.0, where=np.abs(determinant) < np.finfo(float).tiny)

        dcoordinates = dot(inv(hessian, determinant=determinant), fun, mode=(2, 1))

        # limit the step size (trust region) and the range of the natural element
        # coordinates: only projections inside (or close to) a face are used
        np.clip(dcoordinates, -1.0, 1.0, out=dcoordinates)
        coordinates += dcoordinates
        np.clip(coordinates, -2.0, 2.0, out=coordinates)

        if np.all(np.abs(dcoordinates) < tol):
            break

    return coordinates, np.all(np.abs(dcoordinates) < np.sqrt(tol), axis=0)


class ContactSurfacePair:
    """A pair of a secondary (slave) and a primary (master) contact surface with
    methods to evaluate the contact kinematics and to assemble the sparse contact
    vector and matrix contributions.

    Parameters
    ----------
    field : FieldContainer
        A field container with a displacement field, created on a boundary region. The
        weak form of the contact is integrated on the faces of this secondary surface.
    field_primary : FieldContainer
        A field container with a displacement field, created on a boundary region. The
        integration points of the secondary surface are projected on the faces of this
        primary surface.
    weight : float, optional
        A scale factor for the contributions of this surface pair (default is 1.0).

    Notes
    -----
    This class is used internally by :class:`~felupe_contact.SolidBodyContact`.

    A *face* of a boundary region of a three-dimensional mesh is a surface and a face of
    a two-dimensional mesh is an edge. All quantities are formulated for both cases:
    a face is spanned by ``dim - 1`` natural element coordinates, i.e. the metric and
    the curvature of a face are of shape ``(dim - 1, dim - 1)``.

    All arrays are evaluated in the array-layout of FElupe: the axes of the shape
    functions ``a``, of the components ``i`` and of the natural element coordinates
    ``J`` are the leading axes and the batch-axes are the trailing axes. Hence the
    quantities of the face-pairs ``p`` are given as ``h_ap``, ``n_ip`` or ``a_Jip``.
    This is the layout of the elements and of the functions of :mod:`felupe.math`, it
    broadcasts the scalar-valued quantities of the face-pairs without a reshape and it
    is faster than a layout with leading batch-axes.
    """

    def __init__(self, field, field_primary, weight=1.0):
        self.field = field
        self.field_primary = field_primary
        self.weight = weight

        for f in [field, field_primary]:
            if not hasattr(f.region, "normals"):
                raise TypeError(
                    "A field on a boundary region is required, e.g. created on a "
                    "`RegionHexahedronBoundary` or on a `RegionQuadBoundary`."
                )

        region = self.field.region
        region_primary = self.field_primary.region

        self.dim = region.mesh.dim

        if self.dim not in FIELD_TYPES:
            raise NotImplementedError(
                f"Only two- and three-dimensional meshes are supported, got a "
                f"{self.dim}d mesh."
            )

        if region_primary.mesh.cell_type != region.mesh.cell_type:
            raise TypeError(
                "The cell types of the meshes of both boundary regions must be equal, "
                f"got {region.mesh.cell_type} and {region_primary.mesh.cell_type}."
            )

        # in 2d, the out-of-plane behaviour must be defined by the type of the field
        types = FIELD_TYPES[self.dim]
        names = " or ".join([f"`{t.__name__}`" for t in types])

        for f in [field[0], field_primary[0]]:
            if type(f) not in types:
                raise TypeError(
                    f"A field of type {names} is required for a boundary region of a "
                    f"{self.dim}d mesh, got `{type(f).__name__}`."
                )

        if type(field[0]) is not type(field_primary[0]):
            raise TypeError(
                "The displacement fields of both contact surfaces must be of the same "
                f"type, got `{type(field[0]).__name__}` and "
                f"`{type(field_primary[0]).__name__}`."
            )

        # the element formulation of the faces of a boundary region and the indices of
        # the points of a face within the points of a cell of the boundary region
        self.element, index = face_element(region)
        self.element_primary = face_element(region_primary)[0]

        self.cells_faces = region.mesh.cells_faces
        self.cells_faces_primary = region_primary.mesh.cells_faces

        # the shape functions of the faces of the secondary surface, evaluated at the
        # integration points. these are the shape functions of the points of a face of
        # the boundary region and their values are equal for all cells
        self.h = np.ascontiguousarray(region.h[index, :, 0])

        # a field with the deformed coordinates of the points of the mesh, which are
        # interpolated at the integration points of the faces of the secondary surface
        self.deformed = Field(region, dim=self.dim, values=region.mesh.points)

        # differential length (2d) or area (3d) of the faces of the secondary surface
        # in the reference configuration, the weights of the quadrature scheme are
        # already included
        self.dV = region.dV
        self.ncells = self.dV.shape[1]
        self.ncells_primary = len(self.cells_faces_primary)

        # the differential area on which the contact potential is integrated. for an
        # axisymmetric field, this is the area of the surface of revolution
        self.dA = self._init_area(field[0])
        self.dA_primary = self._init_area(field_primary[0])

        # the orientation of the points of the faces of a boundary region is not
        # necessarily aligned with the outward unit normal vectors of the region
        self.orientation = self._init_orientation(region, self.element)
        self.orientation_primary = self._init_orientation(
            region_primary, self.element_primary
        )

        # characteristic size of the faces of both surfaces, i.e. the edge length of a
        # face of a three-dimensional mesh and the length of a face (edge) in 2d
        self.size = min(
            [
                r.dV.sum(axis=0).mean() ** (1 / (self.dim - 1))
                for r in [region, region_primary]
            ]
        )

        # the connected bodies of the mesh: two faces which belong to the same body
        # are never in contact, unless self-contact is considered
        labels = connected_bodies(
            np.vstack([region.mesh.cells, region_primary.mesh.cells]),
            region.mesh.npoints,
        )
        self.body = labels[self.cells_faces[:, 0]]
        self.body_primary = labels[self.cells_faces_primary[:, 0]]

        self.points = np.unique(self.cells_faces)
        self.points_primary = np.unique(self.cells_faces_primary)

        self.area = self.dA.sum()
        self.area_primary = self.dA_primary.sum()

        self.dof = self.field[0].indices.dof
        self.ndof = self.field[0].indices.shape[0]

        # the number of the degrees of freedom of a face, i.e. the components "i" of
        # the displacements of its points "a"
        self.nvars = self.cells_faces.shape[1] * self.dim
        self.nvars_primary = self.cells_faces_primary.shape[1] * self.dim

        # the face of the primary surface of the previous evaluation, which is used to
        # stabilize the face-assignment of the integration points
        self.face = np.full(self.dV.size, -1)

    @staticmethod
    def _init_area(field):
        """Return the differential area on which the contact potential of a surface is
        integrated.

        Parameters
        ----------
        field : Field
            A displacement field, created on a boundary region.

        Returns
        -------
        ndarray of shape (nquadraturepoints, ncells)
            The differential area of the faces of the surface.

        Notes
        -----
        For an axisymmetric field, the faces of a boundary region are the generators of
        the surfaces of revolution, hence the differential area of a face is the product
        of its differential length and the circumference :math:`2 \\pi R` of the circle
        at its radial coordinate :math:`R`. In all other cases, the differential area of
        a face is the differential area (3d) or length (2d) of the boundary region.
        """

        if isinstance(field, FieldAxisymmetric):
            return 2 * np.pi * field.radius * field.region.dV

        return field.region.dV

    def _init_orientation(self, region, element):
        "Return the orientation of the faces w.r.t. the outward unit normal vectors."

        vertices = self.vertices(region.mesh.points, region.mesh.cells_faces)
        normals = unit_normals(self.tangents(vertices, element))

        # the outward unit normal vectors of the boundary region, evaluated as the mean
        # of all quadrature points of each face, define the orientation of a face. the
        # normal vectors of a boundary region with `ensure_3d=True` are padded with a
        # zero out-of-plane component
        reference = region.normals[: self.dim].mean(axis=1)

        return np.sign(dot(normals, reference, mode=(1, 1)))

    @staticmethod
    def vertices(x, cells_faces):
        """Return the coordinates of the points of faces.

        Parameters
        ----------
        x : ndarray of shape (npoints, dim)
            The coordinates of all points of the mesh.
        cells_faces : ndarray of shape (ncells, npoints_per_face)
            The point-connectivity of the faces.

        Returns
        -------
        ndarray of shape (npoints_per_face, dim, ncells)
            The coordinates of the points of the faces.
        """

        return np.ascontiguousarray(x[cells_faces].transpose(1, 2, 0))

    @staticmethod
    def tangents(vertices, element, coordinates=None):
        """Return the in-plane (tangent) vectors of faces, evaluated at the given
        natural element coordinates.

        Parameters
        ----------
        vertices : ndarray of shape (npoints_per_face, dim, ncells)
            The coordinates of the points of the faces.
        element : BatchedElement
            The element formulation of the faces.
        coordinates : ndarray of shape (dim - 1, ncells) or None, optional
            The natural element coordinates at which the in-plane vectors are evaluated
            (default is None). If None, the center of a face is used.

        Returns
        -------
        ndarray of shape (dim - 1, dim, ncells)
            The in-plane vectors of the faces.
        """

        if coordinates is None:
            coordinates = np.zeros((element.dim, vertices.shape[-1]))

        return np.einsum("aJp,aip->Jip", element.gradient(coordinates), vertices)

    def normals(self, x):
        """Return the outward unit normal vectors of the faces of the secondary
        surface, evaluated at the deformed coordinates of their points.

        Parameters
        ----------
        x : ndarray of shape (npoints, dim)
            The deformed coordinates of all points of the mesh.

        Returns
        -------
        ndarray of shape (dim, ncells)
            The outward unit normal vectors of the faces.
        """

        vertices = self.vertices(x, self.cells_faces)
        tangents = self.tangents(vertices, self.element)

        return unit_normals(tangents, self.orientation)

    def kinematics(
        self, x, max_distance, candidates, tolerance, facing, self_contact, workers=1
    ):
        r"""Return the contact kinematics, evaluated at the integration points of the
        faces of the secondary surface.

        Parameters
        ----------
        x : ndarray of shape (npoints, dim)
            The deformed coordinates of all points of the mesh.
        max_distance : float
            The maximum distance between an integration point and a face of the primary
            surface which is considered by the contact search.
        candidates : int
            The number of candidate faces of the primary surface per integration point
            of the secondary surface.
        tolerance : float
            The tolerance for the natural element coordinates of the projected points.
            A projection is only valid if the coordinates are within
            ``[-1 - tolerance, 1 + tolerance]``.
        facing : float
            The minimum opposition of the outward unit normal vectors of a face-pair. A
            projection is only valid if
            :math:`\boldsymbol{n} \cdot \boldsymbol{n}_{primary} < -facing`.
        self_contact : bool
            Flag to consider face-pairs which belong to the same body of the mesh.
        workers : int, optional
            The number of workers used for the tree-query (default is 1).

        Returns
        -------
        dict or None
            A dict with the contact kinematics of the active integration points. None
            is returned if no integration point is in contact.
        """

        # deformed coordinates of the integration points of the secondary surface
        self.deformed.values = x
        points = self.deformed.interpolate().reshape(self.dim, -1)

        # deformed coordinates of the points of the faces of the primary surface
        vertices = self.vertices(x, self.cells_faces_primary)

        # broad-phase contact search: find the nearest faces of the primary surface by
        # a tree-query on the face centers. the tree operates on point-arrays with the
        # coordinates on the trailing axis
        center = vertices.mean(axis=0)
        radius = norm(vertices - center, axis=1).max(axis=0)

        # the tree-query is carried out per body of the primary surface, where the
        # faces of the own body of an integration point are skipped. otherwise the
        # faces of the own body, which are always the closest ones, would occupy the
        # candidates of a query and hide the faces of the other bodies. this is
        # essential if the contact surfaces are not restricted to the region of
        # interest, e.g. if all faces on the outline of a mesh are used
        npoints = points.shape[-1]
        if self_contact:
            groups = [(np.arange(npoints), np.arange(self.ncells_primary))]
        else:
            body = np.tile(self.body, npoints // self.ncells)
            groups = [
                (np.flatnonzero(body != b), np.flatnonzero(self.body_primary == b))
                for b in np.unique(self.body_primary)
            ]

        point, face = [], []
        for rows, cols in groups:
            if len(rows) == 0 or len(cols) == 0:
                continue

            tree = cKDTree(center[:, cols].T)
            k = min(candidates, len(cols))
            distance, nearest = tree.query(points[:, rows].T, k=k, workers=workers)

            distance = distance.reshape(len(rows), k)
            nearest = cols[nearest.reshape(len(rows), k)]

            # discard pairs which are too far away
            mask = distance <= radius[nearest] + max_distance

            point.append(np.broadcast_to(rows[:, None], mask.shape)[mask])
            face.append(nearest[mask])

        empty = np.zeros(0, dtype=int)
        point = np.concatenate(point) if point else empty
        face = np.concatenate(face) if face else empty

        if len(point) == 0:
            return None

        # narrow-phase contact search: closest-point projection
        vertices_face = vertices[..., face]
        coordinates, converged = closest_point_projection(
            points[:, point], vertices_face, self.element_primary
        )

        h = self.element_primary.function(coordinates)
        dhdr = self.element_primary.gradient(coordinates)
        d2hdrdr = self.element_primary.hessian(coordinates)

        xp = np.einsum("ap,aip->ip", h, vertices_face)
        a = np.einsum("aJp,aip->Jip", dhdr, vertices_face)
        dadr = np.einsum("aJKp,aip->JKip", d2hdrdr, vertices_face)

        normal = unit_normals(a, self.orientation_primary[face])

        d = points[:, point] - xp
        gap = dot(d, normal, mode=(1, 1))

        # the metric and the curvature of the primary surface. the modified metric
        # H = a - g * kappa is positive definite as long as the penetration is smaller
        # than the radius of curvature of the primary surface. only in this case, the
        # projection is a (local) minimum of the distance
        metric = dot(a, transpose(a))
        curvature = np.einsum("JKip,ip->JKp", dadr, normal)

        H = metric - gap * curvature
        minimum = (det(H) > 0) & (H[0, 0] > 0)

        # faces which share at least one point are neighbours and must not be in
        # contact. this also removes the projections of a face on itself, which occur
        # if the masks of both boundary regions are overlapping
        cell = point % self.ncells
        cells = self.cells_faces[cell]
        neighbour = np.any(
            cells[:, :, None] == self.cells_faces_primary[face][:, None, :],
            axis=(1, 2),
        )

        # two faces are only able to touch each other if their outward unit normal
        # vectors are opposed. without this criterion, a face detects another face,
        # which is located around a corner, as a valid contact partner. this occurs if
        # the contact surfaces are not restricted to the region of interest, e.g. if
        # all faces on the outline of a mesh are used
        opposed = dot(self.normals(x)[:, cell], normal, mode=(1, 1)) < -facing

        # the face of the previous evaluation is released with a doubled tolerance.
        # this hysteresis prevents an oscillating activation of integration points
        # which are projected near the boundary of the primary surface
        previous = self.face[point] == face
        released = np.where(previous, 2 * tolerance, tolerance)

        # a projection is only valid if it is located inside a face of the primary
        # surface and if the (signed) distance is within the search distance
        inside = np.all(np.abs(coordinates) <= 1 + released, axis=0)
        valid = (
            converged
            & minimum
            & inside
            & opposed
            & ~neighbour
            & (gap < 0)
            & (gap > -max_distance)
        )

        if not np.any(valid):
            return None

        # if an integration point is projected on more than one face, then the closest
        # face is used. the distance is measured to the closest point which is located
        # inside a face, i.e. with clipped natural element coordinates. this is
        # essential: a criterion which is based on the gap is ambiguous for integration
        # points which are located near the edges of the faces of the primary surface
        # and leads to an oscillating face-assignment between the iterations
        hc = self.element_primary.function(np.clip(coordinates, -1.0, 1.0))
        distance = norm(
            points[:, point] - np.einsum("ap,aip->ip", hc, vertices_face), axis=0
        )

        score = np.where(valid, distance, np.inf)

        # the face of the previous evaluation is kept as long as the projection is
        # still located inside this face. this hysteresis is essential for the
        # convergence: without it, the face-assignment of an integration point, which
        # is located on an edge or on a point of the primary surface, oscillates
        # between the iterations of the Newton-Raphson method
        score[valid & previous] = -1.0

        order = np.lexsort((score, point))

        best = np.ones(len(order), dtype=bool)
        best[1:] = point[order][1:] != point[order][:-1]
        best = order[best]
        best = best[valid[best]]

        point, face = point[best], face[best]
        gap, normal, a = gap[best], normal[..., best], a[..., best]
        metric, curvature = metric[..., best], curvature[..., best]
        h, dhdr = h[..., best], dhdr[..., best]

        # store the face-assignment for the next evaluation
        self.face[:] = -1
        self.face[point] = face

        return {
            "point": point,
            "face": face,
            "gap": gap,
            "normal": normal,
            "tangents": a,
            "metric": metric,
            "curvature": curvature,
            "h": h,
            "dhdr": dhdr,
        }

    def variations_gap(self, kinematics):
        r"""Return the variation of the gap w.r.t. the displacements of the points of a
        face-pair.

        Parameters
        ----------
        kinematics : dict
            The contact kinematics, see :meth:`~ContactSurfacePair.kinematics`.

        Returns
        -------
        list of ndarray of shape (nvars, npairs)
            The variation of the gap :math:`\delta g = \boldsymbol{b} \cdot \delta
            \boldsymbol{u}` for the secondary and the primary surface.

        Notes
        -----
        The variation of the gap does not include the variation of the unit normal
        vector nor the variation of the projected coordinates, because both are
        orthogonal to the gap vector :math:`\boldsymbol{d} = g\ \boldsymbol{n}`.

        ..  math::

            \delta g = \left(
                \delta \boldsymbol{u} - \delta \bar{\boldsymbol{u}}
            \right) \cdot \boldsymbol{n}
        """

        point = kinematics["point"]
        normal = kinematics["normal"]
        npairs = len(point)

        # shape functions of the secondary (h) and of the primary (hp) faces
        h = self.h[:, point // self.ncells]
        hp = kinematics["h"]

        # the degrees of freedom of a face are the components "i" of the displacements
        # of its points "a", hence the dyadic products of the shape functions and the
        # vector-valued quantities are reshaped to a flat dof-axis
        return [
            dya(h, normal, mode=1).reshape(self.nvars, npairs),
            -dya(hp, normal, mode=1).reshape(self.nvars_primary, npairs),
        ]

    def variations(self, kinematics):
        r"""Return the variations of the gap and of the surface quantities of the
        primary surface w.r.t. the displacements of the points of a face-pair.

        Parameters
        ----------
        kinematics : dict
            The contact kinematics, see :meth:`~ContactSurfacePair.kinematics`.

        Returns
        -------
        b : list of ndarray of shape (nvars, npairs)
            The variation of the gap :math:`\delta g = \boldsymbol{b} \cdot \delta
            \boldsymbol{u}` for the secondary and the primary surface.
        A : ndarray of shape (dim - 1, nvars_primary, npairs)
            The variation :math:`A_\alpha = \boldsymbol{n} \cdot \delta
            \boldsymbol{a}_\alpha` of the primary surface.
        B : list of ndarray of shape (dim - 1, nvars, npairs)
            The variation :math:`B_\alpha = \boldsymbol{a}_\alpha \cdot \delta
            \boldsymbol{d}` for the secondary and the primary surface.
        """

        point = kinematics["point"]
        normal = kinematics["normal"]
        tangents = kinematics["tangents"]

        npairs = len(point)
        nsurface = self.dim - 1

        h = self.h[:, point // self.ncells]
        hp = kinematics["h"]
        dhdr = kinematics["dhdr"]

        b = self.variations_gap(kinematics)
        B = [
            np.einsum("ap,Jip->Jaip", h, tangents).reshape(
                nsurface, self.nvars, npairs
            ),
            -np.einsum("ap,Jip->Jaip", hp, tangents).reshape(
                nsurface, self.nvars_primary, npairs
            ),
        ]
        A = np.einsum("aJp,ip->Jaip", dhdr, normal).reshape(
            nsurface, self.nvars_primary, npairs
        )

        return b, A, B

    def _indices(self, kinematics):
        "Return the row- and column-indices of the dofs of the face-pairs."

        cell = kinematics["point"] % self.ncells

        return (
            self.dof[self.cells_faces[cell]].reshape(-1, self.nvars),
            self.dof[self.cells_faces_primary[kinematics["face"]]].reshape(
                -1, self.nvars_primary
            ),
        )

    def assemble_vector(self, kinematics, gradient):
        r"""Return the assembled sparse contact force vector.

        Parameters
        ----------
        kinematics : dict
            The contact kinematics, see :meth:`~ContactSurfacePair.kinematics`.
        gradient : ndarray of shape (npairs,)
            The first derivative :math:`\Phi'(g)` of the contact potential w.r.t. the
            gap, i.e. the negative contact pressure.

        Returns
        -------
        scipy.sparse.csr_matrix
            The assembled sparse contact force vector.

        Notes
        -----
        The contact traction is integrated on the faces of the secondary surface, see
        Eq. :eq:`contact-weak-form`. The equal and opposite contribution of the primary
        surface is evaluated at the projected points.

        ..  math::
            :label: contact-weak-form

            \delta \Pi_c = \int_{\Gamma} \Phi'(g) \left(
                \delta \boldsymbol{u} - \delta \bar{\boldsymbol{u}}
            \right) \cdot \boldsymbol{n}\ d\Gamma
        """

        q, c = np.divmod(kinematics["point"], self.ncells)
        dA = self.weight * self.dA[q, c] * gradient

        force = csr_matrix((self.ndof, 1))

        for b, rows in zip(self.variations_gap(kinematics), self._indices(kinematics)):
            force += self._assemble(dA * b, rows)

        return force

    def assemble_matrix(self, kinematics, gradient, hessian, geometric=True):
        r"""Return the assembled sparse contact stiffness matrix.

        Parameters
        ----------
        kinematics : dict
            The contact kinematics, see :meth:`~ContactSurfacePair.kinematics`.
        gradient : ndarray of shape (npairs,)
            The first derivative :math:`\Phi'(g)` of the contact potential w.r.t. the
            gap, i.e. the negative contact pressure.
        hessian : ndarray of shape (npairs,)
            The second derivative :math:`\Phi''(g)` of the contact potential w.r.t. the
            gap.
        geometric : bool, optional
            Flag to add the geometric part of the contact stiffness matrix (default is
            True).

        Returns
        -------
        scipy.sparse.csr_matrix
            The assembled sparse contact stiffness matrix.

        Notes
        -----
        The linearization of the variation of the gap is given in Eq.
        :eq:`contact-linearization` with the modified metric
        :math:`H_{\alpha\beta} = a_{\alpha\beta} - g\ \kappa_{\alpha\beta}` and
        :math:`M^{\alpha\beta} = a^{\alpha\gamma} \kappa_{\gamma\delta} H^{\delta\beta}`
        . The resulting stiffness matrix is symmetric.

        ..  math::
            :label: contact-linearization

            \Delta \delta g = &-H^{\alpha\beta} \left(
                A_\alpha \Delta B_\beta + B_\alpha \Delta A_\beta
            \right)

            &- g\ H^{\alpha\beta} A_\alpha \Delta A_\beta
             - M^{\alpha\beta} B_\alpha \Delta B_\beta
        """

        gap = kinematics["gap"]
        metric = kinematics["metric"]
        curvature = kinematics["curvature"]

        q, c = np.divmod(kinematics["point"], self.ncells)
        dA = self.weight * self.dA[q, c]

        # inverse of the modified metric and the curvature-related fourth-order term.
        # both the metric and the modified metric are positive definite, this is
        # ensured by the contact search
        inverse_metric = inv(metric)
        inverse_H = inv(metric - gap * curvature)
        M = dot(dot(inverse_metric, curvature), inverse_H)

        b, A, B = self.variations(kinematics)

        # material parts of the blocks of the stiffness matrix
        Kss = hessian * dya(b[0], b[0], mode=1)
        Ksm = hessian * dya(b[0], b[1], mode=1)
        Kmm = hessian * dya(b[1], b[1], mode=1)

        if geometric:
            HA = dot(inverse_H, A)
            MB = [dot(M, Bi) for Bi in B]

            # geometric part of the secondary-secondary block
            Kss -= gradient * np.einsum("Jip,Jjp->ijp", MB[0], B[0])

            # geometric part of the secondary-primary coupling block
            Ksm -= gradient * (
                np.einsum("Jip,Jjp->ijp", B[0], HA)
                + np.einsum("Jip,Jjp->ijp", MB[0], B[1])
            )

            # geometric part of the primary-primary block
            T = np.einsum("Jip,Jjp->ijp", HA, B[1])
            Kmm -= gradient * (
                T
                + transpose(T)
                + gap * np.einsum("Jip,Jjp->ijp", HA, A)
                + np.einsum("Jip,Jjp->ijp", MB[1], B[1])
            )

        rows, cols = self._indices(kinematics)
        Ksm = self._assemble(dA * Ksm, rows, cols)

        return (
            self._assemble(dA * Kss, rows, rows)
            + Ksm
            + Ksm.T
            + self._assemble(dA * Kmm, cols, cols)
        )

    def _assemble(self, values, rows, cols=None):
        """Return a sparse vector or matrix, assembled from the dense sub-vectors or
        sub-matrices of the face-pairs. The values are given in the array-layout of
        FElupe, i.e. with the face-pairs on the trailing axis.

        Parameters
        ----------
        values : ndarray of shape (..., npairs)
            The dense sub-vectors or sub-matrices of the face-pairs.
        rows : ndarray of shape (npairs, ...)
            The row-indices of the degrees of freedom of the face-pairs.
        cols : ndarray of shape (npairs, ...) or None, optional
            The column-indices of the degrees of freedom of the face-pairs. If None, a
            sparse vector is assembled (default is None).

        Returns
        -------
        scipy.sparse.csr_matrix
            The assembled sparse vector or matrix.
        """

        # move the axis of the face-pairs to the front, as it is done in the assembly
        # of an integral form
        values = np.moveaxis(values, -1, 0)

        if cols is None:
            return csr_matrix(
                (values.ravel(), (rows.ravel(), np.zeros(rows.size, dtype=int))),
                shape=(self.ndof, 1),
            )

        return csr_matrix(
            (
                values.ravel(),
                (
                    np.broadcast_to(rows[:, :, None], values.shape).ravel(),
                    np.broadcast_to(cols[:, None, :], values.shape).ravel(),
                ),
            ),
            shape=(self.ndof, self.ndof),
        )


class SolidBodyContact:
    r"""A frictionless contact between the surfaces of two solid bodies, formulated for
    three-dimensional, plane strain and axisymmetric problems.

    Parameters
    ----------
    field : FieldContainer
        A field container with a displacement field, created on a boundary region of
        the secondary (slave) surface, e.g. on a
        :class:`~felupe.RegionHexahedronBoundary` or on a
        :class:`~felupe.RegionQuadBoundary`. The weak form of the contact is integrated
        on the faces of this surface.
    field_primary : FieldContainer
        A field container with a displacement field, created on a boundary region of
        the primary (master) surface. The integration points of the secondary surface
        are projected on the faces of this surface.
    items : list of SolidBody or None, optional
        A list of items which are used to estimate the penalty stiffness (default is
        None). If None, ``penalty`` must be given.
    penalty : float or None, optional
        The penalty stiffness :math:`\epsilon` as contact traction per unit penetration
        (default is None). If None, the penalty stiffness is estimated from the mean
        stiffness of the degrees of freedom on both contact surfaces, see Eq.
        :eq:`contact-penalty`.
    penalty_scale : float, optional
        A scale factor which is applied on the estimated penalty stiffness (default is
        10.0). This has no effect if ``penalty`` is given. Increase this factor to
        reduce the penetration, decrease it if the Newton-Raphson method does not
        converge.
    smoothing : float or None, optional
        The length :math:`\delta` of the transition zone of the regularized penalty
        law, see Eq. :eq:`contact-pressure` (default is None). If None, a fraction of
        the characteristic size of the faces of the contact surfaces is used. A value
        of zero deactivates the regularization.
    two_pass : bool, optional
        Flag to evaluate the contact twice with exchanged roles of the surfaces, each
        with half of the contact potential (default is False). This removes the bias
        which is introduced by the choice of the secondary surface at the expense of a
        doubled evaluation time.
    max_distance : float or None, optional
        The maximum distance between an integration point of the secondary surface and
        a face of the primary surface which is considered by the contact search
        (default is None). If None, five times the characteristic size of the faces of
        the contact surfaces is used. This limits the maximum detectable penetration.
        It must be smaller than the distance between two opposed faces which are not
        able to touch, e.g. the distance between the top and the bottom face of a body,
        because such a face-pair is detected as a deeply penetrated one otherwise.
        Decrease it for a coarse mesh of a small body, where the default is larger than
        the body itself.
    candidates : int, optional
        The number of candidate faces of the primary surface which are evaluated per
        integration point of the secondary surface (default is 8).
    tolerance : float, optional
        The relative tolerance for the natural element coordinates of the projected
        points (default is 0.1). A projection is valid if its coordinates are within
        ``[-1 - tolerance, 1 + tolerance]``.
    facing : float, optional
        The minimum opposition of the outward unit normal vectors of a face-pair
        (default is 0.1). Two faces are only able to touch each other if
        :math:`\boldsymbol{n} \cdot \boldsymbol{n}_{primary} < -facing`, i.e. if
        their outward unit normal vectors are opposed. A value of -1.0 deactivates this
        criterion.
    self_contact : bool, optional
        Flag to consider face-pairs which belong to the same body of the mesh (default
        is False). The bodies of a mesh are identified by their point-connectivity: two
        faces belong to the same body if they are connected by a path of cells.
    geometric_stiffness : bool, optional
        Flag to add the geometric part of the contact stiffness matrix (default is
        True). This is required for a quadratic rate of convergence.

    Attributes
    ----------
    penalty : float or None
        The penalty stiffness. This is None until it is estimated on the first assembly.
    results : Results
        The results of the contact, e.g. the gap and the contact pressure at the
        integration points of the secondary surface.

    Notes
    -----
    The contact constraints are enforced by a penalty regularization of the contact
    potential :math:`\Pi_c`, which is integrated on the faces of the secondary surface,
    see Eq. :eq:`contact-potential`. This is a segment-to-segment (surface-to-surface)
    formulation: the contact tractions are evaluated at the integration points of the
    secondary surface and are not lumped to its points.

    ..  math::
        :label: contact-potential

        \Pi_c = \int_\Gamma \Phi(g)\ d\Gamma

    The gap :math:`g` is evaluated by a closest-point projection of the deformed
    coordinates :math:`\boldsymbol{x}` of the integration points of the secondary
    surface onto the deformed primary surface, see Eq. :eq:`contact-gap`. The projected
    coordinates :math:`\bar{\boldsymbol{x}}` and the outward unit normal vector
    :math:`\boldsymbol{n}` of the primary surface are evaluated at the natural element
    coordinates of the projection.

    ..  math::
        :label: contact-gap

        g = \left( \boldsymbol{x} - \bar{\boldsymbol{x}} \right) \cdot \boldsymbol{n}

    The contact pressure :math:`p = -\Phi'(g)` is a regularized penalty law with a
    smooth transition of length :math:`\delta`, see Eq. :eq:`contact-pressure`. In
    contrast to the non-regularized penalty law, both the contact pressure and its
    derivative w.r.t. the gap are continuous. This removes the jump of the tangent
    stiffness at the activation of a contact and hence improves the rate of convergence
    of the Newton-Raphson method significantly.

    ..  math::
        :label: contact-pressure

        p(g) = \begin{cases}
            -\epsilon \left( g + \dfrac{\delta}{2} \right) & g \le -\delta \\
            \epsilon\ \dfrac{g^2}{2 \delta} & -\delta < g < 0 \\
            0 & g \ge 0
        \end{cases}

    If no penalty stiffness is given, it is estimated from the mean of the diagonal
    entries :math:`\bar{k}` of the stiffness matrices of the given ``items``, evaluated
    on the degrees of freedom of the points of a contact surface, and the mean area
    :math:`\bar{a} = A / n_{points}` per point of this surface, see Eq.
    :eq:`contact-penalty`. The softer of both contact surfaces is decisive. This
    estimate scales with :math:`E / h` and hence requires no manual tuning, neither for
    rubber-to-rubber nor for rubber-to-metal contacts.

    ..  math::
        :label: contact-penalty

        \epsilon = \text{scale} \cdot \min{\left(
            \frac{\bar{k}}{\bar{a}},\ \frac{\bar{k}_{primary}}{\bar{a}_{primary}}
        \right)}

    A face-pair is only considered by the contact search if it is able to touch: faces
    which share at least one point are neighbours and are never in contact, faces which
    belong to the same body are only in contact if ``self_contact=True`` and the
    outward unit normal vectors of both faces must be opposed, see ``facing``. Without
    these criteria, a face would detect another face of its own body, which is located
    on the opposite side of the body or around a corner, as a valid contact partner.

    All boundary regions of FElupe are supported: the faces of a boundary region of a
    three-dimensional mesh are (bi-) linear or quadratic quads and the faces of a
    two-dimensional mesh are linear or quadratic lines (edges).

    ..  list-table:: The faces of the boundary regions of FElupe
        :header-rows: 1
        :widths: 70 30

        *   - Boundary region
            - Face
        *   - :class:`~felupe.RegionQuadBoundary`
            - ``line``
        *   - :class:`~felupe.RegionQuadraticQuadBoundary`
            - ``line3``
        *   - :class:`~felupe.RegionBiQuadraticQuadBoundary`
            - ``line3``
        *   - :class:`~felupe.RegionHexahedronBoundary`
            - ``quad``
        *   - :class:`~felupe.RegionQuadraticHexahedronBoundary`
            - ``quad8``
        *   - :class:`~felupe.RegionTriQuadraticHexahedronBoundary`
            - ``quad9``

    In 2d, the out-of-plane behaviour is defined by the type of the displacement field:
    a :class:`~felupe.FieldPlaneStrain` or a :class:`~felupe.FieldAxisymmetric` is
    required, a (Cartesian) :class:`~felupe.Field` is not valid. For an axisymmetric
    field, the contact potential is integrated on the surface of revolution, i.e. the
    differential area of a face is the product of its differential length and the
    circumference :math:`2 \pi R` of the circle at its radial coordinate.

    ..  note::

        The mesh of both boundary regions must be the same mesh as the mesh of the
        region of the solid bodies. Two separate meshes are combined by
        :meth:`MeshContainer.stack() <felupe.MeshContainer.stack>`.

    ..  hint::

        The choice of the secondary surface matters for a single-pass contact: Use the
        softer and finer meshed surface as the secondary surface, e.g. the rubber
        surface of a rubber-to-metal contact. Alternatively, use ``two_pass=True``,
        which removes this bias but requires about twice the evaluation time.

    ..  hint::

        The contact surfaces do not need to be restricted to the region of interest: if
        all faces on the outline of a mesh are used, the contact is evaluated on both
        surfaces of a contact zone. This doubles the contact potential, i.e. it acts
        like a two-pass contact without its scale factors, and hence ``two_pass=True``
        is redundant in this case.

    ..  hint::

        The estimate of the penalty stiffness assumes an equal area per point of a
        contact surface. This is not the case for a boundary region of a mesh with
        quadratic cells, where the diagonal entries of the stiffness matrix differ
        between the points on the edges and the midpoints of a face. Hence the estimate
        is stiffer for quadratic cells. Decrease ``penalty_scale`` or use smaller load
        increments if the Newton-Raphson method does not converge.

    Examples
    --------
    A rubber block is pressed on a (stiffer) block. Both blocks are meshed
    individually and are combined to a single mesh.

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>> import numpy as np
        >>>
        >>> from felupe_contact import SolidBodyContact
        >>>
        >>> bottom = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(4, 4, 3))
        >>> top = fem.Cube(a=(0.15, 0.15, 1.02), b=(0.85, 0.85, 1.62), n=(3, 3, 3))
        >>> container = fem.MeshContainer([bottom, top], merge=True)
        >>> mesh = container.stack()
        >>>
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

    The contact surfaces are created as boundary regions on the same mesh. Only the
    faces on the outline of the mesh which are located in the region of interest are
    used.

    ..  pyvista-plot::
        :context:

        >>> mask = np.logical_and(mesh.z > 1.0, mesh.z < 1.1)
        >>> secondary = fem.FieldContainer(
        ...     [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=mask), dim=3)]
        ... )
        >>> primary = fem.FieldContainer(
        ...     [fem.Field(
        ...         fem.RegionHexahedronBoundary(mesh, mask=np.isclose(mesh.z, 1.0)),
        ...         dim=3,
        ...     )]
        ... )
        >>> contact = SolidBodyContact(secondary, primary, items=[solid])

    The bottom face of the lower block is fixed and the top face of the upper block is
    moved downwards.

    ..  pyvista-plot::
        :context:

        >>> boundaries = {
        ...     "fixed": fem.Boundary(field[0], fz=0.0),
        ...     "clamped": fem.Boundary(field[0], fz=mesh.z.max(), skip=(0, 0, 1)),
        ...     "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0)),
        ... }
        >>> move = fem.math.linsteps([0, -0.3], num=3)
        >>> step = fem.Step(
        ...     items=[solid, contact],
        ...     ramp={boundaries["move"]: move},
        ...     boundaries=boundaries,
        ... )
        >>> job = fem.Job(steps=[step]).evaluate(verbose=0)

    The number of integration points which are in contact and the penalty stiffness,
    which is estimated from the given items, are available in the contact object.

    ..  pyvista-plot::
        :context:

        >>> contact.results.npoints_in_contact
        16

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> solid.plot("Principal Values of Cauchy Stress").show()

    An axisymmetric contact is created on the boundary regions of a two-dimensional
    mesh, where the displacement fields are axisymmetric fields. The rotation axis is
    the first axis of the mesh, i.e. a cylinder is placed on top of a bigger cylinder
    if both blocks are stacked along the first axis.

    ..  pyvista-plot::
        :context: close-figs

        >>> bottom = fem.Rectangle(a=(0, 0), b=(1, 1), n=(3, 5))
        >>> top = fem.Rectangle(a=(1.02, 0), b=(1.62, 0.7), n=(3, 4))
        >>> mesh = fem.MeshContainer([bottom, top], merge=True).stack()
        >>>
        >>> field = fem.FieldContainer(
        ...     [fem.FieldAxisymmetric(fem.RegionQuad(mesh), dim=2)]
        ... )
        >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)
        >>>
        >>> surface = fem.FieldContainer(
        ...     [fem.FieldAxisymmetric(fem.RegionQuadBoundary(mesh), dim=2)]
        ... )
        >>> contact = SolidBodyContact(surface, surface, items=[solid])
        >>> contact.results.npoints_in_contact
        0

    See Also
    --------
    felupe.MultiPointContact : A frictionless point-to-rigid (wall) contact.
    """

    def __init__(
        self,
        field,
        field_primary,
        items=None,
        penalty=None,
        penalty_scale=10.0,
        smoothing=None,
        two_pass=False,
        max_distance=None,
        candidates=8,
        tolerance=0.1,
        facing=0.1,
        self_contact=False,
        geometric_stiffness=True,
    ):
        self.field = field
        self.field_primary = field_primary
        self.items = items

        self.penalty = penalty
        self.penalty_scale = penalty_scale
        self.two_pass = two_pass
        self.candidates = candidates
        self.tolerance = tolerance
        self.facing = facing
        self.self_contact = self_contact
        self.geometric_stiffness = geometric_stiffness

        if items is None and penalty is None:
            raise ValueError("Either `items` or `penalty` must be given.")

        self.pairs = [ContactSurfacePair(field, field_primary)]

        if two_pass:
            self.pairs = [
                ContactSurfacePair(field, field_primary, weight=0.5),
                ContactSurfacePair(field_primary, field, weight=0.5),
            ]

        self.size = self.pairs[0].size

        # all faces of both contact surfaces belong to the same body: the contact search
        # discards every face-pair and hence no contact is ever detected
        pair = self.pairs[0]
        if not self_contact and len(np.union1d(pair.body, pair.body_primary)) == 1:
            warnings.warn(
                "All faces of both contact surfaces belong to the same body of the "
                "mesh, hence no contact will be detected. Use `self_contact=True` to "
                "consider face-pairs which belong to the same body."
            )

        self.smoothing = smoothing
        if smoothing is None:
            self.smoothing = 1e-2 * self.size

        self.max_distance = max_distance
        if max_distance is None:
            self.max_distance = 5 * self.size

        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)
        self.results = Results()
        self.results.gap = []
        self.results.pressure = []
        self.results.npoints_in_contact = 0

        self._kinematics = None
        self._values = None

    def __repr__(self):
        header = "<felupe SolidBodyContact object>"
        penalty = f"  Penalty stiffness: {self.penalty}"
        smoothing = f"  Smoothing: {self.smoothing}"
        contact = f"  Integration points in contact: {self.results.npoints_in_contact}"

        return "\n".join([header, penalty, smoothing, contact])

    def pressure(self, gap):
        r"""Return the contact pressure and its derivative w.r.t. the gap, evaluated by
        a regularized penalty law, see Eq. :eq:`contact-pressure`.

        Parameters
        ----------
        gap : ndarray
            The gap between the contact surfaces.

        Returns
        -------
        pressure : ndarray
            The contact pressure.
        dpressure : ndarray
            The derivative of the contact pressure w.r.t. the gap.
        """

        penalty, smoothing = self.penalty, self.smoothing

        pressure = np.zeros_like(gap)
        dpressure = np.zeros_like(gap)

        if smoothing > 0:
            closed = gap <= -smoothing
            transition = np.logical_and(gap > -smoothing, gap < 0)

            g = gap[transition]
            pressure[transition] = penalty * g**2 / (2 * smoothing)
            dpressure[transition] = penalty * g / smoothing

            pressure[closed] = -penalty * (gap[closed] + smoothing / 2)
        else:
            closed = gap < 0
            pressure[closed] = -penalty * gap[closed]

        dpressure[closed] = -penalty

        return pressure, dpressure

    def update_penalty(self):
        r"""Estimate and update the penalty stiffness from the mean stiffness of the
        degrees of freedom on the contact surfaces, see Eq. :eq:`contact-penalty`.

        Returns
        -------
        float
            The estimated penalty stiffness.
        """

        pair = self.pairs[0]
        diagonal = np.zeros(pair.ndof)

        for item in self.items:
            stiffness = item.results.stiffness

            if stiffness is None:
                stiffness = item.assemble.matrix()

            if item.assemble.multiplier is not None:
                stiffness = stiffness * item.assemble.multiplier

            values = stiffness.diagonal()
            size = min(len(values), pair.ndof)
            diagonal[:size] += values[:size]

        surfaces = [
            (pair.points, pair.area),
            (pair.points_primary, pair.area_primary),
        ]

        penalty = []
        for points, area in surfaces:
            stiffness = np.abs(diagonal[pair.dof[points].ravel()]).mean()
            penalty.append(stiffness * len(points) / area)

        self.penalty = self.penalty_scale * min(penalty)

        return self.penalty

    def _extract(self, field=None, parallel=False):
        """Evaluate and cache the contact kinematics of all surface pairs. The flag
        ``parallel`` activates the threaded tree-query of the contact search."""

        if field is not None:
            self.field = field

        values = self.field[0].values
        self.field_primary[0].values = values

        if self._kinematics is not None and np.array_equal(self._values, values):
            return self._kinematics

        if self.penalty is None:
            self.update_penalty()

        x = self.field.region.mesh.points + values

        self._kinematics = [
            pair.kinematics(
                x=x,
                max_distance=self.max_distance,
                candidates=self.candidates,
                tolerance=self.tolerance,
                facing=self.facing,
                self_contact=self.self_contact,
                workers=-1 if parallel else 1,
            )
            for pair in self.pairs
        ]
        self._values = values.copy()

        self.results.gap = [
            None if kin is None else kin["gap"] for kin in self._kinematics
        ]
        self.results.pressure = [
            None if kin is None else self.pressure(kin["gap"])[0]
            for kin in self._kinematics
        ]
        self.results.npoints_in_contact = sum(
            [0 if kin is None else len(kin["gap"]) for kin in self._kinematics]
        )

        return self._kinematics

    def _vector(self, field=None, parallel=False, resize=None):
        "Assemble the sparse contact force vector."

        kinematics = self._extract(field=field, parallel=parallel)
        force = csr_matrix((self.pairs[0].ndof, 1))

        for pair, kin in zip(self.pairs, kinematics):
            if kin is None:
                continue

            pressure = self.pressure(kin["gap"])[0]
            force += pair.assemble_vector(kin, -pressure)

        if resize is not None:
            force.resize(*resize.shape)

        self.results.force = force

        return force

    def _matrix(self, field=None, parallel=False, resize=None):
        "Assemble the sparse contact stiffness matrix."

        kinematics = self._extract(field=field, parallel=parallel)
        ndof = self.pairs[0].ndof
        stiffness = csr_matrix((ndof, ndof))

        for pair, kin in zip(self.pairs, kinematics):
            if kin is None:
                continue

            pressure, dpressure = self.pressure(kin["gap"])
            stiffness += pair.assemble_matrix(
                kin,
                -pressure,
                -dpressure,
                geometric=self.geometric_stiffness,
            )

        if resize is not None:
            stiffness.resize(*resize.shape)

        self.results.stiffness = stiffness

        return stiffness
