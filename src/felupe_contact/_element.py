# SPDX-License-Identifier: GPL-3.0-or-later

from itertools import product

import numpy as np
from felupe.element import (
    ArbitraryOrderLagrange,
    BiQuadraticQuad,
    Line,
    Quad,
    QuadraticQuad,
)
from scipy.special import binom


def quadratic_line():
    "Return a quadratic line element with the point-ordering of a ``line3`` cell."

    return ArbitraryOrderLagrange(order=2, dim=1)


# the element formulation of the faces of a boundary region, given for the cell type of
# the mesh of the boundary region. the faces of a boundary region of a three-dimensional
# mesh are quads and the faces of a two-dimensional mesh are lines (edges)
FACE_ELEMENTS = {
    "quad": Line,
    "quad8": quadratic_line,
    "quad9": quadratic_line,
    "hexahedron": Quad,
    "hexahedron20": QuadraticQuad,
    "hexahedron27": BiQuadraticQuad,
}


class BatchedElement:
    r"""Evaluate the shape functions of a finite element of FElupe and their first and
    second partial derivatives w.r.t. the natural element coordinates for a batch of
    coordinates.

    Parameters
    ----------
    element : Element
        A finite element formulation of FElupe with polynomial shape functions, e.g.
        :class:`~felupe.Quad`.
    tol : float, optional
        The tolerance for the check of the polynomial expansion of the shape functions
        (default is 1e-10).

    Attributes
    ----------
    element : Element
        The finite element formulation.
    points : ndarray of shape (npoints, dim)
        The natural element coordinates of the points of the element.
    npoints : int
        The number of points of the element.
    dim : int
        The number of natural element coordinates.
    order : ndarray of shape (dim,)
        The polynomial order of the shape functions per direction.
    exponents : ndarray of shape (nmonomials, dim)
        The exponents of the monomials of the polynomial basis.
    coefficients : ndarray of shape (npoints, nmonomials)
        The coefficients of the shape functions w.r.t. the monomials.

    Notes
    -----
    The elements of FElupe evaluate their shape functions at a single point. The
    closest-point projection of a contact search requires the shape functions of the
    faces of the primary surface at an individual point per face-pair, i.e. at a batch
    of natural element coordinates.

    The shape functions of a Lagrange- or a serendipity-element are polynomials, hence
    they are given as a linear combination of the monomials
    :math:`m_m(\boldsymbol{\xi}) = \prod_J \xi_J^{n_{mJ}}` of the polynomial basis, see
    Eq. :eq:`shape-function-polynomial`. The basis contains the monomials up to the
    polynomial order per direction, which is given by the number of the distinct
    coordinates of the points of the element.

    ..  math::
        :label: shape-function-polynomial

        h_a(\boldsymbol{\xi}) = C_{am}\ m_m(\boldsymbol{\xi})

    The coefficients :math:`C_{am}` are evaluated once on initiation: the shape
    functions of the element are sampled on a grid, which is finer than the polynomial
    order, and the resulting (over-determined but consistent) linear system is solved
    for the coefficients. This expansion is exact, not an approximation, and it is
    verified on the sample points.

    The partial derivatives of the monomials are monomials of the same basis, i.e. the
    partial derivatives of the shape functions are given by the coefficients
    :math:`C_{am}\ D_{mn}`, where the (constant) matrix :math:`D_{mn}` differentiates
    the monomials. The coefficients of the shape functions and of all their first and
    second partial derivatives are stacked on initiation. Hence, all of them are
    evaluated by a single evaluation of the monomials and a single matrix product, see
    :meth:`evaluate`.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from felupe import Quad
    >>> from felupe_contact._element import BatchedElement
    >>>
    >>> element = BatchedElement(Quad())
    >>> coordinates = np.zeros((2, 5))
    >>>
    >>> element.function(coordinates).shape
    (4, 5)

    >>> element.gradient(coordinates).shape
    (4, 2, 5)

    >>> element.hessian(coordinates).shape
    (4, 2, 2, 5)
    """

    def __init__(self, element, tol=1e-10):
        self.element = element
        self.points = np.asarray(element.points, dtype=float)
        self.npoints, self.dim = self.points.shape

        # the polynomial order per direction is given by the number of the distinct
        # coordinates of the points of the element
        self.order = np.array(
            [len(np.unique(coordinates)) - 1 for coordinates in self.points.T]
        )

        self.exponents = np.array(
            list(product(*[range(o + 1) for o in self.order])), dtype=int
        )

        # the shape functions are sampled on a grid which is finer than the polynomial
        # order, i.e. the linear system for the coefficients is over-determined
        samples = self._grid(1.0)
        values = np.array([element.function(point) for point in samples.T]).T

        self.coefficients = np.linalg.lstsq(
            self.monomials(samples).T, values.T, rcond=None
        )[0].T

        # the coefficients of the shape functions and of their first and second partial
        # derivatives, stacked along the first axis
        axes = [(), *[(J,) for J in range(self.dim)]]
        axes += [(J, K) for J in range(self.dim) for K in range(self.dim)]

        self._coefficients = np.concatenate(
            [self.coefficients @ self._differentiate(a) for a in axes]
        )

        # the control nets of the faces, cached for the extents of the coordinates
        self._control_nets = {}

        # the expansion of the shape functions in the polynomial basis must be exact
        residual = np.abs(self.function(samples) - values).max()
        if residual > tol:
            raise NotImplementedError(
                f"The shape functions of {type(element).__name__} are not represented "
                f"by the polynomial basis (residual {residual})."
            )

    def _grid(self, extent, refinement=1):
        """Return a grid of natural element coordinates within ``[-extent, extent]``,
        which is finer than the polynomial order, shape (dim, nsamples). The grid
        includes the corners and it has ``refinement * (order + 1) + 1`` points per
        direction."""

        axes = [
            np.linspace(-extent, extent, refinement * (o + 1) + 1) for o in self.order
        ]
        grid = np.meshgrid(*axes, indexing="ij")

        return np.array([axis.ravel() for axis in grid])

    def _differentiate(self, axes):
        """Return the matrix ``D_mn`` which differentiates the monomials ``m_m`` w.r.t.
        the given axes, i.e. the derivative of a monomial is ``D_mn m_n``."""

        order = np.bincount(np.asarray(axes, dtype=int), minlength=self.dim)
        exponents = self.exponents - order

        # the k-th derivative of r^n is n (n-1) ... (n-k+1) r^(n-k). it vanishes for
        # k > n, i.e. if an exponent of the derivative is negative
        coefficients = np.ones(len(exponents))
        for axis, k in enumerate(order):
            for i in range(k):
                coefficients = coefficients * (self.exponents[:, axis] - i)

        index = {tuple(n): m for m, n in enumerate(self.exponents)}
        matrix = np.zeros((len(exponents), len(exponents)))

        for m, n in enumerate(exponents):
            if np.all(n >= 0):
                matrix[m, index[tuple(n)]] = coefficients[m]

        return matrix

    def monomials(self, coordinates):
        r"""Return the monomials of the polynomial basis, evaluated for a batch of
        natural element coordinates.

        Parameters
        ----------
        coordinates : ndarray of shape (dim, ...)
            The natural element coordinates.

        Returns
        -------
        ndarray of shape (nmonomials, ...)
            The monomials of the polynomial basis.
        """

        monomials = np.ones((len(self.exponents), *coordinates.shape[1:]))

        # the powers of a coordinate are evaluated once and gathered for the monomials
        for axis, order in enumerate(self.order):
            powers = coordinates[axis] ** np.arange(order + 1).reshape(
                -1, *[1] * (coordinates.ndim - 1)
            )
            monomials *= powers[self.exponents[:, axis]]

        return monomials

    def evaluate(self, coordinates):
        """Return the shape functions and their first and second partial derivatives
        w.r.t. the natural element coordinates, evaluated for a batch of coordinates.

        Parameters
        ----------
        coordinates : ndarray of shape (dim, ...)
            The natural element coordinates.

        Returns
        -------
        h : ndarray of shape (npoints, ...)
            The shape functions.
        dhdr : ndarray of shape (npoints, dim, ...)
            The gradients of the shape functions.
        d2hdrdr : ndarray of shape (npoints, dim, dim, ...)
            The hessians of the shape functions.

        Notes
        -----
        The monomials are evaluated once and all derivatives are obtained by a single
        matrix product with the stacked coefficients.
        """

        batch = coordinates.shape[1:]
        monomials = self.monomials(coordinates).reshape(len(self.exponents), -1)
        values = self._coefficients @ monomials

        n, dim = self.npoints, self.dim
        h = values[:n].reshape(n, *batch)
        dhdr = values[n : n * (1 + dim)].reshape(dim, n, *batch).swapaxes(0, 1)
        d2hdrdr = values[n * (1 + dim) :].reshape(dim, dim, n, *batch)

        return h, dhdr, np.moveaxis(d2hdrdr, 2, 0)

    def function(self, coordinates):
        """Return the shape functions ``h_a``, evaluated for a batch of natural element
        coordinates.

        Parameters
        ----------
        coordinates : ndarray of shape (dim, ...)
            The natural element coordinates.

        Returns
        -------
        ndarray of shape (npoints, ...)
            The shape functions.
        """

        return np.einsum(
            "am,m...->a...", self.coefficients, self.monomials(coordinates)
        )

    def gradient(self, coordinates):
        """Return the partial derivatives ``dhdr_aJ`` of the shape functions w.r.t. the
        natural element coordinates, evaluated for a batch of coordinates.

        Parameters
        ----------
        coordinates : ndarray of shape (dim, ...)
            The natural element coordinates.

        Returns
        -------
        ndarray of shape (npoints, dim, ...)
            The gradients of the shape functions.
        """

        return self.evaluate(coordinates)[1]

    def hessian(self, coordinates):
        """Return the second partial derivatives ``d2hdrdr_aJK`` of the shape functions
        w.r.t. the natural element coordinates, evaluated for a batch of coordinates.

        Parameters
        ----------
        coordinates : ndarray of shape (dim, ...)
            The natural element coordinates.

        Returns
        -------
        ndarray of shape (npoints, dim, dim, ...)
            The hessians of the shape functions.
        """

        return self.evaluate(coordinates)[2]

    def control_net(self, extent=1.0):
        r"""Return the matrix which transforms the points of a face into the control
        points of a Bézier patch. The Bézier patch is the face on the (extended) range
        ``[-extent, extent]`` of the natural element coordinates.

        Parameters
        ----------
        extent : float, optional
            The extent of the natural element coordinates (default is 1.0).

        Returns
        -------
        ndarray of shape (ncontrolpoints, npoints)
            The matrix ``T_ja`` which transforms the points ``x_a`` of a face into the
            control points ``b_j = T_ja x_a``.

        Notes
        -----
        The shape functions span the polynomials up to the polynomial order per
        direction, the tensor-product Bernstein polynomials
        :math:`B_j(\boldsymbol{\xi})` of the same order span the same space. Hence,
        a face is given exactly by the control points of a Bézier patch, see Eq.
        :eq:`bezier-patch`.

        ..  math::
            :label: bezier-patch

            \boldsymbol{x}(\boldsymbol{\xi})
                = h_a(\boldsymbol{\xi})\ \boldsymbol{x}_a
                = B_j(\boldsymbol{\xi})\ \boldsymbol{b}_j

        The Bernstein polynomials are non-negative and sum up to one, i.e. a face is
        located within the convex hull of its control points (and so is any linear
        function of its coordinates within the range of its control points). This gives
        rigorous bounds of a face, which are used by the contact search to discard
        face-pairs without the expensive closest-point projection.

        Examples
        --------
        >>> import numpy as np
        >>>
        >>> from felupe import Quad
        >>> from felupe_contact._element import BatchedElement
        >>>
        >>> element = BatchedElement(Quad())
        >>> vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        >>>
        >>> control_points = element.control_net(extent=1.0) @ vertices
        >>> np.allclose(np.sort(control_points, axis=0), np.sort(vertices, axis=0))
        True
        """

        if extent not in self._control_nets:
            samples = self._grid(extent)

            # the 1d-Bernstein polynomials of the coordinates, scaled to [0, 1]
            u = (samples / extent + 1) / 2
            bernstein = np.ones((len(self.exponents), samples.shape[1]))

            for axis, order in enumerate(self.order):
                i = self.exponents[:, axis, None]
                bernstein *= (
                    binom(order, i) * u[axis] ** i * (1 - u[axis]) ** (order - i)
                )

            # the shape functions in terms of the bernstein polynomials (exact)
            values = self.function(samples)
            matrix = np.linalg.lstsq(bernstein.T, values.T, rcond=None)[0]

            residual = np.abs(matrix.T @ bernstein - values).max()
            if residual > 1e-10:  # pragma: no cover
                raise ValueError(
                    "The shape functions are not represented by the Bernstein "
                    f"polynomials (residual {residual})."
                )

            self._control_nets[extent] = matrix

        return self._control_nets[extent]


def face_element(region):
    """Return the element formulation of the faces of a boundary region and the indices
    of the points of a face within the points of a cell of the boundary region.

    Parameters
    ----------
    region : RegionBoundary
        A boundary region, e.g. a :class:`~felupe.RegionHexahedronBoundary` or a
        :class:`~felupe.RegionQuadBoundary`.

    Returns
    -------
    element : BatchedElement
        The element formulation of the faces of the boundary region.
    index : ndarray of shape (npoints_per_face,)
        The indices of the points of a face within the points of a cell.

    Notes
    -----
    The cells of a boundary region are rotated cells of the mesh with the face of
    interest as their first face. Hence the points of a face are a subset of the points
    of a cell of the boundary region and the shape functions of a face are the shape
    functions of these points, evaluated on the face.
    """

    cell_type = region.mesh.cell_type

    if cell_type not in FACE_ELEMENTS:
        raise NotImplementedError(
            f"The cell type {cell_type} is not supported. Supported cell types are "
            f"{list(FACE_ELEMENTS.keys())}."
        )

    element = BatchedElement(FACE_ELEMENTS[cell_type]())

    cells = region.mesh.cells
    cells_faces = region.mesh.cells_faces

    index = np.array([np.flatnonzero(cells[0] == a)[0] for a in cells_faces[0]])

    if not np.array_equal(cells[:, index], cells_faces):
        raise ValueError(
            "The points of the faces of the boundary region are not located at the "
            "same positions within the points of their cells."
        )

    return element, index
