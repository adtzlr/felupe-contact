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

from itertools import product

import numpy as np

from felupe.element import (
    ArbitraryOrderLagrange,
    BiQuadraticQuad,
    Line,
    Quad,
    QuadraticQuad,
)


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
    verified on the sample points. The partial derivatives of the shape functions are
    then obtained by differentiating the monomials analytically.

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
        order = [len(np.unique(coordinates)) - 1 for coordinates in self.points.T]

        self.exponents = np.array(
            list(product(*[range(o + 1) for o in order])), dtype=int
        )

        # the shape functions are sampled on a grid which is finer than the polynomial
        # order, i.e. the linear system for the coefficients is over-determined
        grid = np.meshgrid(*[np.linspace(-1, 1, o + 2) for o in order], indexing="ij")
        samples = np.array([axis.ravel() for axis in grid])
        values = np.array([element.function(point) for point in samples.T]).T

        self.coefficients = np.linalg.lstsq(
            self.monomials(samples).T, values.T, rcond=None
        )[0].T

        # the expansion of the shape functions in the polynomial basis must be exact
        residual = np.abs(self.function(samples) - values).max()
        if residual > tol:
            raise NotImplementedError(
                f"The shape functions of {type(element).__name__} are not represented "
                f"by the polynomial basis (residual {residual})."
            )

    def monomials(self, coordinates, order=None):
        r"""Return the monomials of the polynomial basis or their partial derivatives,
        evaluated for a batch of natural element coordinates.

        Parameters
        ----------
        coordinates : ndarray of shape (dim, ...)
            The natural element coordinates.
        order : ndarray of shape (dim,) or None, optional
            The order of the partial derivative per direction (default is None). If
            None, the monomials are returned.

        Returns
        -------
        ndarray of shape (nmonomials, ...)
            The monomials of the polynomial basis or their partial derivatives.
        """

        exponents = self.exponents

        if order is None:
            order = np.zeros(self.dim, dtype=int)

        # the k-th derivative of r^n is n (n-1) ... (n-k+1) r^(n-k). it vanishes for
        # k > n, in this case the coefficient is zero and the exponent is clipped
        coefficients = np.ones(len(exponents))
        for axis, k in enumerate(order):
            for i in range(k):
                coefficients = coefficients * (exponents[:, axis] - i)

        shape = (-1, *[1] * (coordinates.ndim - 1))
        monomials = np.ones((len(exponents), *coordinates.shape[1:]))

        for axis, k in enumerate(order):
            powers = np.maximum(exponents[:, axis] - k, 0)
            monomials = monomials * coordinates[axis] ** powers.reshape(shape)

        return coefficients.reshape(shape) * monomials

    def _derivative(self, coordinates, axes=()):
        "Return the partial derivative of the shape functions w.r.t. the given axes."

        order = np.bincount(axes, minlength=self.dim)

        return np.einsum(
            "am,m...->a...", self.coefficients, self.monomials(coordinates, order)
        )

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

        return self._derivative(coordinates)

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

        return np.stack(
            [self._derivative(coordinates, (J,)) for J in range(self.dim)], axis=1
        )

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

        return np.stack(
            [
                np.stack(
                    [self._derivative(coordinates, (J, K)) for K in range(self.dim)],
                    axis=1,
                )
                for J in range(self.dim)
            ],
            axis=1,
        )


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
