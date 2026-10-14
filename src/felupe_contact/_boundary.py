# SPDX-License-Identifier: GPL-3.0-or-later

from felupe import (
    FieldContainer,
    RegionBiQuadraticQuad,
    RegionBiQuadraticQuadBoundary,
    RegionBoundary,
    RegionHexahedron,
    RegionHexahedronBoundary,
    RegionQuad,
    RegionQuadBoundary,
    RegionQuadraticHexahedron,
    RegionQuadraticHexahedronBoundary,
    RegionQuadraticQuad,
    RegionQuadraticQuadBoundary,
    RegionTriQuadraticHexahedron,
    RegionTriQuadraticHexahedronBoundary,
)

# the boundary region which belongs to the region of a solid body
BOUNDARY_REGIONS = {
    RegionQuad: RegionQuadBoundary,
    RegionQuadraticQuad: RegionQuadraticQuadBoundary,
    RegionBiQuadraticQuad: RegionBiQuadraticQuadBoundary,
    RegionHexahedron: RegionHexahedronBoundary,
    RegionQuadraticHexahedron: RegionQuadraticHexahedronBoundary,
    RegionTriQuadraticHexahedron: RegionTriQuadraticHexahedronBoundary,
}


def boundary_region(region, mask=None):
    """Return the boundary region of a region, see ``BOUNDARY_REGIONS``.

    Parameters
    ----------
    region : Region
        A region of a solid body, e.g. a :class:`~felupe.RegionHexahedron`.
    mask : ndarray of bool or None, optional
        A mask of the points of the mesh (default is None). Only the faces on the
        outline of the mesh with all their points in the mask are used. If None, all
        faces on the outline of the mesh are used.

    Returns
    -------
    RegionBoundary
        The boundary region, e.g. a :class:`~felupe.RegionHexahedronBoundary`.

    Notes
    -----
    The boundary region is looked up for the type of the region and for its base
    classes, i.e. a subclass of a region of FElupe is supported as well.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> from felupe_contact._boundary import boundary_region
    >>>
    >>> mesh = fem.Cube(n=3)
    >>> boundary = boundary_region(fem.RegionHexahedron(mesh))
    >>> type(boundary).__name__
    'RegionHexahedronBoundary'

    >>> boundary.mesh.ncells
    24
    """

    for cls in type(region).__mro__:
        if cls in BOUNDARY_REGIONS:
            return BOUNDARY_REGIONS[cls](region.mesh, mask=mask)

    names = ", ".join([f"`{cls.__name__}`" for cls in BOUNDARY_REGIONS])

    raise NotImplementedError(
        f"No boundary region is available for `{type(region).__name__}`. Supported "
        f"regions are {names}."
    )


def contact_surface(body, mask=None):
    """Return a field container on a boundary region, i.e. a contact surface, and the
    cells of the solid body.

    Parameters
    ----------
    body : SolidBody or FieldContainer
        A solid body (or any item with a field container ``field``) or a field
        container. The first field of the field container must be a displacement field.
        If its region is a boundary region, the field container is a contact surface.
        Otherwise, a boundary region is created on its region.
    mask : ndarray of bool or None, optional
        A mask of the points of the mesh (default is None). Only the faces on the
        outline of the mesh with all their points in the mask are used. If None, all
        faces on the outline of the mesh are used.

    Returns
    -------
    field : FieldContainer
        A field container with a displacement field on a boundary region.
    cells : ndarray of shape (ncells, npoints_per_cell) or None
        The point-connectivity of the cells of the solid body or None if the region of
        the given field container is a boundary region.

    Notes
    -----
    The displacement field of the contact surface is of the same type as the
    displacement field of the solid body, e.g. a :class:`~felupe.FieldAxisymmetric`.
    Its values are linked to the values of the displacement field of the solid body.
    """

    field = body if isinstance(body, FieldContainer) else getattr(body, "field", None)

    if not isinstance(field, FieldContainer):
        raise TypeError(
            "A solid body (an item with a field container `field`) or a field "
            f"container is required, got `{type(body).__name__}`."
        )

    displacement = field[0]
    region = displacement.region

    if isinstance(region, RegionBoundary):
        if mask is not None:
            raise ValueError(
                "A mask is only supported for a solid body or for a field container "
                "on the region of a solid body, the given field container is already "
                "created on a boundary region."
            )

        return field, None

    surface = FieldContainer(
        [
            type(displacement)(
                boundary_region(region, mask=mask),
                dim=displacement.dim,
                dtype=displacement.values.dtype,
            )
        ]
    )
    surface.link(field)

    return surface, region.mesh.cells
