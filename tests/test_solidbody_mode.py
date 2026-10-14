# SPDX-License-Identifier: GPL-3.0-or-later

import felupe as fem
import numpy as np
import pytest

from felupe_contact import SolidBodyContact
from felupe_contact._boundary import (
    BOUNDARY_REGIONS,
    boundary_region,
    contact_surface,
)
from felupe_contact._solidbody_contact import ContactSurfacePair


def two_blocks():
    "Return the mesh of a block which is placed on top of a bigger block."

    bottom = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(4, 4, 3))
    top = fem.Cube(a=(0.15, 0.15, 1.02), b=(0.85, 0.85, 1.62), n=(3, 3, 3))

    return fem.MeshContainer([bottom, top], merge=True)


def solid_body(region):
    "Return the field and the solid body of a region."

    field = fem.FieldContainer([fem.Field(region, dim=3)])
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

    return field, solid


def surface(mesh, mask=None):
    "Return a field container on a boundary region of a mesh."

    region = fem.RegionHexahedronBoundary(mesh, mask=mask)

    return fem.FieldContainer([fem.Field(region, dim=3)])


def compress(field, items, move=-0.3, num=3):
    "Move the top face of the upper block downwards."

    z = field.region.mesh.points[:, 2]
    boundaries = {
        "fixed": fem.Boundary(field[0], fz=0.0),
        "clamped": fem.Boundary(field[0], fz=z.max(), skip=(0, 0, 1)),
        "move": fem.Boundary(field[0], fz=z.max(), skip=(1, 1, 0)),
    }
    step = fem.Step(
        items=items,
        ramp={boundaries["move"]: fem.math.linsteps([0, move], num=num)},
        boundaries=boundaries,
    )
    fem.Job(steps=[step]).evaluate(verbose=0)

    return field[0].values.copy()


def test_solid_body_uses_all_faces():
    """A contact of a solid body is equal to a contact of boundary regions with all
    faces on the outline of the mesh, see Ex. 2."""

    mesh = two_blocks().stack()

    field, solid = solid_body(fem.RegionHexahedron(mesh))
    contact = SolidBodyContact(solid)
    values = compress(field, [solid, contact])

    assert contact.items == [solid]
    assert contact.field.region.mesh.ncells == surface(mesh).region.mesh.ncells

    reference_field, reference_solid = solid_body(fem.RegionHexahedron(mesh))
    reference = SolidBodyContact(surface(mesh), surface(mesh), items=[reference_solid])
    reference_values = compress(reference_field, [reference_solid, reference])

    assert contact.results.npoints_in_contact == reference.results.npoints_in_contact
    assert contact.results.npoints_in_contact == 32
    assert np.allclose(values, reference_values)


def test_solid_body_with_masks():
    """A contact of a solid body with masks is equal to a contact of masked boundary
    regions, see Ex. 1."""

    mesh = two_blocks().stack()
    secondary = np.logical_and(mesh.z > 1.0, mesh.z < 1.1)
    primary = np.isclose(mesh.z, 1.0)

    field, solid = solid_body(fem.RegionHexahedron(mesh))
    contact = SolidBodyContact(solid, mask=secondary, mask_primary=primary)
    values = compress(field, [solid, contact])

    reference_field, reference_solid = solid_body(fem.RegionHexahedron(mesh))
    reference = SolidBodyContact(
        surface(mesh, mask=secondary),
        surface(mesh, mask=primary),
        items=[reference_solid],
    )
    reference_values = compress(reference_field, [reference_solid, reference])

    assert contact.field is not contact.field_primary
    assert contact.results.npoints_in_contact == 16
    assert np.allclose(values, reference_values)


def test_solid_body_with_a_mask_only():
    """Without a primary solid body and without a primary mask, the masked secondary
    surface is also the primary surface."""

    mesh = two_blocks().stack()
    field, solid = solid_body(fem.RegionHexahedron(mesh))

    mask = np.logical_and(mesh.z > 0.9, mesh.z < 1.1)
    contact = SolidBodyContact(solid, mask=mask)

    assert contact.field_primary is contact.field
    assert (
        contact.field.region.mesh.ncells == surface(mesh, mask=mask).region.mesh.ncells
    )


def test_two_solid_bodies_of_sub_meshes():
    """A contact of the solid bodies of two sub-meshes is equal to a contact of the
    boundary regions of the sub-meshes, see Ex. 3."""

    container = two_blocks()

    def solve(contact_of):
        fields, solids = zip(*[solid_body(fem.RegionHexahedron(m)) for m in container])
        field = fem.field.merge(list(fields))
        contact = contact_of(solids)

        return contact, solids, compress(field, [*solids, contact])

    contact, solids, values = solve(lambda solids: SolidBodyContact(*solids))

    reference, _, reference_values = solve(
        lambda solids: SolidBodyContact(
            *[surface(m) for m in container], items=list(solids)
        )
    )

    assert contact.items == list(solids)
    assert contact.results.npoints_in_contact == 16
    assert contact.results.npoints_in_contact == reference.results.npoints_in_contact
    assert np.allclose(values, reference_values)


def test_items_of_the_same_solid_body_are_not_repeated():
    "A solid body which is given for both surfaces is used only once as an item."

    mesh = two_blocks().stack()
    field, solid = solid_body(fem.RegionHexahedron(mesh))

    contact = SolidBodyContact(solid, solid)

    assert contact.items == [solid]


def test_given_items_are_kept():
    "The given items are used instead of the solid bodies."

    mesh = two_blocks().stack()
    field, solid = solid_body(fem.RegionHexahedron(mesh))
    other_field, other_solid = solid_body(fem.RegionHexahedron(mesh))

    contact = SolidBodyContact(solid, items=[other_solid])

    assert contact.items == [other_solid]


def test_field_container_on_the_region_of_a_solid_body():
    """A field container on the region of a solid body is supported, the penalty
    stiffness must be given."""

    mesh = two_blocks().stack()
    field, solid = solid_body(fem.RegionHexahedron(mesh))

    with pytest.raises(ValueError, match="penalty"):
        SolidBodyContact(field)

    contact = SolidBodyContact(field, penalty=1.0)

    assert contact.items is None
    assert type(contact.field.region) is fem.RegionHexahedronBoundary


def test_mixed_field_container():
    """The displacement field of a mixed field container is used, i.e. the contact
    surface has a single field which is linked to the first field."""

    mesh = two_blocks().stack()
    field = fem.FieldsMixed(fem.RegionHexahedron(mesh), n=3)
    solid = fem.SolidBody(
        umat=fem.ThreeFieldVariation(fem.NeoHooke(mu=1.0, bulk=50.0)), field=field
    )

    contact = SolidBodyContact(solid)

    assert len(contact.field.fields) == 1
    assert contact.field[0].values is field[0].values

    compress(field, [solid, contact])

    assert contact.results.npoints_in_contact == 32


def test_a_body_with_a_cavity_is_a_single_body():
    """The cells of a solid body identify its connected bodies. The cells of a boundary
    region are disconnected for a body with an enclosed cavity and a wall of more than
    two cells, i.e. the inner surface is detected as another body which penetrates the
    outer surface."""

    n, wall = 11, 3
    mesh = fem.Cube(n=n)
    centers = mesh.points[mesh.cells].mean(axis=1)
    h = 1 / (n - 1)
    cavity = np.all((centers > wall * h) & (centers < 1 - wall * h), axis=1)
    mesh.update(cells=mesh.cells[~cavity])

    field, solid = solid_body(fem.RegionHexahedron(mesh))
    with pytest.warns(UserWarning, match="same body"):
        contact = SolidBodyContact(solid)

    contact.assemble.vector()

    assert len(np.unique(contact.pairs[0].body)) == 1
    assert contact.results.npoints_in_contact == 0

    # the cells of the boundary region are not sufficient
    boundary = surface(mesh)
    boundary.link(field)
    reference = SolidBodyContact(boundary, items=[solid])
    reference.assemble.vector()

    assert len(np.unique(reference.pairs[0].body)) == 2
    assert reference.results.npoints_in_contact > 0


def test_a_mask_requires_a_solid_body():
    "A mask is not supported for a field container on a boundary region."

    mesh = two_blocks().stack()

    with pytest.raises(ValueError, match="mask"):
        SolidBodyContact(surface(mesh), penalty=1.0, mask=np.ones(mesh.npoints, bool))

    with pytest.raises(ValueError, match="mask"):
        SolidBodyContact(
            surface(mesh), penalty=1.0, mask_primary=np.ones(mesh.npoints, bool)
        )


def test_requires_a_solid_body_or_a_field_container():
    "A mesh is neither a solid body nor a field container."

    mesh = two_blocks().stack()

    with pytest.raises(TypeError, match="solid body"):
        SolidBodyContact(mesh, penalty=1.0)


@pytest.mark.parametrize("region, boundary", list(BOUNDARY_REGIONS.items()))
def test_boundary_regions(region, boundary):
    "The boundary region of a region is looked up."

    if "Hexahedron" in region.__name__:
        mesh = fem.Cube(n=3)
        convert = {
            fem.RegionQuadraticHexahedron: dict(
                order=2, calc_midfaces=False, calc_midvolumes=False
            ),
            fem.RegionTriQuadraticHexahedron: dict(
                order=2, calc_midfaces=True, calc_midvolumes=True
            ),
        }
    else:
        mesh = fem.Rectangle(n=3)
        convert = {
            fem.RegionQuadraticQuad: dict(order=2, calc_midfaces=False),
            fem.RegionBiQuadraticQuad: dict(order=2, calc_midfaces=True),
        }

    if region in convert:
        mesh = fem.mesh.convert(mesh, **convert[region])

    mask = mesh.points[:, 0] > 0.9

    assert type(boundary_region(region(mesh))) is boundary
    assert boundary_region(region(mesh), mask=mask).mesh.ncells == (
        boundary(mesh, mask=mask).mesh.ncells
    )


def test_boundary_region_of_a_subclass():
    "The boundary region of a subclass of a region of FElupe is looked up."

    class MyRegion(fem.RegionHexahedron):
        pass

    region = boundary_region(MyRegion(fem.Cube(n=3)))

    assert type(region) is fem.RegionHexahedronBoundary


def test_boundary_region_of_an_unsupported_region():
    "No boundary region is available for a tetrahedral region."

    mesh = fem.Cube(n=3).triangulate()

    with pytest.raises(NotImplementedError, match="RegionTetra"):
        boundary_region(fem.RegionTetra(mesh))


def test_contact_surface_of_a_boundary_region():
    "A field container on a boundary region is a contact surface."

    mesh = two_blocks().stack()
    container = surface(mesh)

    field, cells = contact_surface(container)

    assert field is container
    assert cells is None


def test_surface_pair_uses_the_cells_of_the_boundary_regions():
    "Without cells, the bodies are identified by the cells of the boundary regions."

    mesh = two_blocks().stack()
    pair = ContactSurfacePair(surface(mesh), surface(mesh))

    assert len(np.unique(pair.body)) == 2
