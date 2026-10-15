# SPDX-License-Identifier: GPL-3.0-or-later

import warnings

import numpy as np
from felupe import (
    AreaChange,
    FieldAxisymmetric,
    FieldContainer,
    RegionBoundary,
    SolidBodyNearlyIncompressible,
)
from felupe.math import det

from ._element import face_element
from ._solidbody_contact import SolidBodyContact, contiguous


def select(arrays, mask):
    """Return a dict with the arrays of the face-pairs within a mask.

    Parameters
    ----------
    arrays : dict
        A dict with arrays, whose last axis is the axis of the face-pairs, e.g. the
        contact kinematics.
    mask : ndarray of bool of shape (npairs,)
        The mask of the face-pairs to keep.

    Returns
    -------
    dict
        A dict with the C-contiguous arrays of the selected face-pairs.
    """

    return contiguous({key: value[..., mask] for key, value in arrays.items()})


def volume_cells(cells, cells_volume):
    """Return the cells of a mesh which contain the cells of a boundary region, together
    with the permutation of their points.

    Parameters
    ----------
    cells : ndarray of shape (ncells, npoints_per_cell)
        The point-connectivity of the cells of a boundary region. A cell of a boundary
        region is a cell of the mesh with rotated points, i.e. its first face is located
        on the boundary.
    cells_volume : ndarray of shape (ncells_volume, npoints_per_cell)
        The point-connectivity of the cells of the mesh.

    Returns
    -------
    cell : ndarray of shape (ncells,)
        The cell of the mesh for each cell of the boundary region.
    permutation : ndarray of shape (ncells, npoints_per_cell)
        The permutation of the points, i.e. ``cells[c] == cells_volume[cell[c]][
        permutation[c]]``.
    """

    # the cells which share all points are equal, i.e. the sorted rows are compared
    rows = np.sort(np.concatenate([cells_volume, cells]), axis=1)
    inverse = np.unique(rows, axis=0, return_inverse=True)[1].ravel()

    lookup = np.full(inverse.max() + 1, -1)
    lookup[inverse[: len(cells_volume)]] = np.arange(len(cells_volume))
    cell = lookup[inverse[len(cells_volume) :]]

    order_volume = np.argsort(cells_volume[cell], axis=1)
    order = np.argsort(cells, axis=1)

    permutation = np.empty_like(order)
    np.put_along_axis(permutation, order, order_volume, axis=1)

    return cell, permutation


def validate(solid):
    """Raise an error if the first Piola-Kirchhoff stress tensor of a solid body is not
    available at the integration points of its contact surface."""

    field = getattr(solid, "field", None)

    if not hasattr(solid, "umat") or not isinstance(field, FieldContainer):
        raise TypeError(
            "A solid body with a material model `umat` and a field container `field` "
            "is required, e.g. a `SolidBody` or a `SolidBodyNearlyIncompressible`, got "
            f"`{type(solid).__name__}`. The contact tractions of a Nitsche contact are "
            "evaluated by the stress tensors of the solid bodies."
        )

    if len(field.fields) > 1:
        raise NotImplementedError(
            "Solid bodies with mixed-field formulations are not supported, the field "
            "container of a solid body must only contain the displacement field. Use a "
            "`SolidBodyNearlyIncompressible` for (nearly) incompressible materials."
        )

    if isinstance(field[0].region, RegionBoundary):
        raise TypeError(
            "The displacement field of a solid body must be created on the region of "
            "the cells, not on a boundary region."
        )


class SurfaceStress:
    r"""The first Piola-Kirchhoff stress tensor of a solid body, evaluated at the
    integration points of the faces of a contact surface, along with its change w.r.t.
    the displacements of the points of the cells of the faces.

    Parameters
    ----------
    solid : SolidBody or SolidBodyNearlyIncompressible
        The solid body with the material model ``umat``.
    surface : FieldContainer
        The contact surface of the solid body, i.e. a field container with a
        displacement field on a boundary region of the region of the solid body.

    Notes
    -----
    This class is used internally by :class:`~felupe_contact.SolidBodyContactNitsche`.

    The cells of a boundary region of FElupe are the cells of the mesh which contain
    its faces, i.e. the deformation gradient tensor :math:`\boldsymbol{F}` and the
    first Piola-Kirchhoff stress tensor :math:`\boldsymbol{P}` are evaluated at the
    integration points of the faces by the shape functions of the cells. The traction
    vector :math:`\boldsymbol{P} \boldsymbol{N}` is given per unit undeformed area,
    where :math:`\boldsymbol{N}` is the outward unit normal vector of the undeformed
    face.

    The material model is evaluated with own state variables at the integration points
    of the contact surface. These are only updated after a converged increment, like
    the state variables of a solid body.

    For a :class:`~felupe.SolidBodyNearlyIncompressible`, the stress tensor contains the
    cell-wise constant pressure :math:`p = K (v / V - 1)` of the cell of a face, see Eq.
    :eq:`nitsche-pressure`, where :math:`v` and :math:`V` are the deformed and the
    undeformed volume of the cell. This is the pressure of the solid body at the
    converged solution of an increment.

    ..  math::
        :label: nitsche-pressure

        \boldsymbol{P} = \frac{\partial \psi}{\partial \boldsymbol{F}}
            + p\ \frac{\partial J}{\partial \boldsymbol{F}}
    """

    def __init__(self, solid, surface):
        region = surface.region
        displacement = surface[0]
        umat = solid.umat

        self.umat = umat
        self.dim = region.mesh.dim
        self.axisymmetric = isinstance(displacement, FieldAxisymmetric)
        self.area_change = AreaChange()

        # a displacement field on the boundary region. its values are set to the values
        # of the global displacement field on each evaluation
        self.field = FieldContainer(
            [
                type(displacement)(
                    region, dim=displacement.dim, dtype=displacement.values.dtype
                )
            ]
        )

        # the cells of the faces and the indices of the points of a face within the
        # points of a cell
        self.cells = region.mesh.cells
        self.ncells = region.mesh.ncells
        self.index = face_element(region)[1]
        self.nvars = self.cells.shape[1] * self.dim

        # the shape functions of the cells and their gradients w.r.t. the undeformed
        # coordinates, evaluated at the integration points of the faces
        self.h = np.ascontiguousarray(region.h[..., 0])
        self.dhdX = region.dhdX

        # the outward unit normal vectors of the undeformed faces with three components
        self.normals = np.zeros((3, *region.dV.shape))
        self.normals[: self.dim] = region.normals[: self.dim]

        self.radius = displacement.radius if self.axisymmetric else None

        # the shape of the state variables of the material model
        shape = umat.x[-1].shape if hasattr(umat, "x") else (0,)
        self.statevars_shape = (*shape, *region.dV.shape)

        self.incompressible = isinstance(solid, SolidBodyNearlyIncompressible)

        if self.incompressible:
            self._init_volume(solid)

    def _init_volume(self, solid):
        "Initialize the cells of the solid body for the cell-wise constant pressure."

        region = solid.field.region

        self.bulk = solid.bulk
        self.volume = solid.V
        self.cells_volume = region.mesh.cells
        self.cell, self.permutation = volume_cells(self.cells, self.cells_volume)

        self.h_volume = np.ascontiguousarray(region.h[..., 0])
        self.dhdX_volume = region.dhdX
        self.dV_volume = region.dV

        if self.axisymmetric:
            self.radius_volume = solid.field[0].radius
            self.dV_volume = 2 * np.pi * self.radius_volume * region.dV

    def statevars(self):
        "Return the initial state variables of the material model (zeros)."

        return np.zeros(self.statevars_shape)

    def evaluate(self, values, statevars):
        """Return the deformation gradient tensor, the first Piola-Kirchhoff stress
        tensor of the material model and the trial state variables at all integration
        points of the contact surface.

        Parameters
        ----------
        values : ndarray of shape (npoints, dim)
            The displacements of all points of the mesh.
        statevars : ndarray
            The state variables of the material model of the last converged increment.

        Returns
        -------
        dict
            A dict with the deformation gradient tensor ``F``, the first Piola-Kirchhoff
            stress tensor ``P`` and the trial state variables ``statevars``.
        """

        self.field[0].values = values

        F = self.field.extract()[0]
        gradient = self.umat.gradient([F, statevars])

        return {"F": F, "P": gradient[0], "statevars": gradient[-1]}

    def cell_volumes(self, values, cells):
        r"""Return the deformed volumes of cells of the solid body and their variations.

        Parameters
        ----------
        values : ndarray of shape (npoints, dim)
            The displacements of all points of the mesh.
        cells : ndarray of shape (ncells,)
            The cells of the solid body.

        Returns
        -------
        volume : ndarray of shape (ncells,)
            The deformed volumes :math:`v` of the cells.
        variation : ndarray of shape (npoints_per_cell, dim, ncells)
            The variations :math:`\delta v = \int_V \frac{\partial J}{\partial
            \boldsymbol{F}} : \delta \boldsymbol{F}\ dV` of the deformed volumes w.r.t.
            the displacements of the points of the cells.
        """

        dim = self.dim
        dhdX = self.dhdX_volume[..., cells]
        dV = self.dV_volume[:, cells]

        u = values[self.cells_volume[cells]].transpose(1, 2, 0)

        F = np.zeros((3, 3, *dV.shape))
        F[:dim, :dim] = np.einsum("aip,aJqp->iJqp", u, dhdX)
        F[[0, 1, 2], [0, 1, 2]] += 1.0

        if self.axisymmetric:
            radius = self.radius_volume[:, cells]
            F[2, 2] += np.einsum("aq,ap->qp", self.h_volume, u[:, 1]) / radius

        cofactor = self.area_change.function([F])[0]

        volume = np.einsum("qp,qp->p", det(F), dV)
        variation = np.einsum("kLqp,aLqp,qp->akp", cofactor[:dim, :dim], dhdX, dV)

        if self.axisymmetric:
            variation[:, 1] += np.einsum(
                "qp,aq,qp->ap", cofactor[2, 2], self.h_volume, dV / radius
            )

        return volume, variation

    def traction(self, kinematics, values, evaluated):
        r"""Return the traction vectors of the solid body and their normal components
        at the integration points in contact.

        Parameters
        ----------
        kinematics : dict
            The contact kinematics, see
            :meth:`ContactSurfacePair.kinematics()
            <felupe_contact._solidbody_contact.ContactSurfacePair.kinematics>`.
        values : ndarray of shape (npoints, dim)
            The displacements of all points of the mesh.
        evaluated : dict
            The stress tensors at all integration points of the contact surface, see
            :meth:`evaluate`.

        Returns
        -------
        dict
            A dict with the normal component of the traction vector ``normal``, the
            traction vector ``vector`` (three components), the deformation gradient
            tensor ``F`` and the outward unit normal vectors of the undeformed faces
            ``N`` at the integration points in contact. For a
            :class:`~felupe.SolidBodyNearlyIncompressible`, the pressure, the cofactor
            of the deformation gradient tensor and the variations of the cell volumes
            are included.

        Notes
        -----
        The normal component of the traction vector :math:`\lambda_N` is evaluated
        w.r.t. the outward unit normal vector :math:`\boldsymbol{n}` of the primary
        surface at the projected point, see Eq. :eq:`nitsche-traction`. It is positive
        for a compressive normal stress.

        ..  math::
            :label: nitsche-traction

            \lambda_N = \left( \boldsymbol{P} \boldsymbol{N} \right) \cdot \boldsymbol{n}
        """

        q, c = np.divmod(kinematics["point"], self.ncells)

        F = evaluated["F"][:, :, q, c]
        P = evaluated["P"][:, :, q, c]
        N = self.normals[:, q, c]

        result = {"F": F, "N": N}

        if self.incompressible:
            cells, inverse = np.unique(self.cell[c], return_inverse=True)
            volume, variation = self.cell_volumes(values, cells)

            pressure = self.bulk * (volume / self.volume[cells] - 1)
            cofactor = self.area_change.function([F])[0]
            P = P + pressure[inverse] * cofactor

            # the variations of the cell volumes, given for the (rotated) points of the
            # cells of the boundary region
            permutation = self.permutation[c].T[:, None, :]
            variation = np.take_along_axis(variation[..., inverse], permutation, axis=0)

            result["pressure"] = pressure[inverse]
            result["cofactor"] = cofactor
            result["variation"] = variation
            result["bulk"] = self.bulk / self.volume[cells][inverse]

        n = np.zeros_like(N)
        n[: self.dim] = kinematics["normal"]

        traction = np.einsum("iJp,Jp->ip", P, N)

        result["vector"] = traction
        result["normal"] = np.einsum("ip,ip->p", traction, n)

        return contiguous(result)

    def variation(self, kinematics, traction, statevars):
        r"""Return the change of the normal component of the traction vector w.r.t. the
        displacements of the points of the cells of the faces, evaluated for a fixed
        outward unit normal vector of the primary surface.

        Parameters
        ----------
        kinematics : dict
            The contact kinematics.
        traction : dict
            The traction vectors at the integration points in contact, see
            :meth:`traction`.
        statevars : ndarray
            The state variables of the material model of the last converged increment.

        Returns
        -------
        ndarray of shape (npoints_per_cell * dim, npairs)
            The change of the normal component of the traction vector.

        Notes
        -----
        The change of the normal component of the traction vector is given in Eq.
        :eq:`nitsche-traction-change` with the fourth-order elasticity tensor
        :math:`\mathbb{A} = \partial \boldsymbol{P} / \partial \boldsymbol{F}` of the
        material model and the change of the deformation gradient tensor
        :math:`\Delta \boldsymbol{F} = \Delta \boldsymbol{u} \otimes \nabla_X h`.

        ..  math::
            :label: nitsche-traction-change

            \Delta \lambda_N = \boldsymbol{n} \cdot \left(
                \mathbb{A} : \Delta \boldsymbol{F} \right) \boldsymbol{N}

        For a :class:`~felupe.SolidBodyNearlyIncompressible`, the change of the pressure
        :math:`\Delta p = K\ \Delta v / V` and the change of the cofactor of the
        deformation gradient tensor are included. For an axisymmetric field, the change
        of the circumferential stretch :math:`\Delta u_R / R` is included.
        """

        dim = self.dim
        q, c = np.divmod(kinematics["point"], self.ncells)
        npairs = len(c)

        F = traction["F"]
        N = traction["N"]
        n = np.zeros_like(N)
        n[:dim] = kinematics["normal"]

        # the elasticity tensor of the material model at the integration points in
        # contact, evaluated with the state variables of the last converged increment
        x = [F[:, :, None], statevars[..., q, c][..., None, :]]
        A = self.umat.hessian(x)[0][..., 0, :]

        if self.incompressible:
            A = A + traction["pressure"] * self.area_change.gradient([F])[0]

        # the change of the traction vector, projected on the normal vector
        T = np.einsum("ip,iJkLp,Jp->kLp", n, A, N)
        result = np.einsum("kLp,aLp->akp", T[:dim, :dim], self.dhdX[:, :, q, c])

        if self.axisymmetric:
            result[:, 1] += T[2, 2] * self.h[:, q] / self.radius[q, c]

        if self.incompressible:
            normal = np.einsum("iJp,Jp,ip->p", traction["cofactor"], N, n)
            result += normal * traction["bulk"] * traction["variation"]

        return result.reshape(self.nvars, npairs)

    def position(self, pair):
        """Return the positions of the degrees of freedom of a face-pair, see
        :meth:`ContactSurfacePair.dofs()
        <felupe_contact._solidbody_contact.ContactSurfacePair.dofs>`, within the
        degrees of freedom of the cell of the secondary surface and the face of the
        primary surface, see :meth:`dofs`."""

        secondary = self.index[:, None] * self.dim + np.arange(self.dim)

        return np.concatenate(
            [secondary.ravel(), self.nvars + np.arange(pair.nvars_primary)]
        )

    def dofs(self, pair, kinematics):
        """Return the degrees of freedom of the cells of the faces of the secondary
        surface, followed by those of the faces of the primary surface.

        Parameters
        ----------
        pair : ContactSurfacePair
            The surface pair.
        kinematics : dict
            The contact kinematics.

        Returns
        -------
        ndarray of shape (npairs, npoints_per_cell * dim + nvars_primary)
            The degrees of freedom of the face-pairs.
        """

        cell = kinematics["point"] % self.ncells

        return np.concatenate(
            [
                pair.dof[self.cells[cell]].reshape(-1, self.nvars),
                pair.dof[pair.cells_faces_primary[kinematics["face"]]].reshape(
                    -1, pair.nvars_primary
                ),
            ],
            axis=1,
        )

    def matrix(
        self,
        pair,
        kinematics,
        traction,
        statevars,
        pressure,
        dpressure,
        penalty,
        geometric=True,
    ):
        r"""Return the contact stiffness matrices of the face-pairs of a Nitsche
        contact.

        Parameters
        ----------
        pair : ContactSurfacePair
            The surface pair.
        kinematics : dict
            The contact kinematics.
        traction : dict
            The traction vectors at the integration points in contact, see
            :meth:`traction`.
        statevars : ndarray
            The state variables of the material model of the last converged increment.
        pressure : ndarray of shape (npairs,)
            The (weighted) contact pressure :math:`w\ p_N`.
        dpressure : ndarray of shape (npairs,)
            The derivative of the (weighted) contact pressure w.r.t. the gap.
        penalty : float
            The Nitsche parameter :math:`\gamma`.
        geometric : bool, optional
            Flag to add the geometric part of the stiffness matrix (default is True).

        Returns
        -------
        ndarray of shape (nvars, nvars, npairs)
            The stiffness matrices of the face-pairs, given for the degrees of freedom
            of :meth:`dofs`. The rows are the test- and the columns are the
            trial-functions.

        Notes
        -----
        The contact pressure depends on the gap and on the normal component of the
        traction vector, see Eq. :eq:`nitsche-pressure-change`. The change of the
        normal component of the traction vector contains the change of the stress
        tensor, see :meth:`variation`, and the change of the outward unit normal vector
        of the primary surface, see :meth:`ContactSurfacePair.variations_projection()
        <felupe_contact._solidbody_contact.ContactSurfacePair.variations_projection>`.

        ..  math::
            :label: nitsche-pressure-change

            \Delta p_N = \frac{\partial p_N}{\partial g} \left(
                \Delta g - \frac{\Delta \lambda_N}{\gamma} \right)

        The stiffness matrix of the penalty contact, evaluated with the contact
        pressure :math:`p_N` and its derivative w.r.t. the gap, is extended by the
        (non-symmetric) dyadic product of the variation of the gap and the change of
        the normal component of the traction vector.
        """

        dim = self.dim
        q, c = np.divmod(kinematics["point"], pair.ncells)
        npairs = len(c)

        position = self.position(pair)
        nvars = self.nvars + pair.nvars_primary

        # the stiffness matrix w.r.t. the gap, given for the points of the faces
        matrix = np.zeros((nvars, nvars, npairs))
        matrix[np.ix_(position, position)] = pair.matrix(
            kinematics, -pressure, -dpressure, geometric=geometric
        )

        # the variation of the gap and the change of the normal traction
        b = np.zeros((nvars, npairs))
        b[position] = np.concatenate(pair.variations_gap(kinematics))

        change = np.zeros((nvars, npairs))
        change[: self.nvars] = self.variation(kinematics, traction, statevars)

        if geometric:
            _, A, B = pair.variations(kinematics)
            N = pair.variations_projection(kinematics, A, B)[1]
            change[position] += np.einsum("ip,ivp->vp", traction["vector"][:dim], N)

        dA = pair.weight * pair.dA[q, c]
        matrix += np.einsum("vp,wp->vwp", dA * dpressure / penalty * b, change)

        return matrix


class SolidBodyContactNitsche(SolidBodyContact):
    r"""A frictionless contact between the surfaces of solid bodies, enforced by
    Nitsche's method, formulated for three-dimensional, plane strain and axisymmetric
    problems.

    Parameters
    ----------
    solid : SolidBody or SolidBodyNearlyIncompressible
        The solid body of the secondary surface. The contact surface is created on the
        outline of the mesh of the solid body (or the faces within ``mask``). The
        normal stress of the solid body at the integration points of the contact
        surface is used by the contact.
    solid_primary : SolidBody or SolidBodyNearlyIncompressible or None, optional
        The solid body of the primary surface (default is None). If None, the secondary
        surface (or the faces within ``mask_primary`` of the same solid body) is used as
        primary surface, i.e. all bodies of the mesh are able to touch each other.
    items : list of SolidBody or None, optional
        A list of items which are used to estimate the stiffness of the bodies, i.e.
        the Nitsche parameter and the weights of the contact tractions (default is
        None). If None, the given solid bodies are used.
    penalty : float or None, optional
        The Nitsche parameter :math:`\gamma` as contact pressure per unit gap (default
        is None). If None, it is estimated from the stiffness of the softest body on
        the contact surfaces, see Eq. :eq:`nitsche-stiffness`.
    penalty_scale : float, optional
        A scale factor which is applied on the estimated Nitsche parameter (default is
        10.0). This has no effect if ``penalty`` is given.
    smoothing : float or None, optional
        The length :math:`\delta` of the transition zone of the regularized contact
        pressure, see Eq. :eq:`nitsche-regularization` (default is None). If None, a
        fraction of the characteristic size of the faces of the contact surfaces is
        used. A value of zero deactivates the regularization.
    two_pass : bool, optional
        Flag to evaluate the contact on both surfaces (default is True). The contact
        tractions of both passes are weighted by the stiffness of the bodies, see Eq.
        :eq:`nitsche-weights`, i.e. the stress tensors of both solid bodies are used.
        With ``two_pass=False``, only the secondary surface is evaluated and only the
        stress tensor of the solid body of the secondary surface is used. If the
        secondary surface is also the primary surface, e.g. for all faces on the
        outline of a mesh, the contact is always evaluated on both surfaces of a
        contact zone (weighted) and this flag has no effect.
    max_distance : float or None, optional
        The maximum penetration of an integration point of the secondary surface which
        is considered by the contact search (default is None). If None, five times the
        characteristic size of the faces of the contact surfaces is used.
    max_gap : float or None, optional
        The maximum (positive) gap of an integration point in front of the primary
        surface which is considered by the contact search (default is None). If None,
        half of the characteristic size of the faces of the contact surfaces is used.
        A Nitsche contact is active as long as :math:`g < \lambda_N / \gamma`, i.e. it
        is also active for small positive gaps if the normal stress of the solid body
        is compressive.
    candidates : int, optional
        The number of candidate faces of the primary surface which are evaluated per
        integration point of the secondary surface (default is 8).
    tolerance : float, optional
        The relative tolerance for the natural element coordinates of the projected
        points (default is 0.1).
    facing : float, optional
        The minimum opposition of the outward unit normal vectors of a face-pair
        (default is 0.1).
    self_contact : bool, optional
        Flag to consider face-pairs which belong to the same body of the mesh (default
        is False).
    geometric_stiffness : bool, optional
        Flag to add the geometric part of the contact stiffness matrix (default is
        True). This is required for a quadratic rate of convergence.
    mask : ndarray of bool or None, optional
        A mask of the points of the mesh of the secondary solid body (default is None).
        Only the faces on the outline of the mesh with all their points in the mask are
        used. If None, all faces on the outline are used.
    mask_primary : ndarray of bool or None, optional
        A mask of the points of the mesh of the primary solid body, see ``mask``
        (default is None).

    Attributes
    ----------
    penalty : float or None
        The Nitsche parameter. This is None until it is estimated on the first assembly.
    stiffness : ndarray or None
        The stiffness :math:`k_b` of the bodies on the contact surfaces, given for the
        labels of the connected bodies of the mesh, see Eq. :eq:`nitsche-stiffness`.
        This is None until it is estimated on the first assembly.
    weighted : bool
        A flag which is True if the contact is evaluated on both surfaces of a contact
        zone with weighted contact tractions, see Eq. :eq:`nitsche-weights`.
    results : Results
        The results of the contact, e.g. the gap, the contact pressure and the normal
        component of the traction vector of the solid body (``traction``) at the
        integration points in contact.

    Notes
    -----
    In contrast to a penalty contact, see :class:`~felupe_contact.SolidBodyContact`,
    the contact pressure :math:`p_N` of Nitsche's method contains the normal component
    :math:`\lambda_N` of the traction vector of the solid body, see Eq.
    :eq:`nitsche-contact-pressure`, where :math:`\boldsymbol{P}` is the first
    Piola-Kirchhoff stress tensor, :math:`\boldsymbol{N}` is the outward unit normal
    vector of the undeformed contact surface and :math:`\boldsymbol{n}` is the outward
    unit normal vector of the primary surface at the projected point [1]_ [2]_.

    ..  math::
        :label: nitsche-contact-pressure

        p_N &= \left[ \lambda_N - \gamma\ g \right]_+

        \lambda_N &= \left( \boldsymbol{P} \boldsymbol{N} \right) \cdot \boldsymbol{n}

    The contact pressure replaces the normal stress of the solid body on the contact
    surface. At the exact solution, the contact pressure is equal to the normal stress
    of the solid body and the gap is zero in a contact zone, independent of the Nitsche
    parameter :math:`\gamma`, i.e. the formulation is consistent. At the discrete
    solution, the gap :math:`g = (\lambda_N - p_N) / \gamma` is given by the difference
    of the normal stress at the integration points and the contact pressure, which is
    in equilibrium with the stresses of the cells. Hence, the penetration doesn't grow
    with the contact pressure, like the penetration of a penalty contact, but with the
    discretization error of the stress at the contact surface. For a homogeneous stress
    state on matching meshes, the gap is zero (patch test). This is the non-symmetric
    variant of Nitsche's method (:math:`\theta = 0`), which requires the first
    derivative of the stress tensor only, i.e. the elasticity tensor of the material
    model. The Nitsche parameter must not be too small, i.e. it should scale with the
    stiffness of the body, like the estimate of Eq. :eq:`nitsche-stiffness`.

    The virtual work of the contact pressure is integrated at the integration points of
    the faces of the secondary surface, see Eq. :eq:`nitsche-weak-form`, like the
    contact potential of a penalty contact. The gap :math:`g` is evaluated by a
    closest-point projection of the integration points of the secondary surface on the
    primary surface. This is not a segment-based (mortar) integration, i.e. for
    non-matching meshes, the integrand is not smooth within a face of the secondary
    surface and the patch test is only passed approximately.

    ..  math::
        :label: nitsche-weak-form

        \delta W_c = -\int_\Gamma w\ p_N \left(
            \delta \boldsymbol{u} - \delta \bar{\boldsymbol{u}}
        \right) \cdot \boldsymbol{n}\ d\Gamma

    By default, the contact is evaluated on both contact surfaces, where the stress
    tensor of each solid body is used on its own surface (``two_pass=True``) [2]_. This
    removes the bias of the choice of the secondary surface. The contact tractions of
    an integration point of the body :math:`b`, which is projected on the body
    :math:`\bar{b}`, are weighted by the stiffness of both bodies, see Eq.
    :eq:`nitsche-weights`, like the weighted average of the tractions of a robust
    Nitsche's formulation for interface problems [3]_. Both weights of a contact zone
    sum up to one, i.e. the formulation is consistent. For equal bodies, both weights
    are one half. For a rubber body on a stiff body, the contact tractions of the
    stiff body vanish, i.e. the stress of the rubber body is used. Without these
    weights, the contact pressure of the stiff body would require a much higher Nitsche
    parameter. A single-pass contact (``two_pass=False``) is not weighted
    (:math:`w = 1`).

    ..  math::
        :label: nitsche-weights

        w = \frac{k_{\bar{b}}}{k_b + k_{\bar{b}}}

    The stiffness :math:`k_b` of a body is estimated from the mean of the diagonal
    entries :math:`\bar{k}_b` of the stiffness matrices of the given ``items``,
    evaluated on the degrees of freedom of the points of the body on the contact
    surfaces, and the mean area :math:`\bar{a}_b` per point, see Eq.
    :eq:`nitsche-stiffness`. This is the estimate of the penalty stiffness of
    :class:`~felupe_contact.SolidBodyContact`, evaluated per body of the mesh. Its
    order of magnitude is :math:`E / h`. The softest body is decisive for the Nitsche
    parameter.

    ..  math::
        :label: nitsche-stiffness

        k_b = \frac{\bar{k}_b}{\bar{a}_b}, \qquad
        \gamma = \text{scale} \cdot \min_b{k_b}

    The contact pressure is regularized with a smooth transition of length
    :math:`\delta` w.r.t. the effective penetration :math:`s = \lambda_N / \gamma - g`,
    see Eq. :eq:`nitsche-regularization`. Both the contact pressure and its derivative
    are continuous. In contrast to the regularized penalty law of
    :class:`~felupe_contact.SolidBodyContact`, the contact pressure is exact
    (unregularized) for :math:`s \ge \delta` and touching surfaces without a normal
    stress (:math:`s = 0`) are free of contact pressure. Within the transition zone,
    the regularization causes a penetration of at most :math:`4 \delta / 27`, i.e. less
    than one percent of the size of a face for the default :math:`\delta`,
    independent of the contact pressure.

    ..  math::
        :label: nitsche-regularization

        p_N = \begin{cases}
            0 & s \le 0 \\
            \gamma\ \dfrac{s^2 \left( 2 \delta - s \right)}{\delta^2}
                & 0 < s < \delta \\
            \gamma\ s & s \ge \delta
        \end{cases} \quad \text{with} \quad s = \frac{\lambda_N}{\gamma} - g

    The consistent stiffness matrix contains the change of the stress tensor of the
    solid body w.r.t. the displacements of all points of the cell of a face, see
    :meth:`SurfaceStress.matrix()
    <felupe_contact._nitsche.SurfaceStress.matrix>`. It is not symmetric. The material
    model is evaluated with own state variables at the integration points of the
    contact surfaces, which are updated after a converged increment like the state
    variables of the solid bodies. For a :class:`~felupe.SolidBodyNearlyIncompressible`,
    the cell-wise constant pressure of the solid body is included in the stress tensor.

    ..  note::

        Only solid bodies with a displacement field are supported, i.e. field
        containers with mixed fields are not supported. Use a
        :class:`~felupe.SolidBodyNearlyIncompressible` for (nearly) incompressible
        materials. The meshes of all contact surfaces must share the points with the
        mesh of the global field container, see :class:`~felupe_contact.SolidBodyContact`.

    ..  hint::

        The normal stress of a solid body is evaluated at the integration points of the
        faces of its contact surface. The accuracy of the contact depends on the
        accuracy of this stress: the stress of a coarse mesh of linear elements is not
        accurate and the stress of a :class:`~felupe.SolidBody` with a nearly
        incompressible material oscillates (volumetric locking). Use a
        :class:`~felupe.SolidBodyNearlyIncompressible` for rubber and a finer mesh at
        the contact surfaces or quadratic elements.

    ..  hint::

        Use two solid bodies (or the masks of the contact surfaces) for a contact
        between bodies of different stiffness, e.g. a rubber body on a metal body. The
        stiffness of each body is estimated from the given ``items``, i.e. if a single
        solid body with different materials is given, its stiffness matrix contains the
        stiffness of all bodies of the mesh.

    Examples
    --------
    A rubber block is pressed on a stiff base. Both blocks are meshed individually and
    are combined to a single mesh. Each block is a solid body with its own material
    model: the rubber block is nearly incompressible and the base is a thousand times
    stiffer.

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>> import numpy as np
        >>>
        >>> from felupe_contact import SolidBodyContactNitsche
        >>>
        >>> base = fem.Cube(a=(0, 0, 0), b=(1, 1, 0.5), n=(5, 5, 3))
        >>> block = fem.Cube(a=(0.15, 0.15, 0.52), b=(0.85, 0.85, 1.12), n=(4, 4, 4))
        >>> container = fem.MeshContainer([base, block], merge=True)
        >>> mesh = container.stack()
        >>>
        >>> field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
        >>> fields = [
        ...     fem.FieldContainer([fem.Field(fem.RegionHexahedron(m), dim=3)])
        ...     for m in container.meshes
        ... ]
        >>> for f in fields:
        ...     f.link(field)
        >>>
        >>> metal = fem.SolidBody(fem.NeoHooke(mu=1000.0, bulk=5000.0), fields[0])
        >>> rubber = fem.SolidBodyNearlyIncompressible(
        ...     fem.NeoHooke(mu=1.0), fields[1], bulk=100.0
        ... )

    The contact surfaces are created on the outlines of the meshes of both solid
    bodies. The secondary surface is the rubber surface.

    ..  pyvista-plot::
        :context:

        >>> contact = SolidBodyContactNitsche(rubber, metal)

    The bottom face of the base is fixed and the top face of the rubber block is moved
    downwards. A job with several solid bodies requires the global field container.

    ..  pyvista-plot::
        :context:

        >>> boundaries = {
        ...     "fixed": fem.Boundary(field[0], fz=0.0),
        ...     "clamped": fem.Boundary(field[0], fz=mesh.z.max(), skip=(0, 0, 1)),
        ...     "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0)),
        ... }
        >>> move = fem.math.linsteps([0, -0.15], num=3)
        >>> step = fem.Step(
        ...     items=[metal, rubber, contact],
        ...     ramp={boundaries["move"]: move},
        ...     boundaries=boundaries,
        ... )
        >>> job = fem.Job(steps=[step]).evaluate(x0=field, verbose=0)

    The contact tractions of the integration points of the rubber surface are weighted
    by about 0.99 and those of the metal surface by about 0.01, i.e. mainly the stress
    of the rubber block is used. The penetration is about a tenth of a percent of the
    size of a face, i.e. ten times smaller than the penetration of a penalty contact
    with the same estimated stiffness.

    ..  pyvista-plot::
        :context:

        >>> gap = np.concatenate([g for g in contact.results.gap if g is not None])
        >>> bool(-gap.min() < 2e-3 * contact.size)
        True

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> rubber.plot("Principal Values of Cauchy Stress").show()

    References
    ----------
    ..  [1] F. Chouly and P. Hild, "A Nitsche-based method for unilateral contact
        problems: numerical analysis", SIAM Journal on Numerical Analysis, vol. 51,
        no. 2, pp. 1295–1307, 2013. DOI: 10.1137/12088344X.
    ..  [2] R. Mlika, Y. Renard and F. Chouly, "An unbiased Nitsche's formulation of
        large deformation frictional contact and self-contact", Computer Methods in
        Applied Mechanics and Engineering, vol. 325, pp. 265–288, 2017. DOI:
        10.1016/j.cma.2017.07.015.
    ..  [3] C. Annavarapu, M. Hautefeuille and J. E. Dolbow, "A robust Nitsche's
        formulation for interface problems", Computer Methods in Applied Mechanics and
        Engineering, vol. 225–228, pp. 44–54, 2012. DOI: 10.1016/j.cma.2012.03.008.

    See Also
    --------
    felupe_contact.SolidBodyContact : A penalty contact between the surfaces of solid
        bodies with optional Coulomb friction.
    """

    def __init__(
        self,
        solid,
        solid_primary=None,
        items=None,
        penalty=None,
        penalty_scale=10.0,
        smoothing=None,
        two_pass=True,
        max_distance=None,
        max_gap=None,
        candidates=8,
        tolerance=0.1,
        facing=0.1,
        self_contact=False,
        geometric_stiffness=True,
        mask=None,
        mask_primary=None,
    ):
        solids = [solid] if solid_primary is None else [solid, solid_primary]

        for body in solids:
            validate(body)

        super().__init__(
            solid,
            solid_primary,
            items=items,
            penalty=penalty,
            penalty_scale=penalty_scale,
            smoothing=smoothing,
            two_pass=two_pass,
            max_distance=max_distance,
            candidates=candidates,
            tolerance=tolerance,
            facing=facing,
            self_contact=self_contact,
            geometric_stiffness=geometric_stiffness,
            mask=mask,
            mask_primary=mask_primary,
        )

        # a contact surface which is also the primary surface is evaluated on both
        # surfaces of a contact zone, i.e. a second pass would count the contact
        # tractions twice
        shared, same = self._overlap()

        if same:
            self.pairs = self.pairs[:1]

        elif shared > 0:
            warnings.warn(
                f"The contact surfaces share {shared} faces, but they are not equal. "
                "The contact tractions of the integration points on the shared faces "
                "are counted twice. Use either equal or disjoint contact surfaces."
            )

        # the contact is evaluated on both surfaces of a contact zone, i.e. the contact
        # tractions of the integration points are weighted by the stiffness of the
        # bodies (instead of the constant weights of the surface pairs)
        self.weighted = same or two_pass

        if self.weighted:
            for pair in self.pairs:
                pair.weight = 1.0

        # the solid body of the secondary surface of each surface pair
        bodies = {id(self.field): solids[0], id(self.field_primary): solids[-1]}
        self.surfaces = [
            SurfaceStress(bodies[id(pair.field)], pair.field) for pair in self.pairs
        ]

        self.max_gap = max_gap
        if max_gap is None:
            self.max_gap = 0.5 * self.size

        # the state variables of the material models at the integration points of the
        # contact surfaces
        self.results.statevars = {
            "material": [surface.statevars() for surface in self.surfaces]
        }
        self.results.traction = []

        self.stiffness = None
        self._traction = None
        self._weights = None

    def _overlap(self):
        """Return the number of faces, which are shared by the secondary and the primary
        surface, and a flag if both surfaces are equal."""

        if self.field is self.field_primary:
            return len(self.field.region.mesh.cells_faces), True

        faces = [
            np.unique(np.sort(f.region.mesh.cells_faces, axis=1), axis=0)
            for f in [self.field, self.field_primary]
        ]
        unique = len(np.unique(np.concatenate(faces), axis=0))
        shared = len(faces[0]) + len(faces[1]) - unique

        return shared, shared == len(faces[0]) == len(faces[1])

    def __repr__(self):
        header = "<felupe SolidBodyContactNitsche object>"
        penalty = f"  Nitsche parameter: {self.penalty}"
        smoothing = f"  Smoothing: {self.smoothing}"
        contact = f"  Integration points in contact: {self.results.npoints_in_contact}"

        return "\n".join([header, penalty, smoothing, contact])

    def checkpoint(self):
        """Return a checkpoint of the state of the contact search, of the state
        variables and of the results, see :meth:`SolidBodyContact.checkpoint()
        <felupe_contact.SolidBodyContact.checkpoint>`. The normal components of the
        traction vectors are included.

        Returns
        -------
        dict
            The checkpoint of the contact.
        """

        checkpoint = super().checkpoint()
        checkpoint["traction"] = list(self.results.traction)

        return checkpoint

    def restore(self, checkpoint):
        """Restore a checkpoint of the contact inplace, see :meth:`checkpoint`.

        Parameters
        ----------
        checkpoint : dict
            A dict with the checkpoint of the contact.
        """

        super().restore(checkpoint)

        self.results.traction = list(checkpoint["traction"])
        self._traction = None
        self._weights = None

    def pressure(self, gap, traction=0.0):
        r"""Return the contact pressure and its derivative w.r.t. the gap, evaluated by
        a regularized Nitsche law, see Eq. :eq:`nitsche-regularization`.

        Parameters
        ----------
        gap : ndarray
            The gap between the contact surfaces.
        traction : ndarray or float, optional
            The normal component of the traction vector of the solid body (default is
            0.0).

        Returns
        -------
        pressure : ndarray
            The contact pressure.
        dpressure : ndarray
            The derivative of the contact pressure w.r.t. the gap. The derivative
            w.r.t. the normal component of the traction vector is
            ``-dpressure / penalty``.
        """

        penalty, smoothing = self.penalty, self.smoothing
        penetration = traction / penalty - np.asarray(gap)

        pressure = np.zeros_like(penetration)
        dpressure = np.zeros_like(penetration)

        if smoothing > 0:
            closed = penetration >= smoothing
            transition = np.logical_and(penetration > 0, penetration < smoothing)

            s = penetration[transition]
            pressure[transition] = penalty * s**2 * (2 * smoothing - s) / smoothing**2
            dpressure[transition] = (
                -penalty * s * (4 * smoothing - 3 * s) / smoothing**2
            )
        else:
            closed = penetration > 0

        pressure[closed] = penalty * penetration[closed]
        dpressure[closed] = -penalty

        return pressure, dpressure

    def update_stiffness(self):
        r"""Estimate and update the stiffness of the bodies on the contact surfaces,
        see Eq. :eq:`nitsche-stiffness`.

        Returns
        -------
        ndarray
            The stiffness :math:`k_b` of each body of the mesh, given for the labels of
            the connected bodies. The stiffness of a body without a face on the contact
            surfaces is NaN.
        """

        pair = self.pairs[0]
        diagonal = np.abs(self._diagonal())

        surfaces = [(pair.cells_faces, pair.dA, pair.body)]

        if pair.field_primary is not pair.field:
            surfaces.append(
                (pair.cells_faces_primary, pair.dA_primary, pair.body_primary)
            )

        faces = np.concatenate([s[0] for s in surfaces])
        area = np.concatenate([s[1].sum(axis=0) for s in surfaces])
        body = np.concatenate([s[2] for s in surfaces])

        # faces which belong to both contact surfaces are counted once
        unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)[1]
        faces, area, body = faces[unique], area[unique], body[unique]

        stiffness = np.full(body.max() + 1, np.nan)

        for label in np.unique(body):
            selected = body == label
            points = np.unique(faces[selected])
            mean = diagonal[pair.dof[points].ravel()].mean()
            stiffness[label] = mean * len(points) / area[selected].sum()

        if not np.all(stiffness[body] > 0):
            raise ValueError(
                "The estimated stiffness of a body on the contact surfaces is zero, i.e. "
                "the given `items` don't contain all solid bodies of the contact "
                "surfaces."
            )

        self.stiffness = stiffness

        return stiffness

    def update_penalty(self):
        r"""Estimate and update the Nitsche parameter from the stiffness of the softest
        body on the contact surfaces, see Eq. :eq:`nitsche-stiffness`.

        Returns
        -------
        float
            The estimated Nitsche parameter.
        """

        stiffness = self.update_stiffness()
        self.penalty = self.penalty_scale * np.nanmin(stiffness)

        return self.penalty

    def weights(self, pair, kinematics):
        r"""Return the weights of the contact tractions of the integration points in
        contact, see Eq. :eq:`nitsche-weights`.

        Parameters
        ----------
        pair : ContactSurfacePair
            The surface pair.
        kinematics : dict
            The contact kinematics.

        Returns
        -------
        ndarray of shape (npairs,) or float
            The weights of the contact tractions. A single-pass contact is not
            weighted, i.e. its weight is one.
        """

        if not self.weighted:
            return 1.0

        body = pair.body[kinematics["point"] % pair.ncells]
        body_primary = pair.body_primary[kinematics["face"]]

        stiffness = self.stiffness[body]
        stiffness_primary = self.stiffness[body_primary]

        return stiffness_primary / (stiffness + stiffness_primary)

    def _extract(self, field=None, parallel=False):
        """Evaluate and cache the contact kinematics of all surface pairs, together
        with the stress tensors of the solid bodies at the integration points of the
        contact surfaces and the trial state variables of the material models. Only the
        integration points with a positive contact pressure are kept. The cache is
        re-evaluated for new displacements and for new (converged) state variables."""

        if field is not None:
            self.field = field

        values = self.field[0].values
        self.field_primary[0].values = values

        if (
            self._kinematics is not None
            and np.array_equal(self._values, values)
            and self._state is self.results.statevars
        ):
            return self._kinematics

        if self.penalty is None:
            self.update_penalty()

        if self.stiffness is None:
            self.update_stiffness()

        x = self.field.region.mesh.points + values
        statevars = self.results.statevars["material"]

        kinematics, tractions, weights, material = [], [], [], []

        for pair, surface, sv in zip(self.pairs, self.surfaces, statevars):
            kin = pair.kinematics(
                x=x,
                max_distance=self.max_distance,
                candidates=self.candidates,
                tolerance=self.tolerance,
                facing=self.facing,
                self_contact=self.self_contact,
                workers=-1 if parallel else 1,
                max_gap=self.max_gap,
            )

            # the stress is evaluated at all integration points of the contact surface,
            # i.e. the state variables of the material model evolve also without contact
            evaluated = surface.evaluate(values, sv)
            material.append(evaluated["statevars"])

            traction, weight = None, None

            if kin is not None:
                traction = surface.traction(kin, values, evaluated)
                pressure, dpressure = self.pressure(kin["gap"], traction["normal"])

                # integration points with a positive gap are considered by the contact
                # search, but only those with a positive contact pressure are in contact
                active = (pressure > 0) | (dpressure != 0)

                if np.any(active):
                    kin, traction = select(kin, active), select(traction, active)
                    weight = self.weights(pair, kin)
                else:
                    kin, traction = None, None

            kinematics.append(kin)
            tractions.append(traction)
            weights.append(weight)

        self._kinematics = kinematics
        self._traction = tractions
        self._weights = weights
        self._values = values.copy()
        self._state = self.results.statevars

        self.results._statevars = {"material": material}

        self.results.gap = [None if k is None else k["gap"] for k in kinematics]
        self.results.traction = [None if t is None else t["normal"] for t in tractions]
        self.results.pressure = [
            None if k is None else self.pressure(k["gap"], t["normal"])[0]
            for k, t in zip(kinematics, tractions)
        ]
        self.results.npoints_in_contact = sum(
            [0 if k is None else len(k["gap"]) for k in kinematics]
        )
        self.results.shear = [None] * len(self.pairs)
        self.results.slip = [None] * len(self.pairs)

        return self._kinematics

    def _vector(self, field=None, parallel=False, resize=None):
        """Assemble the sparse contact force vector. The vectors of the face-pairs of
        all surface pairs are summed up at once."""

        kinematics = self._extract(field=field, parallel=parallel)

        values, dofs = [], []
        for pair, kin, pressure, weight in zip(
            self.pairs, kinematics, self.results.pressure, self._weights
        ):
            if kin is not None:
                values.append(pair.vector(kin, -weight * pressure))
                dofs.append(pair.dofs(kin))

        return self._assemble_vector(values, dofs, resize=resize)

    def _matrix(self, field=None, parallel=False, resize=None):
        """Assemble the sparse contact stiffness matrix. The matrices of the face-pairs
        are given for the degrees of freedom of the cells of the faces of the secondary
        surface and of the faces of the primary surface."""

        kinematics = self._extract(field=field, parallel=parallel)
        statevars = self.results.statevars["material"]

        values, dofs = [], []
        for pair, surface, kin, traction, weight, sv in zip(
            self.pairs,
            self.surfaces,
            kinematics,
            self._traction,
            self._weights,
            statevars,
        ):
            if kin is not None:
                pressure, dpressure = self.pressure(kin["gap"], traction["normal"])

                # the weights are constant, i.e. the weighted contact pressure and its
                # derivative are passed to the stiffness matrix
                values.append(
                    surface.matrix(
                        pair,
                        kin,
                        traction,
                        sv,
                        weight * pressure,
                        weight * dpressure,
                        self.penalty,
                        geometric=self.geometric_stiffness,
                    )
                )
                dofs.append(surface.dofs(pair, kin))

        return self._assemble_matrix(values, dofs, resize=resize)
