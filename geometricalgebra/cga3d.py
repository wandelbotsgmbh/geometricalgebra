"""Handling 3d CGA Tensors"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations
from typing import Iterable, Tuple, Union

import numpy as np

# from jax import Array
from geometricalgebra.algebra import GeometricAlgebra
from geometricalgebra.cga import CGAVector

Array = None

ALGEBRA = GeometricAlgebra((1, 1, 1, 1, -1))


class Vector(CGAVector):
    """A 3d CGA Tensor"""

    @classmethod  # type: ignore
    @property
    def algebra(cls):
        return ALGEBRA

    @classmethod
    def e_0(cls):
        return e_0

    @classmethod
    def e_1(cls):
        return e_1

    @classmethod
    def e_2(cls):
        return e_2

    @classmethod
    def e_3(cls):
        return e_3

    @classmethod
    def e_inf(cls):
        return e_inf

    @classmethod
    def i3(cls):
        return i3

    @classmethod
    def i5(cls):
        return i5

    @classmethod
    def minkovski_plane(cls):
        return minkovski_plane

    def __init__(self, values, grade: Union[int, Iterable[int]], algebra=GeometricAlgebra((1, 1, 1, 1, -1))):
        algebra = algebra or GeometricAlgebra((1, 1, 1, 1, -1))
        if algebra != GeometricAlgebra((1, 1, 1, 1, -1)):
            raise TypeError(f"Unexpected algebra {algebra} expected {GeometricAlgebra((1, 1, 1, 1, -1))} instead")
        super().__init__(values, grade)

    def to_barycentric_coordinates(self) -> Tuple[Array, Vector]:  # type: ignore
        """Represent points (grade 1) in barycentric coordinated with a tetraeder

        Example:
        >>> x = Vector.from_euclid([1, 2, 3])
        >>> coordinates, control_points = x.to_barycentric_coordinates()
        >>> np.asarray(control_points.to_direction(False).T @ coordinates)
        array([1., 2., 3.])
        """
        return TETRAHEDRON_GEN.scalar_product(self[..., None]), TETRAHEDRON + e_0

    def to_barycentric_coordinates_full(self, grade=1) -> Tuple[Array, Vector]:  # type: ignore
        """Represent points (grade 1) in barycentric coordinated with a tetraeder

        Example:
        >>> x = Vector.from_euclid([1, 2, 3])
        >>> coordinates, control_points = x.to_barycentric_coordinates()
        >>> np.asarray(control_points.to_direction(False).T @ coordinates)
        array([1., 2., 3.])
        """
        # assert self.grade == grade
        a, control_points = _barycentric_coordinates_transformation_matrix(grade)
        return self.xnp().einsum("ij,...j->...i", a, self._values), control_points


e_1, e_2, e_3, e_plus, e_minus = Vector.basis()
e_inf = e_plus + e_minus
e_0 = (e_minus - e_plus) / 2
i3 = e_1 ^ e_2 ^ e_3
i5 = e_1 ^ e_2 ^ e_3 ^ e_inf ^ e_0
euclidean_basis = Vector.stack([e_1, e_2, e_3])
quaternion_basis = Vector.concatenate([Vector.from_scalar(1)[None], euclidean_basis.dual(i3)])
translation_basis = euclidean_basis ^ e_inf
homogeneous_basis = Vector.stack([e_1, e_2, e_3, e_0])
homogeneous_basis_gen = Vector.stack([e_1, e_2, e_3, -e_inf])
multivector_basis = [Vector(np.eye(32), range(6))[e_0.algebra.grade_to_slice[grade]](grade) for grade in range(6)]
conformal_basis = Vector(np.eye(32), range(6))
minkovski_plane = e_inf ^ e_0


TETRAHEDRON = Vector.from_direction(np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]))
TETRAHEDRON_GEN = (
    -1 / 4 * Vector.from_translator(e_inf ^ TETRAHEDRON.normed() / 3**0.5).apply(TETRAHEDRON.dual())
).dual()

FRAME = sum([e_0 + 3 * e_inf / 2, e_0 ^ e_1 ^ e_inf, (e_0 ^ e_2 ^ e_inf).dual(), e_3.dual()])
POSE_ORIGIN = Vector.concatenate([e_0[None], TETRAHEDRON / 2])

BASIS_LINE = [
    Vector.concatenate([(euclidean_basis.dual(i3) ^ e_0), euclidean_basis ^ e_inf ^ e_0]),
    Vector.concatenate([(euclidean_basis.dual(i3) ^ e_inf), euclidean_basis ^ e_inf ^ e_0]),
]

ALL_2PI_ORIENTATIONS = Vector.from_quaternion(np.eye(4))
ALL_PI_ORIENTATIONS = Vector.from_quaternion(
    np.concatenate(
        [
            [(1, 0, 0, 0)],
            list(set(permutations([0, 1, 0, 0])).difference([(1, 0, 0, 0)])),
            list(set(permutations([0.5**0.5, 0.5**0.5, 0, 0]))),
            list(i for i in set(permutations([0.5**0.5, -(0.5**0.5), 0, 0])) if np.cumsum(i).sum() > 0),
            [[0.5, i, j, k] for i in [-0.5, 0.5] for j in [-0.5, 0.5] for k in [-0.5, 0.5]],
        ]
    )
)


@lru_cache
def _barycentric_coordinates_transformation_matrix(grade=None):
    if grade is None or grade == 1:
        control_points = TETRAHEDRON + e_0
    else:
        control_points = Vector(np.eye(32), range(6))(grade)[
            e_0._algebra.grade_to_slice[grade]  # pylint: disable=protected-access
        ]
    return np.linalg.pinv(control_points._values).T, control_points  # pylint: disable=protected-access


def fit_points_to_plane(points):
    """Fit points to plane by least squares.

    Args:
        points: the points which shall be fitted

    Returns:
        fitted plane

    """
    basis = Vector.stack([*euclidean_basis, e_inf]).dual()
    a = basis ^ points[:, None]
    mat = -sum(a[:, :, None].scalar_product(a[:, None, :]))
    return sum(basis * np.linalg.eigh(mat)[1][:, 0]).normed()


def pose_distance(a, b, mode="both"):
    tmp = Vector.concatenate([e_0[None] * 2**0.5, TETRAHEDRON / 2**1.5])
    if mode == "both":
        reference = tmp
    elif mode == "position":
        reference = tmp[:1]
        return -a[..., None].apply(reference).checked().product(b[..., None].apply(reference), 0).sum(-1).to_scalar()
    elif mode == "orientation":
        reference = tmp[1:]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return 3 / 2 - a[..., None].apply(reference).checked().product(b[..., None].apply(reference), 0).sum(-1).to_scalar()
    # Use this one if it is supported
    # return -a.apply(FRAME).checked().scalar_product(b.apply(FRAME))


def get_origin_pose(weight: float = 1.0):
    """Get an multivector tuple which can calculate pose differences by the inner product"""
    return Vector.concatenate(
        [e_0[None], weight * TETRAHEDRON / 2, weight * Vector.from_scalar(np.sqrt(3), pseudo=True)[None]]
    )
