# pylint: disable=too-many-lines
"""Elementary data types for geometric algebra
"""
from __future__ import annotations

from abc import abstractmethod
from typing import List, Tuple, Type, TypeVar

import numpy as np
from scipy.spatial.transform import Rotation

from geometricalgebra.algebra import ProductType
from geometricalgebra.vector import Array, ArrayLike, Vector, allclose  # noqa: F401

Subtype = TypeVar("Subtype", bound="CGAVector")


class CGAVector(Vector):  # pylint: disable=too-many-public-methods
    """Array of multi vectors in 3d conformal geometric algebra

    The last dimension is the vector (e.g. 32 dimensional for conformal 3D space)

    Represented in basis {1, e1, e2, e3, e+, e-, ...}

    All geometric entities are given on geometric outer product null space [1, Chap 4.3.7]

    References:
        [1] "Geometric Algebra with Applications in Engineering, Perwas 2009

    OBSTACLES:
        __radd__: see test_scalar_addition for documentation
    """

    @classmethod
    @abstractmethod
    def e_0(cls):
        pass

    @classmethod
    @abstractmethod
    def e_1(cls):
        pass

    @classmethod
    @abstractmethod
    def e_2(cls):
        pass

    @classmethod
    @abstractmethod
    def e_3(cls):
        pass

    @classmethod
    @abstractmethod
    def i3(cls):
        pass

    @classmethod
    @abstractmethod
    def minkovski_plane(cls):
        pass

    @classmethod
    @abstractmethod
    def i5(cls):
        pass

    def apply(self: Subtype, other) -> Subtype:
        if apply_versor := getattr(other, "__apply_versor__", None):
            return apply_versor(self)
        return self.sandwich(other)

    @classmethod
    @abstractmethod
    def e_inf(cls):
        pass

    @classmethod
    def from_euclid(cls: Type[Subtype], vectors: ArrayLike) -> Subtype:
        """Create a positional vector from Euclidean points (via Hestenes mapping)

        Args:
            vectors: any array of shape (..., 3)

        Returns:
            multivector tensor of shape (..., 32) where each vector is special normalized
        """
        result = cls.from_direction(vectors).up()
        return result

    @classmethod
    def from_euclid_2d(cls: Type[Subtype], vectors: ArrayLike) -> Subtype:
        """Create a positional vector from Euclidean points in 2D (via Hestenes mapping) in the xy-plane

        Args:
            vectors: any array of shape (..., 2)

        Returns:
            multivector tensor of shape (..., 32) where each vector is special normalized
        """
        vectors = cls.xnp().asarray(vectors)
        vectors = cls.xnp().pad(vectors, [*([(0, 0)] * (vectors.ndim - 1)), (0, 1)], mode="constant")
        return cls.from_euclid(vectors)

    def up(self: Subtype) -> Subtype:
        """Performs a stereographic embedding and the homogenization (conformal mapping)"""
        return self + (self.square() ^ self.e_inf() / 2) + self.e_0()

    def down(self: Subtype) -> Subtype:
        """
        >>> from geometricalgebra import cga3d
        >>> a = cga3d.Vector.from_direction([1, 2, 3])
        >>> np.asarray(a.up().down().values)
        array([0., 1., 2., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        """
        return type(self).from_direction(self.normed_special().to_direction(False))

    def is_flat(self, eps=1e-6) -> Array:
        """Return True if segment is line and False is segment if segment is a circular arc
        >>> from geometricalgebra import cga3d
        >>> a, b, c, d = cga3d.Vector.from_euclid([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        >>> np.asarray(b.is_flat())
        array(False)
        >>> np.asarray((b ^ c).is_flat())  # point pair
        array(False)
        >>> np.asarray((b ^ cga3d.e_inf).is_flat())  # direction
        array(True)
        >>> np.asarray((b ^ c ^ d).is_flat())  # circle
        array(False)
        >>> np.asarray((b ^ c ^ cga3d.e_inf).is_flat())  # line
        array(True)
        >>> np.asarray((a ^ b ^ c ^ d).is_flat())  # sphere
        array(False)
        >>> np.asarray((b ^ c ^ d ^ cga3d.e_inf).is_flat())  # plane
        array(True)
        """
        return abs((self ^ self.e_inf()).square_norm()) < eps

    @classmethod
    def from_direction(cls: Type[Subtype], vectors: ArrayLike, dtype=np.float64) -> Subtype:
        """Create a directional vector

        Example:
        >>> from geometricalgebra import cga3d
        >>> cga3d.Vector.from_direction([[1, 2, 3]])
        Vector([[0., 1., 2., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        """
        vectors = cls.xnp().asarray(vectors, dtype=dtype)
        paddings = [(0, 0)] * (len(vectors.shape) - 1) + [(0, 2)]
        vectors = cls.xnp().pad(vectors, paddings, mode="constant")
        return cls(vectors, 1)

    def to_direction(self, normalize: bool) -> Array:
        """Extract the direction from a vector"""
        euclid_dim = len(self._algebra.signature) - 2
        if 1 in self.grades:
            v = self(1).normed() if normalize else self(1)
            result = v._values[..., :euclid_dim]  # pylint: disable=protected-access
            return result
        shape = [*[1 if s is None else s for s in self.shape], euclid_dim]
        return self.xnp().zeros(shape, self.dtype)

    @classmethod
    def from_sphere(cls: Type[Subtype], center: ArrayLike, radius: ArrayLike) -> Subtype:
        """Create a sphere (grade-four) multivector"""
        return (cls.from_euclid(center) - cls.xnp().square(radius) / 2 * cls.e_inf()).dual()

    @classmethod
    def from_plane(cls: Type[Subtype], normal: ArrayLike, distance: ArrayLike) -> Subtype:
        """Create a plane (grade-four) multivector from a parametric plane: ax+by+cz=d.

        Example:
        >>> from geometricalgebra import cga3d
        >>> cga3d.Vector.from_plane([1, 2, 3], 1.5)
        Vector([ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                 0. ,  0. ,  0. ,  0. ,  1.5,  1.5, -3. ,  2. , -1. ,  0. ])

        """
        return (cls.from_direction(normal) + distance * cls.e_inf()).dual()

    def dual(self: Subtype, subspace=None) -> Subtype:
        """The dual defined by $A I_5^{-1}$"""
        subspace = self.i5() if subspace is None else subspace
        if subspace in (self.i5(), self.i3()):  # TODO i3
            return (-self & subspace)({subspace.grade - i for i in self._grades})
        raise NotImplementedError()

    def meet(self: Subtype, other) -> Subtype:
        """Meet defined by $(A ∨ B)^*  := B^* ∧ A^*"""
        return (self.dual() ^ other.dual()).dual()

    def normed(self: Subtype, eps: float = 0, reverse_norm: bool = False) -> Subtype:
        """Normalize the vector

        Args:
            eps: provide to avoid numerical division by zero
            reverse_norm: if True the reverse_norm is used instead

        Returns:
            The normed vectors
        """
        if reverse_norm:
            square_norm = self.reverse_norm()
        else:
            square_norm = self.square_norm()
        return self / self.xnp().sqrt(eps + abs(square_norm))

    def normed_special(self: Subtype) -> Subtype:
        """Return a normalized vector such that :math:`x \\cdot e_{\\inf} = -1`

        Example:
        >>> from geometricalgebra import cga3d
        >>> a = 2.3 * cga3d.Vector.from_euclid((0, 2, 3))
        >>> np.asarray((a.normed_special() | cga3d.e_inf).to_scalar())
        array(-1.)
        """
        assert self.grades == {1}, "Must be a grade-one blade"
        return -self / self.inner_prod(self.e_inf(), 0).to_scalar()

    def normed_unsigned_sphere(self: Subtype) -> Subtype:
        return ((self ^ self.e_inf()).to_scalar(pseudo=True) * self).normed()

    def to_euclid(self) -> Array:
        """Return the 1 grade as Euclidean vector (tensor)

        Example:
        >>> from geometricalgebra import cga3d
        >>> np.asarray(cga3d.Vector.from_euclid([1.4, 2, 0]).to_euclid())
        array([1.4, 2. , 0. ])
        """
        normed = self.normed_special()(1)
        return normed.to_direction(False)

    def center(self: Subtype) -> Subtype:
        """Return center point of point a round

        Note: prefactor (including sign) might change in future"""
        return -(self & self.e_inf() & self)(1)
        # TODO sandwiching

    def normal(self: Subtype) -> Subtype:
        """Return the normal direction of a plane

        Example:
        >>> from geometricalgebra import cga3d
        >>> a, b, c = cga3d.Vector.from_euclid([(1, 0, 0), (0, 1, 0), (0, 0, 2)])
        >>> plane = a ^ b ^ c ^ cga3d.e_inf
        >>> np.asarray(plane.normal().to_direction(True))
        array([0.66666667, 0.66666667, 0.33333333])
        """
        return -self.dual() - ((self.dual() | self.e_0()) ^ self.e_inf())

    def isclose(self, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
        """Returns a boolean array where two arrays are element-wise equal within a tolerance.

        NOTE: these function was introduced to handle addition with scalars properly (which differs from np.ndarray)
        """
        return np.isclose(self.view(np.ndarray), b.view(np.ndarray), rtol, atol, equal_nan)

    def is_inside_sphere(self: Subtype, points: Subtype, eps=0) -> Array:
        """Return True is point(s) are inside the sphere

        Args:
            points: shall be conformal vector (points)
            eps: Numerical inaccuracy which is still treated as being inside

        Returns:
            Array whether the points are inside the shpere

        Note
        Example:
        >>> from geometricalgebra import cga3d
        >>> radius = 1
        >>> sphere = (cga3d.Vector.from_euclid((1, 2, 3)) - radius**2 / 2 * cga3d.e_inf).dual()
        >>> np.asarray(sphere.is_inside_sphere(cga3d.Vector.from_euclid((1, 2, 3))))
        array(True)
        >>> np.asarray(sphere.is_inside_sphere(cga3d.Vector.from_euclid((2.1, 2, 3))))
        array(False)
        """
        return points(1).inner_prod(self.dual()(1), 0).to_scalar() < eps

    def to_point_direction(self: Subtype) -> Tuple[Subtype, Subtype]:
        point = self.center().normed_special()
        direction = (self | self.e_inf()) ^ self.e_inf()
        return point, direction

    def circle_to_center_normal_radius(
        self: Subtype,
    ) -> Tuple[Subtype, Subtype, Subtype]:
        """
        Note that radius square might be negative
        """
        sphere, plane = self.circle_to_sphere_and_plane()
        center, square_radius = sphere.sphere_to_center_squared_radius()
        radius = square_radius.sqrt()
        normal = -plane.dual() + ((plane.dual() | self.e_0()) ^ self.e_inf())
        return center, normal.normed().up(), radius

    def circle_to_sphere_and_plane(self: Subtype) -> Tuple[Subtype, Subtype]:
        """Returns that sphere and a plane whose meet is the circle"""
        sphere = to_sphere(self)
        plane = self ^ self.e_inf()
        return sphere, plane

    def sphere_to_center_squared_radius(self: Subtype) -> Tuple[Subtype, Subtype]:
        squared_radius = self.dual().normed_special().square()
        return self.center(), squared_radius

    def line_to_point_direction(self: Subtype) -> Tuple[Subtype, Subtype]:
        # compare with plane to normal
        l_star = self & self.i5()
        T = self.e_0() | l_star
        m_hat = -(l_star - (T & self.e_inf())) & self.i3()
        m_hat = m_hat(1)
        p: Subtype = project_to_flat(self.e_0(), onto=self)
        return p, m_hat

    def point_pair_to_end_points(self: Subtype, normalize: bool = True) -> Subtype:
        """Return start and end (normed_special) assuming it is a point pair

        TODO: why does this work?
        Reference:
            Ch. 4.1.1 in Lasenby, Lasenby, & Wareham, A Covariant Approach to Geometry using Geometric Algebra.
        """
        a = self.stack([-self, self]) + abs(self.square()).sqrt()
        result = a.geometric_prod(self | (self.e_0() + self.e_inf()), 1)
        if normalize:
            return result.normed_special()
        return result

    def on_same_circle(self) -> bool:
        """Assumes all multivectors are points. Returns True if all points share exactly the same circle."""
        # at least 3 points are required to form a circle
        circles = (self[:-2] ^ self[1:-1] ^ self[2:]).normed(eps=1e-8)
        return all(
            self.xnp().allclose(
                self.xnp().abs(circles[i].values),
                self.xnp().abs(circles[i + 1].values),
                atol=1e-06,
            )
            for i in range(len(circles) - 1)  # TODO abs is not sufficient
        )

    def dyadic_square(self, product_type: ProductType, internal_grade=None, dtype=None) -> Array:
        """Compute the dyadic product with itself

        Args:
            product_type: the kind of dyadic product
            internal_grade: mask
            dtype: the dtype of the result

        Returns:
            array of shape (*self.shape, 32, 32)

        Example:
        >>> from geometricalgebra import cga3d
        >>> a, b = cga3d.e_0, cga3d.e_1.up()
        >>> np.asarray((a ^ b).square_norm())
        array(0.25)
        >>> np.einsum("i,ij,j", b.values, a.dyadic_square(ProductType.OUTER), b.values)  # equivalent to above
        0.25
        """
        mask = self._algebra.grade_to_slice[internal_grade] if internal_grade is not None else slice(None)
        a = self.left_product_matrix(product_type)[..., mask, :]
        result = np.einsum(
            "...ik,i,...il->...kl",
            a,
            (self._algebra.get_adjoint_generator() * self._algebra.get_square_norm_generator())[mask],
            a,
            dtype=dtype,
        )
        return result

    @classmethod
    def from_motor_estimation(
        cls,
        p: Subtype,
        q: Subtype,
        *,
        translation: bool = True,
        rotation: bool = True,
        dilation: bool = False,
        only_2d: bool = False,
    ):
        """Estimate optimal rotor given geometric entities in two different frames

        Args:
            p: geometric entities in the initial frame
            q: same entities as in p but in the displaced frame
            translation: whether to allow translation
            rotation: whether to allow rotation
            dilation: whether to allow dilation (i.e., scaling). Note that this activated
                dilation it is no longer a rigid body motion (and thus no motor anymore)
            only_2d: Whether only 2d motors are valid

        Returns:
            motor that maps p to the entities q (as far as possible)

        Reference:
            Valkenburg, Dorst, "Estimating Motors from a Variety of Geometric Data in 3D Conformal Geometric Algebra" (2011)

        """
        xnp = p.xnp()
        if not p.shape:
            p = p[None]
        if not q.shape:
            q = q[None]
        basis_vectors: List[CGAVector]
        if only_2d:
            basis_vectors = [cls.from_scalar(1), cls.e_1() ^ cls.e_2()]
            if dilation:
                basis_vectors.extend([cls.e_0() ^ cls.e_inf(), cls.e_1() ^ cls.e_2() ^ cls.e_0() ^ cls.e_inf()])
            if translation:
                basis_vectors.extend([cls.e_1() ^ cls.e_inf(), cls.e_2() ^ cls.e_inf()])
            s = 2
        else:
            basis_vectors = [cls.from_scalar(1), cls.e_1() ^ cls.e_2(), cls.e_1() ^ cls.e_3(), cls.e_2() ^ cls.e_3()]
            if dilation:
                basis_vectors.extend(
                    [
                        cls.e_0() ^ cls.e_inf(),
                        cls.e_0() ^ cls.e_2() ^ cls.e_3() ^ cls.e_inf(),
                        cls.e_1() ^ cls.e_0() ^ cls.e_3() ^ cls.e_inf(),
                        cls.e_1() ^ cls.e_2() ^ cls.e_0() ^ cls.e_inf(),
                    ]
                )
            if translation:
                basis_vectors.extend(
                    [
                        cls.e_1() ^ cls.e_inf(),
                        cls.e_2() ^ cls.e_inf(),
                        cls.e_3() ^ cls.e_inf(),
                        cls.e_1() ^ cls.e_2() ^ cls.e_3() ^ cls.e_inf(),
                    ]
                )
            s = 4
        basis = cls.stack(basis_vectors)
        q_checked = q.checked()
        tmp = (
            (q_checked[..., None] & basis & p[..., None])
            + (q_checked.reversed()[..., None] & basis & p[..., None].reversed())
        ).sum(-2)
        lagrangian = basis[..., None].reversed().scalar_product(tmp[..., None, :]) / 2
        l_rr = lagrangian[..., :s, :s]
        l_rq = lagrangian[..., :s, s:]
        l_qr = lagrangian[..., s:, :s]
        l_qq = lagrangian[..., s:, s:]
        l_qq_inv = xnp.linalg.pinv(l_qq)
        l_dash = l_rr - l_rq @ l_qq_inv @ l_qr
        if rotation:
            r = xnp.linalg.eigh(l_dash)[1][..., -1]
        else:
            r = xnp.identity(s)[0] * np.ones([*p.shape[:-1], 1])
        t = xnp.einsum("...ij,...jk,...k->...i", l_qq_inv, l_qr, -r)
        m = xnp.concatenate([r, t], axis=-1)
        motor = (m * basis).sum(-1).view(cls)
        return motor / motor.reverse_norm() ** 0.5

    @classmethod
    def from_identity(cls):
        """Create Versor as identity

        Example:
        >>> from geometricalgebra import cga3d
        >>> allclose(cga3d.Vector.from_identity(), cga3d.Vector.from_scalar(1))
        True
        """
        return cls.from_scalar(1.0)

    @classmethod
    def from_tangent(cls, point, direction, normalize=False):
        result = point ^ (point | (direction ^ cls.e_inf()))
        if normalize:
            result = result.normalize_tangent_as_line()
        return result

    def normalize_tangent_as_line(self, eps=1e-8):
        """Normalize such that self ^ e_inf is normalized (e.g. if self is a point ^ direction)

        >>> from geometricalgebra import cga3d
        >>> p = cga3d.Vector.from_euclid([1,2,3])
        >>> d = cga3d.Vector.from_direction([4,5,6])
        >>> round(np.asarray(((p ^ d).normalize_tangent_as_line() ^ cga3d.e_inf).square_norm()).item(), 5)
        1.0
        """
        return self / ((self ^ self.e_inf()).square_norm() + eps) ** 0.5

    @classmethod
    def from_translator(cls, direction: Subtype) -> Subtype:
        """Create Versor from a direction

        Args:
            direction: the shift in the form length * euclidean_direction & e_inf

        Returns:
            The translator

        Example:
            >>> from geometricalgebra import cga3d
            >>> direction = 3 * cga3d.e_2 ^ cga3d.e_inf
            >>> allclose(cga3d.Vector.from_translator(direction), (- direction / 2).exp())
            True

        Reference:
            "Guide to Geometric Algebra in Practice", Chp 21.4
        """
        return 1 - direction / 2

    @classmethod
    def from_quaternion(cls, quaternion: ArrayLike):
        """Create a rotator from a quaternion (tensor)

        Args:
            quaternion: array of shape (..., 4) where the last dimension is
                the quaternion in the representation (w,x,y,z)

        Returns:
            The vector representing the quaternion
        """
        q = cls.xnp().asarray(quaternion)
        a = cls.from_scalar(q[..., 0])
        b = -cls.i3() | cls.from_direction(q[..., 1:])
        result = a + b
        return result.view(cls)

    def to_quaternion(self) -> Array:
        """Create a rotator from a quaternion (tensor)

        Returns:
            array of shape (..., 4) where the last dimension is
                the quaternion in the representation (w,x,y,z)

        Example:
        >>> from geometricalgebra import cga3d
        >>> np.asarray(cga3d.Vector.from_quaternion([1., 2., 3., 4.]).to_quaternion())
        array([1., 2., 3., 4.])
        """

        a = self.to_scalar()[..., None]
        b = (self.i3() | self).to_direction(normalize=False)
        return self.xnp().concatenate([a, b], axis=-1)

    @classmethod
    def from_pos_and_rot_vector(cls, pose: ArrayLike) -> CGAVector:
        """Construct pose from rotation vector (pose[..., 3:6]) and position vector (pose[..., 0:3])

        Args:
            pose: array of 6dim vectors of format (x, y, z, a, b, c). First the pose is rotated around (a, b, c)
                and then translated by (x,y,z)

        Returns:
            the resulting pose(s) as MultiVectorTensor

        Examples:Subtype
        >>> from geometricalgebra import cga3d
        >>> pose = cga3d.Vector.from_pos_and_rot_vector([1, 0, 0, 0, np.pi / 2, 0])
        >>> np.asarray(pose.apply(cga3d.e_0).to_euclid())
        array([1., 0., 0.])
        >>> pose = cga3d.Vector.from_pos_and_rot_vector([1, 0, 0, 0, 0, np.pi / 2])
        >>> np.allclose([0, 1, 0], pose.apply(cga3d.e_1).to_direction(False))
        True
        """
        posearray = cls.xnp().asarray(pose)
        translator = cls.from_translator(cls.from_direction(posearray[..., :3]) ^ cls.e_inf())
        rotator = cls.from_rotator(-cls.from_direction(posearray[..., 3:]).dual(cls.i3()))
        return translator & rotator

    def to_pos_and_rot_vector(self):
        """Return an array containing position and rotation vector of the pose

        Examples:
        >>> from geometricalgebra import cga3d
        >>> v = cga3d.Vector.from_pos_and_rot_vector([4, 5, 6, .1, .2, .3])
        >>> np.asarray(v.to_pos_and_rot_vector())
        array([4. , 5. , 6. , 0.1, 0.2, 0.3])
        >>> v = cga3d.Vector.from_pos_and_rot_vector( [0, 0, 0, 0, 0, np.pi / 2])
        >>> allclose(v.apply(cga3d.e_1), cga3d.e_2)
        True
        """
        translator, rotator = self.decompose_motor_in_translation_and_rotation_bivector()
        t = (self.e_0() | translator).to_direction(False)
        r = rotator.dual(self.i3()).to_direction(False)
        return np.concatenate([t, r], axis=-1)

    def to_scale_and_motor(self):
        scale = (-self.apply(self.e_inf()) | self.e_0()).to_scalar()
        v = self & self.from_scaling(1 / scale)
        return scale, v

    def to_scale_pos_and_rot_vector(self):
        """Decompose rotation translation and scaling transformation

        Example:
        >>> from geometricalgebra import cga3d
        >>> v = cga3d.Vector.from_pos_and_rot_vector([3, 4, 5, 0, 1, 2]) & cga3d.Vector.from_scaling(10)
        >>> scale, pos_rot = v.to_scale_pos_and_rot_vector()
        >>> np.asarray(scale)
        array(10.)
        >>> np.asarray(pos_rot)
        array([3., 4., 5., 0., 1., 2.])
        """
        scale, v = self.to_scale_and_motor()
        return scale, v.to_pos_and_rot_vector()

    def decompose_versor_in_scaling_and_motor(self) -> Tuple[CGAVector, CGAVector]:
        scale = (-self.apply(self.e_inf()) | self.e_0()).to_scalar()
        scaling = self.from_scaling(scale)
        motor = self & scaling.inverse()
        return scaling, motor

    @classmethod
    def from_pose(cls, position: np.ndarray, orientation: np.ndarray):
        translator = cls.from_translator(cls.from_direction(position) ^ cls.e_inf())
        rotator = cls.from_quaternion(orientation)
        return translator & rotator

    @classmethod
    def from_rotation_matrix(cls, matrix):
        quaternion = Rotation.from_matrix(matrix).as_quat()[..., [3, 0, 1, 2]]
        return cls.from_quaternion(quaternion)

    def to_pose(self) -> Tuple[Array, Array]:
        """Return a position and quaterion (as w,x,y,z format)

        Example:
        >>> from geometricalgebra import cga3d
        >>> pose = cga3d.Vector.from_pose([1,2,3], [3,4,5,6])
        >>> np.asarray(pose.to_pose()[0])
        array([1., 2., 3.])
        >>> np.asarray(pose.to_pose()[1])
        array([3., 4., 5., 6.])
        """
        position, orientation = self.decompose_into_pose()
        return position.to_euclid(), orientation.to_quaternion()

    def decompose_into_pose(self) -> Tuple[CGAVector, CGAVector]:
        translator, rotator = self.decompose_motor_in_translator_and_rotor()
        orientation = rotator
        position = translator.apply(self.e_0())
        return position, orientation

    @classmethod
    def from_array(cls, array: np.ndarray) -> CGAVector:
        return cls(array, {0, 2, 4})

    def to_array(self) -> Array:
        return self._expanded_grades(frozenset([0, 2, 4]))._values  # pylint: disable=protected-access

    @classmethod
    def from_rotator(cls: Type[Subtype], plane: Subtype) -> Subtype:
        """Vector an euclidean vector x * e_1 + y * e_2 + y * e_3

        Examples:
            >>> from geometricalgebra import cga3d
            >>> rotator = cga3d.Vector.from_rotator(3 * cga3d.e_1 ^ cga3d.e_2)
            >>> allclose(rotator, (- 3 * cga3d.e_1 ^ cga3d.e_2 / 2).exp())
            True
            >>> round(np.asarray(rotator.reverse_norm()).item(), 5)
            1.0

        Reference:
            "Guide to Geometric Algebra in Practice", Chp 21.4
        """
        # Use sinc instead of normalizing and multiplying with sin since latter option has a singularity at plane = 0
        phi = cls.xnp().sqrt(plane.reverse_norm())
        return (cls.from_scalar(cls.xnp().cos(phi / 2)) - plane * cls.xnp().sinc(phi / (2 * np.pi)) / 2).view(cls)

    @classmethod
    def from_scaling(cls, factor: ArrayLike) -> CGAVector:
        """Vector an euclidean vector x * e_1 + y * e_2 + y * e_3

        Reference:
            "Guide to Geometric Algebra in Practice", Chp 21.4
        """
        log_factor = np.log(factor)
        return cls.from_scalar(np.cosh(log_factor / 2)) + np.sinh(log_factor / 2) * (cls.e_0() ^ cls.e_inf())

    @classmethod
    def from_screw(cls, axis: Subtype, pitch) -> Subtype:
        """Create a versor from a screw

        This screw motion is given by a rotation around the axis (magnitute is the angle to rotate) and
        a translation along the axis by the pitch * axis

        Example:
        >>> from geometricalgebra import cga3d
        >>> versor = cga3d.Vector.from_screw(  (0.3 * cga3d.e_1).up() ^ cga3d.e_2 ^ cga3d.e_inf, 0)
        >>> with np.printoptions(3, suppress=True):from itertools import permutations

        ...     print(versor.apply(MultiVectorTensor.from_euclid([0.3, 10, 0])).to_euclid())
        [ 0.3 10.  -0. ]
        >>> versor = cga3d.Vector.from_screw(np.pi / 2 * cga3d.e_0 ^ cga3d.e_1 ^ cga3d.e_inf, 0)
        >>> with np.printoptions(3, suppress=True):
        ...     print(versor.apply(cga3d.e_2.up()).to_euclid())
        [ 0. -0.  1.]
        >>> angle = np.pi / 2
        >>> versor = cga3d.Vector.from_screw(angle * cga3d.e_0 ^ cga3d.e_1 ^ cga3d.e_inf, 7 / angle)
        >>> with np.printoptions(3, suppress=True):
        ...     print(versor.apply(cga3d.e_2.up()).to_euclid())
        [ 7. -0.  1.]
        """
        direction = axis | cls.minkovski_plane()
        versor = (1 / 2 * (axis.dual() + (cls.e_inf() ^ direction * pitch))).exp()
        return versor.view(cls)

    def motor_interpolation(self, param: ArrayLike, method="tr") -> CGAVector:
        """Pose interpolation

        Args:
            param: values (between zero and one) to indicate the interpolation
                (between one and self)
            method: the interpolation method

        Returns:
            The interpolated motors

        Example:
        >>> from geometricalgebra import cga3d
        >>> v = cga3d.Vector.from_pos_and_rot_vector([4,5,6,.1,.2,.3])
        >>> v.motor_interpolation(0.5).to_pos_and_rot_vector()
        array([2.  , 2.5 , 3.  , 0.05, 0.1 , 0.15])
        """
        assert method == "tr"
        v = self.decompose_motor_in_translation_and_rotation_bivector()
        t, r = param * v
        return self.from_translator(t) & self.from_rotator(r)

    def motor_to_screw(self):
        """
        Returns:
            screw axis
            theta: angle to rotate
            distance:

        TODO:
            handle case of zero rotation
        Example:
        """
        t_as_bivector, r = self.decompose_motor_in_translation_and_rotation_bivector()
        _, R = self.decompose_motor_in_translator_and_rotor()
        t = self.e_0() | t_as_bivector
        # I = r.normed(reverse_norm=True)
        w = (t ^ r) | r.inverse()
        # w = (t ^ r) | r / r.square_norm()
        tmp = 1 - (R & R)
        v = (t & r)(1) & (r / r.square_norm()) & (tmp / tmp.reverse_norm())
        a = (v.up() ^ r.dual(self.i3()) ^ self.e_inf())(3)
        pitch = (w.square_norm() / a.square_norm()) ** 0.5
        return a, pitch

    def decompose_motor_in_translator_and_rotor(self):
        """Decompose a motor M = TR into translator T and rotor R

        Returns: Tuple (translator, rotor)

        Note: assumes that versor is motor

        Reference:
            Tingelstad et al., Motor Estimation using Heterogeneous Sets of Objects in Conformal Geometric,
                Algebra. Adv. Appl. Clifford Algebras 27, 2035–2049 (2017).

        Returns:
            tuple: translator, rotor
        """
        rotor = -self.e_0() | (self & self.e_inf())
        translation_vector = -2 * (self.e_0() | self) & rotor.inverse()
        translator = self.from_translator(translation_vector ^ self.e_inf())
        return translator, rotor

    @classmethod
    def from_rotation_between_two_directions(cls, p: Subtype, q: Subtype, normalized=False) -> Subtype:
        if not normalized:
            p = p.normed()
            q = q.normed()
        versor = 1 + (q & p)
        return versor.normed(reverse_norm=True)

    def decompose_motor_in_translation_and_rotation_bivector(self, from_unnormalized=False) -> CGAVector:
        """Extract translation and rotation (bi)vector from motor


        See also:
            decompose_motor_in_translator_and_rotor

        Example:
        >>> from geometricalgebra import cga3d
        >>> r = 2 * cga3d.e_2 ^ cga3d.e_3
        >>> t = (0.2 * cga3d.e_3 + 1.3 * cga3d.e_2) ^ cga3d.e_inf
        >>> v = cga3d.Vector.from_translator(t) & cga3d.Vector.from_rotator(r)
        >>> allclose(v.decompose_motor_in_translation_and_rotation_bivector()[0], t)
        True
        >>> allclose(v.decompose_motor_in_translation_and_rotation_bivector()[1], r)
        True
        >>> allclose((2 * v).decompose_motor_in_translation_and_rotation_bivector(from_unnormalized=False)[1], r)
        False
        >>> allclose((2 * v).decompose_motor_in_translation_and_rotation_bivector(from_unnormalized=True)[1], r)
        True
        >>> cga3d.Vector.from_identity().decompose_motor_in_translation_and_rotation_bivector()
        Vector([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        """

        if from_unnormalized:
            return self.normed(reverse_norm=True).decompose_motor_in_translation_and_rotation_bivector(False)

        s = self * self.xnp().where(
            self(0).to_scalar() >= 0, 1, -1
        )  # this ensures that the theta (see below) is 0 <= theta <= pi
        rotor = -self.e_0() | (s & self.e_inf())
        translation_vector = -2 * (self.e_0() | s) & rotor.inverse()
        theta = np.arccos(self.xnp().clip(rotor(0).to_scalar(), -1, 1))
        rotation_vector = -2 * rotor(2) / np.sinc(theta / np.pi)
        return self.stack([translation_vector ^ self.e_inf(), rotation_vector])

    def to_rotation_matrix(self, normalize=True):
        """Convert to a rotation matrix

        Args:
            normalize: This works only for normalized motors, if this vector is already normalized, setting this
                to false is more efficient.

        Returns:
            The rotation matrix

        Example:
        >>> from geometricalgebra import cga3d
        >>> v = cga3d.Vector.from_pos_and_rot_vector([0, 0, 0, np.pi/2, 0, 0])
        >>> with np.printoptions(suppress=True): np.asarray(v.to_rotation_matrix())
        array([[ 1.,  0.,  0.],
               [ 0.,  0., -1.],
               [ 0.,  1.,  0.]])
        """
        if normalize:
            normed_self = self.normed(reverse_norm=True)
        else:
            normed_self = self
        return (
            normed_self.reversed().right_product_matrix(ProductType.GEOMETRIC)
            @ normed_self.left_product_matrix(ProductType.GEOMETRIC)
        )[1:4, 1:4]

    def to_transformation_matrix(self):
        """Return a 4x4 transformation matrix

        Returns:
            The transformation matrix

        Example:
        >>> from geometricalgebra import cga3d
        >>> v = cga3d.Vector.from_pos_and_rot_vector([1, 2, 3, np.pi/2, 0, 0])
        >>> with np.printoptions(suppress=True): np.asarray(v.to_transformation_matrix() + 1e-16)
        array([[ 1.,  0.,  0.,  1.],
               [ 0.,  0., -1.,  2.],
               [ 0.,  1.,  0.,  3.]])
        """
        a = self.reversed().right_product_matrix(ProductType.GEOMETRIC) @ self.left_product_matrix(
            ProductType.GEOMETRIC
        )
        b = self.stack([self.e_1(), self.e_2(), self.e_3(), self.e_0()]).values
        return b[:3] @ a @ b.T


# class TangentLine(MultiVectorTensor):
#     """A tangent"""


def orthogal_projection(obj: Subtype, onto: Subtype) -> Subtype:
    """Orthogonal projection of `obj' onto `onto'

    Args:
        obj: the points which shall be projected
        onto: the object onto with the points shall be projected

    Returns:
        The projection

    Reference:
        page 444 in Guide to geometruc Algebra in Practice
    """
    return (obj | onto) | onto.inv()


def project(points: Subtype, onto: Subtype) -> Subtype:
    # what if points ^ onto.dual() == 0
    return onto.meet(points.e_inf() ^ points ^ onto.dual()).point_pair_to_end_points()[0]


def project_to_flat(points: Subtype, onto: Subtype) -> Subtype:
    """Project points onto an flat object

    Args:
        points: the points which shall be projected
        onto: the object onto with the points shall be projected

    Returns:
        The projection

    Example:
    >>> from geometricalgebra import cga3d
    >>> tmp = cga3d.Vector.from_euclid([[-1, 0, 0], [-1, 0, 1], [-1, 1, 0]])
    >>> points = cga3d.Vector.from_euclid([2, 3, 4])
    >>> plane = tmp[0] ^ tmp[1] ^ tmp[2] ^ cga3d.e_inf
    >>> np.asarray(project_to_flat(points, onto=plane).to_euclid())
    array([-1.,  3.,  4.])
    >>> line = tmp[0] ^ tmp[1] ^ cga3d.e_inf
    >>> np.asarray(project_to_flat(points, onto=line).to_euclid())
    array([-1., -0.,  4.])
    >>> line = tmp[1] ^ cga3d.e_inf
    >>> np.asarray(project_to_flat(points, onto=line).to_euclid())
    array([-1., -0.,  1.])
    >>> np.asarray(project_to_flat(points, onto=cga3d.i5).to_euclid())
    array([2., 3., 4.])
    """
    return (onto | points).center().normed_special()


def project_point_to_line(point: Subtype, line: Subtype) -> Subtype:
    proj_point = point | line
    return (proj_point & point.e_inf() & proj_point)(1)


def to_sphere(obj: Subtype) -> Subtype:
    """Makes the smallest sphere containing obj (point pair, circle, or sphere)

    Args:
        obj: the entity which shall be converted to a sphere

    Returns:
        The sphere

    Example:
    >>> from geometricalgebra import cga3d
    >>> a, b, c, d = cga3d.Vector.from_euclid([(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)])
    >>> np.asarray(to_sphere(a ^ b).sphere_to_center_squared_radius()[0].to_euclid())
    array([1.5, 1. , 1. ])
    >>> np.asarray(to_sphere(a ^ b ^ c).sphere_to_center_squared_radius()[0].to_euclid())
    array([1.5, 1.5, 1. ])
    >>> np.asarray(to_sphere(a ^ b ^ c ^ d).sphere_to_center_squared_radius()[0].to_euclid())
    array([1.5, 1.5, 1.5])
    """
    return obj ^ (obj ^ obj.e_inf()).dual()


def project_point_to_plane(point: Subtype, plane: Subtype) -> Subtype:
    proj_point = (point | plane) & plane
    return -(proj_point & point.e_inf() & proj_point)(1) / 2


def project_to_round(points: Subtype, onto: Subtype, closest: bool = True) -> Subtype:
    """Projects entities onto a round object

    Args:
        points: the points which shall be projected
        onto: the object onto with the points shall be projected
        closest: If true the closest point (e.g. on a cirle) is retuned otherwise the furthermost point.

    Returns:
        The projections

    Example:
    >>> from geometricalgebra import cga3d
    >>> a, b, c, d, e  = cga3d.Vector.from_euclid([(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (5, 5, 5)])
    >>> np.asarray(project_to_round(e, cga3d.Vector.stack([a ^ b ^ c ^ d, -a ^ b ^ c ^ d])).to_euclid())
    array([[2., 2., 2.],
           [2., 2., 2.]])
    >>> np.asarray(project_to_round(e, cga3d.Vector.stack([a ^ b ^ c, -a ^ b ^ c])).to_euclid())
    array([[2., 2., 1.],
           [2., 2., 1.]])
    >>> np.asarray(project_to_round(e, cga3d.Vector.stack([a ^ b, b ^ a])).to_euclid())
    array([[2., 1., 1.],
           [2., 1., 1.]])
    >>> np.asarray(project_to_round(e, a ^ b, closest=False).to_euclid())
    array([1., 1., 1.])
    """
    index = 1 if closest else 0
    points_on_flat = project_to_flat(points, onto=onto ^ points.e_inf()).normed_special()
    b = onto.copy()
    if 3 in onto.grades:
        tmp = np.ones(onto._values.shape[-1])  # pylint: disable=protected-access
        tmp[onto._mask(frozenset([3]))] *= -1  # pylint: disable=protected-access
        b._values = points.xnp().asarray(tmp * onto._values)  # pylint: disable=protected-access
    oriented_center = -onto & points.e_inf() & b
    point_pair = -(points_on_flat ^ oriented_center ^ points.e_inf()).meet(to_sphere(onto))
    return point_pair.point_pair_to_end_points()[index]


def commutator(a, b):
    return a.commutator(b)


def anticommutator(a, b):
    return a.commutator(b, anti=True)


class VersorTensor(CGAVector):  # pylint: disable=abstract-method
    """Versors define conformal transformation"""


def transformation_to_zero_mean_unit_variance(points: Subtype, eps: float = 10e-6) -> Subtype:
    """Return a versor V such that V.apply(points).to).euclid() has a zero mean and unit variance"""
    imaginary_sphere = points.mean()  # The center is the sphere is mean of points
    # and squared radius is negative variance [Chap. 7.3.3]
    displacement_from_origin = -imaginary_sphere.center().down()
    translation = points.from_translator(displacement_from_origin ^ points.e_inf())
    factor = 1 / (np.maximum(-imaginary_sphere.dual().sphere_to_center_squared_radius()[1].to_scalar(), 0) ** 0.5 + eps)
    scaling = points.from_scaling(factor)
    return scaling & translation


def get_motor_from_object_pair(p: Subtype, q: Subtype) -> VersorTensor:
    """Given pairs of geometric entities, the optimal rotor between these objects is returned

    Args:
        q: Conformal objects with shape (..., n)
        p: Conformal objects (of same grade as p) with shape (..., n)

    Returns:
        the optimal versors of shape (...,) mapping all q on p (along last axis)

    References: J. Lasenby et al., "Calculating the Rotor Between Conformal Objects", Adv. Appl. Clifford Algebras (2019)
    """
    p = p.normed()
    q = q.normed()
    K = 2 + anticommutator(p, q)

    lambda_ = -K(4).square_norm()
    mu = K(0).square_norm() + lambda_
    sqrt_mu = np.sqrt(np.clip(mu, 1e-8, None))
    beta = np.sqrt(0.5 / (sqrt_mu + K(0).to_scalar()))
    Y = -1 / (2 * beta + 1e-8) + beta * K(4)
    # note if q and p are line, this could be simplified to
    # Y = 1 - K(4) / (2 * K(0).to_scalar())
    motor = 1 / sqrt_mu * Y & (1 + (q & p))
    return -motor
