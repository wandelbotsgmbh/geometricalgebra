"""Test MultiVectorTensors"""

import math
import random
from operator import add, and_, mul, sub, xor

import pytest

from geometricalgebra import cga2d, cga3d
from geometricalgebra.algebra import ProductType
from geometricalgebra.cga import project_point_to_plane
from geometricalgebra.cga3d import e_0, e_1, e_2, e_inf, fit_points_to_plane, np
from geometricalgebra.vector import allclose


def test_products_of_tensors():
    a = cga3d.Vector(np.random.normal(size=[2, 1, 32]), grade=set(range(6)))
    b = cga3d.Vector(np.random.normal(size=[6, 32]), grade=set(range(6)))
    assert allclose((a & b)[1, 3], a[1, 0] & b[3])


def test_from_and_to_euclid():
    a = np.random.normal(size=[10, 3])
    b = cga3d.Vector.from_euclid(a).to_euclid()
    assert np.allclose(a, b)


def test_norm():
    a = cga3d.Vector.from_direction([1, 2, 3]).normed()
    assert np.allclose(a.square_norm(), 1)


def test_ndim():
    assert cga3d.Vector.from_euclid([2, 3, 4]).ndim == 0


def test_from_and_to_scalar():
    a = np.array([0, -12, 242.432])
    b = cga3d.Vector.from_scalar(a).to_scalar()
    assert np.allclose(a, b)


def test_point_pair_to_end_points():
    a = np.array((0, 1, 2))
    b = np.array((3, -1, 0))
    point_pair = cga3d.Vector.from_euclid(a) ^ cga3d.Vector.from_euclid(b)
    result = point_pair.point_pair_to_end_points()
    assert allclose(result, result.normed_special())
    assert np.allclose(a, result[0].to_euclid())
    assert np.allclose(b, result[1].to_euclid())


def test_grades():
    a = np.ones(32)
    assert cga3d.Vector(a[6:16], grade={2}).grades == {2}
    assert cga3d.Vector(a[6:16], grade={2}).grade == 2
    assert cga3d.Vector(a[1:16], grade={1, 2}).grades == {1, 2}


def test_circle_to_center_normal_radius():
    circle = (
        cga3d.Vector.from_euclid((-1, 0, 0)) ^ cga3d.Vector.from_euclid((0, 1, 0)) ^ cga3d.Vector.from_euclid((1, 0, 0))
    )
    center, normal, radius = circle.circle_to_center_normal_radius()
    assert np.allclose(radius.to_scalar(), 1)
    assert np.allclose(center.to_euclid(), 0)
    assert np.allclose(normal.to_euclid(), (0, 0, -1))


@pytest.mark.parametrize("shape", [(), (3,), (3, 5)])
def test_multi_vector_tensor_square_norm(shape):
    a = cga3d.Vector(np.random.normal(size=[*shape, 32]), grade=set(range(6)))
    assert np.allclose((a & a).to_scalar(), a.square_norm())


def test_grades_of_product():
    assert cga3d.ALGEBRA.grades_of_product(frozenset({1}), frozenset({3}), ProductType.GEOMETRIC) == {2, 4}
    assert cga3d.ALGEBRA.grades_of_product(frozenset({1}), frozenset({3}), ProductType.INNER) == {2}
    assert cga3d.ALGEBRA.grades_of_product(frozenset({1}), frozenset({3}), ProductType.OUTER) == {4}
    assert cga3d.ALGEBRA.grades_of_product(frozenset({1}), frozenset({3, 4}), ProductType.GEOMETRIC) == {2, 3, 4, 5}
    assert cga3d.ALGEBRA.grades_of_product(frozenset({1}), frozenset({3, 4}), ProductType.INNER) == {2, 3}
    assert cga3d.ALGEBRA.grades_of_product(frozenset({1}), frozenset({3, 4}), ProductType.OUTER) == {4, 5}


def test_add():
    a = cga3d.Vector.from_scalar(1)
    b = cga3d.Vector.from_scalar(5)
    assert (a + b).to_scalar() == 6
    assert (a - b).to_scalar() == -4
    a += b
    assert a.to_scalar() == 6
    a -= b
    assert a.to_scalar() == 1


@pytest.mark.parametrize("radius", [1, 10, 100, 1000])
def test_on_same_circle(radius):
    """test on_same_circle member function"""

    def get_points_on_circle(sample_size, r, offset=0) -> cga3d.Vector:
        points = []
        for theta in np.linspace(0, 1.5 * math.pi, sample_size):
            points.append(
                [
                    r * math.cos(theta) + offset * random.random(),
                    r * math.sin(theta) + offset * random.random(),
                    offset * random.random(),
                ]
            )
        return cga3d.Vector.from_euclid(np.array(points))

    points_on_circle = get_points_on_circle(10, radius, 0)
    points_not_on_circle = get_points_on_circle(10, radius, 1)
    assert points_on_circle.on_same_circle()
    assert not points_not_on_circle.on_same_circle()


def test_exp_of_rotation():
    """For a rotation parallel to an plane I (with I & I == -1), e.g., I = e_1 ^ e_2, the exponential
    function simplifies. See Chap 21.6.
    """

    plane = e_1 ^ e_2
    phi = 1.5
    a = (-phi * plane / 2).exp()
    b = cga3d.Vector.from_scalar(np.cos(phi / 2)) - (np.sin(phi / 2) * plane)
    assert allclose(a, b)


def test_exp_of_translation():
    """For a translation parallel to a direction the exponential
    function simplifies. See Chap 21.6.
    """
    t = cga3d.Vector.from_direction([1, 2, 3])
    a = (-t ^ e_inf / 2).exp()
    b = cga3d.Vector.from_scalar(1) - (t ^ e_inf / 2)
    assert np.allclose((a & e_0 & a.adjoint())(1).to_euclid(), [1, 2, 3])
    assert allclose(a, b)


@pytest.mark.parametrize("a_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("b_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("operator", [add, mul, xor, and_, sub])
def test_dtype_binary_op(a_dtype, b_dtype, operator):
    a = cga3d.Vector.from_euclid([1, 2.3, 3]).astype(a_dtype)
    b = cga3d.Vector.from_euclid([5.7, 0, 0]).astype(b_dtype)
    assert operator(a, b).dtype == np.result_type(a, b)


# @pytest.mark.parametrize("dtype", [np.float32, np.float64])
# @pytest.mark.parametrize("operator", [neg, MultiVectorTensor.adjoint, MultiVectorTensor.exp])
# def test_dtype(dtype, operator):
#     a = MultiVectorTensor.from_euclid([1, 2.3, 3]).astype(dtype)
#     assert operator(a).dtype == dtype


def test_scalar_addition():
    """Test scalar addition."""
    x = cga3d.Vector.from_euclid([1, 2, 4])
    assert allclose(x + 1, 1 + x)
    assert allclose(x + np.array(1), np.array(1) + x)
    assert allclose(x + math.cos(0), math.cos(0) + x)
    assert allclose(x + np.cos(0), np.cos(0) + x)
    assert allclose(x + np.array(np.cos(0)), np.array(np.cos(0)) + x)


def test_commutator():
    a, b = cga3d.Vector(np.random.random([2, 32]), range(6))
    allclose((a & b) - (b & a), a.commutator(b))
    allclose((a & b) + (b & a), a.commutator(b, anti=True))


def test_cga2d():
    a = cga2d.Vector.from_direction([1, 2])
    assert a._algebra.dim == 16  # pylint: disable=protected-access
    b = cga2d.Vector.from_direction([0, 1])
    c = a ^ b
    assert c.grades == {2}


def test_multivectortensor_size():
    t = cga3d.Vector.from_euclid(np.zeros([4, 6, 3]))
    assert t.size == 4 * 6


def test_plane_fitting():
    np.random.seed(0)
    points = cga3d.Vector.from_euclid(np.random.random((10, 3)))
    plane = (points[0] ^ points[1] ^ points[2] ^ e_inf).normed()
    points_in_plane = project_point_to_plane(points, plane)
    plane_fit = fit_points_to_plane(points_in_plane)
    allclose(plane, plane_fit)


def test_plane_constructor():
    np.random.seed(0)
    points = cga3d.Vector.from_euclid(np.random.random((3, 3)))
    plane = (points[0] ^ points[1] ^ points[2] ^ e_inf).normed()
    normal = plane.normal().to_direction(True)
    distance = (plane.dual().normed() | cga3d.Vector.from_identity()).to_scalar()
    allclose(plane, cga3d.Vector.from_plane(normal, distance))
