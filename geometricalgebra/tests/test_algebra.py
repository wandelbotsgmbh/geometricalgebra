"""Test the geometricalgebra.algebra module"""

from geometricalgebra import cga2d, cga3d, cga4d


def test_cga2d():
    assert cga2d.ALGEBRA.dim == 16
    assert cga2d.ALGEBRA.dims_of_grade == (1, 4, 6, 4, 1)


def test_cga3d():
    assert cga3d.ALGEBRA.dim == 32
    assert cga3d.ALGEBRA.dims_of_grade == (1, 5, 10, 10, 5, 1)


def test_cga4d():
    assert cga4d.ALGEBRA.dim == 64
    assert cga4d.ALGEBRA.dims_of_grade == (1, 6, 15, 20, 15, 6, 1)
