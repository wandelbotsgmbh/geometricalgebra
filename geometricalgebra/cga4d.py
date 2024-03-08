"""Handling 4d CGA Tensors"""

from typing import Iterable, Union

from geometricalgebra.algebra import GeometricAlgebra
from geometricalgebra.cga import CGAVector

ALGEBRA = GeometricAlgebra((1, 1, 1, 1, 1, -1))


class Vector(CGAVector):
    """A 4d CGA Tensor"""

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
    def e_4(cls):
        return e_4

    @classmethod
    def e_inf(cls):
        return e_inf

    @classmethod
    def i3(cls):
        raise NotImplementedError()

    @classmethod
    def i5(cls):
        raise NotImplementedError()

    @classmethod
    def minkovski_plane(cls):
        raise NotImplementedError()

    def __init__(self, values, grade: Union[int, Iterable[int]], algebra=None):
        algebra = algebra or ALGEBRA
        if algebra != ALGEBRA:
            raise TypeError(f"Unexpected algebra: {algebra}")
        super().__init__(values, grade)


e_1, e_2, e_3, e_4, e_plus, e_minus = Vector.basis()
e_inf = e_plus + e_minus
e_0 = (e_minus - e_plus) / 2
i6 = e_1 ^ e_2 ^ e_3 ^ e_4 ^ e_inf ^ e_0
i4 = e_1 ^ e_2 ^ e_3 ^ e_4
