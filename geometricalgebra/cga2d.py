"""Handling 2d CGA Tensors"""

from geometricalgebra.algebra import GeometricAlgebra
from geometricalgebra.cga import CGAVector

ALGEBRA = GeometricAlgebra((1, 1, 1, -1))


class Vector(CGAVector):
    """A 2d CGA Tensor"""

    @property
    def algebra(self):
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
        raise NotImplementedError()

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


e_1 = Vector([1, 0, 0, 0], grade=1)  # type: ignore
e_2 = Vector([0, 1, 0, 0], grade=1)  # type: ignore
e_plus = Vector([0, 0, 1, 0], grade=1)  # type: ignore
e_minus = Vector([0, 0, 0, 1], grade=1)  # type: ignore
e_inf = e_plus + e_minus
e_0 = (e_minus - e_plus) / 2
i4 = e_1 ^ e_2 ^ e_inf ^ e_0
i2 = e_1 ^ e_2
