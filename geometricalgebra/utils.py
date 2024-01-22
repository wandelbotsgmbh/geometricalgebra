"""Collection of utilities"""

from geometricalgebra.vector import FRAMEWORK


def solve_gauss_normal_equations(a, b):
    xnp = FRAMEWORK.numpy
    at_a_inv = xnp.linalg.pinv(xnp.einsum("...jk,...ji->...ki", a, a))
    return xnp.einsum("...ij,...kj,...k->...i", at_a_inv, a, b)
