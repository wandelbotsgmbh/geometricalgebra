"""Collection of utilities"""

from geometricalgebra.vector import FRAMEWORK
from typing import Tuple

def solve_gauss_normal_equations(a, b):
    xnp = FRAMEWORK.numpy
    at_a_inv = xnp.linalg.pinv(xnp.einsum("...jk,...ji->...ki", a, a))
    return xnp.einsum("...ij,...kj,...k->...i", at_a_inv, a, b)


def reshape_last_dimensions(array, num_of_dim: int, new_shape: Tuple[int, ...]):
    """Reshape a tensor with shape (..., 4, 3) to shape (..., 12)"""
    return array.reshape([*array.shape[:-num_of_dim], *new_shape])