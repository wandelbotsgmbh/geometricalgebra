import numpy as np
import pytest

from geometricalgebra import cga3d
from geometricalgebra.vector import ga_numpy

jax = pytest.importorskip("jax")


@pytest.mark.skipif(ga_numpy != "jax", reason="This gradient computation works only with the jax backend")
def test_jax_grad():
    def func(a):
        b = cga3d.Vector.from_euclid(a)
        return b.scalar_product(cga3d.e_0)

    func_and_grad = jax.value_and_grad(func)
    value, grad = func_and_grad(jax.numpy.array([1.0, 2.0, 3.0]))
    assert np.allclose(grad, [-1, -2, -3])
