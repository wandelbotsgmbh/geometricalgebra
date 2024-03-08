# geometricalgebra

![badge](https://github.com/wandelbotsgmbh/geometricalgebra/actions/workflows/python-app.yml/badge.svg)

Library implementing conformal geometric algebra.

The key features are:
- Fast numerical implementation of multivector and its exterior algebra
- The library supports various backends (numpy, tensorflow, jax)
- Full support of autograd works when using jax and tensorflow
- All operation work for single multivector or tensors of multivector. Broadcasting is also supported.


# Installation

    pip install geometricalgebra

# Example

    from geometricalgebra import cga3d
    a = cga3d.e_0
    b = cga3d.e_1.up()
    c = cga3d.e_2.up()
    circle = a ^ b ^ c
    # The radius of a circle going through [0, 0, 0], [1, 0, 0], and [0, 1, 0]
    radius = circle.circle_to_center_normal_radius()[2].to_scalar()
