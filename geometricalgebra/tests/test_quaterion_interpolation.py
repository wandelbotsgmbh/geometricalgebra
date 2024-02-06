"""Example for quaternion interpolation"""

import numpy as np

from geometricalgebra import cga3d, cga4d


def test_quaternion_interpolation():
    x1 = cga4d.Vector([*cga3d.Vector.from_pos_and_rot_vector([0, 0, 0, 0.2, 0, 0]).to_quaternion(), 0, 0], grade=1)
    q3 = [-0.3, 0.1, 0.4]
    x3 = cga4d.Vector([*(cga3d.Vector.from_pos_and_rot_vector([0, 0, 0, *q3]).to_quaternion()), 0, 0], grade=1)
    q2 = [-0.2, 0.1, 0.3]
    x2 = cga4d.Vector([*(cga3d.Vector.from_pos_and_rot_vector([0, 0, 0, *q2]).to_quaternion()), 0, 0], grade=1)

    rot_plane = (x3 - x1) ^ (x3 - x2)
    phi = np.arctan2(rot_plane.reverse_norm() ** 0.5, ((x2 - x1) | (x3 - x2)).to_scalar()) % (2 * np.pi)

    tmp = (-phi * rot_plane.normed(1e-16)).exp(25)
    x3_recovered = (
        cga3d.Vector.from_quaternion(tmp.apply(x1).to_direction(False))
        .normed(reverse_norm=True)
        .to_pos_and_rot_vector()[..., 3:]
    )
    assert np.allclose(q3, x3_recovered)

    phi_intermediate = np.arctan2(
        ((x3 - x1) ^ (x2 - x3)).reverse_norm() ** 0.5, ((x3 - x1) | (x2 - x3)).to_scalar()
    ) % (2 * np.pi)

    tmp = (phi_intermediate * rot_plane.normed(1e-16)).exp(25)
    x2_recovered = (
        cga3d.Vector.from_quaternion(tmp.apply(x1).to_direction(False))
        .normed(reverse_norm=True)
        .to_pos_and_rot_vector()[..., 3:]
    )
    assert np.allclose(q2, x2_recovered)
