"""Test the coherent point drift algorithm"""

from pathlib import Path

import numpy as np

from geometricalgebra import cga2d, cga3d
from geometricalgebra.coherent_point_drift import pose_and_correspondence_registration


def get_test_data():
    path = Path(__file__).parent.parent.parent / "test_data" / "point_registration"
    p = np.loadtxt(path / "source.txt")
    q = np.loadtxt(path / "target.txt")
    p = p[400:-400:16] + [40, 0]
    q = q[::16]
    p = np.array([i for i in p if not np.isnan(i).any()])
    q = np.array([i for i in q if not np.isnan(i).any()])
    p = cga2d.Vector.from_euclid(p)
    q = cga2d.Vector.from_euclid(q)
    return p, q


def test_pose_and_correspondence_registration_2d():
    p, q = get_test_data()
    a, _, variance = pose_and_correspondence_registration(p, q, only_2d=True)
    assert variance < 1
    # plt.scatter(*p.to_euclid().T)
    # plt.scatter(*q.to_euclid().T)
    # plt.scatter(*a.apply(p).to_euclid().T)


def test_pose_and_correspondence_registration_3d():
    p = cga3d.Vector.from_euclid(10 * np.random.random([5, 3]))
    v = cga3d.Vector.from_pos_and_rot_vector([0.04, 0.05, 0.16, 0.1, 0.2, 0.1])
    ids = np.arange(len(p))
    # np.random.shuffle(ids)
    q = v.apply(p)[ids]
    estimation, _, _ = pose_and_correspondence_registration(p, q)
    print(v.to_pos_and_rot_vector())
    print(estimation.to_pos_and_rot_vector())
    assert np.allclose(v.to_pos_and_rot_vector(), estimation.to_pos_and_rot_vector())
