"""Test versors, i.e., various kinds of transformations
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from geometricalgebra import cga3d
from geometricalgebra.cga import get_motor_from_object_pair, transformation_to_zero_mean_unit_variance
from geometricalgebra.cga3d import FRAME, e_1, e_2, e_3, e_inf, i3, pose_distance
from geometricalgebra.vector import allclose


def _get_optimal_rotation(p: cga3d.Vector, q: cga3d.Vector) -> cga3d.Vector:
    """Get optimal rotation to translate a list of geometric entities p to the set q

    THIS IS DEPRECATED: use VerstorTensor.get_optimal_motor(..., translation=False) instead

    Reference:
        "Guide to Geometric Algebra in Practice", Chap. 2.4.2
    """

    def get_basis_rotator():
        return cga3d.Vector.stack([cga3d.Vector.from_scalar(1), e_1 ^ e_2, e_1 ^ e_3, e_2 ^ e_3])

    basis = get_basis_rotator()
    lagrangian = (basis[:, None].adjoint() & sum((q[:, None]).checked() & basis & p[:, None])).to_scalar()
    eigval, eigvec = np.linalg.eig(lagrangian)
    vector = eigvec[:, np.argmax(-eigval.real)]
    result = sum(vector * basis, start=0)
    return result


def _get_optimal_translation(p: cga3d.Vector, q: cga3d.Vector) -> cga3d.Vector:
    """Get optimal translation to translate a list of geometric entities p to the set q

    THIS IS DEPRECATED: use VerstorTensor.get_optimal_motor(..., rotation=False) instead

    Reference:
        "Guide to Geometric Algebra in Practice", Chap. 2.4.2
    """

    def get_basis_translator():
        return cga3d.Vector.stack([cga3d.Vector.from_scalar(1), e_1 ^ e_inf, e_2 ^ e_inf, e_3 ^ e_inf])

    basis = get_basis_translator()
    lagrangian = (basis[:, None] & sum((p[:, None]).checked() & basis.adjoint() & q[:, None])).to_scalar()
    vector = np.concatenate([[1], (-np.linalg.pinv(lagrangian[1:, 1:]) @ lagrangian[1:, 0])])
    result = sum(vector * basis)
    return result


def test_translation():
    points = np.eye(4, 3)
    translation = np.array([1, 2, 3])
    reference = points + translation
    versor = cga3d.Vector.from_pose(translation, [1, 0, 0, 0])
    result = versor.apply(cga3d.Vector.from_euclid(points)).to_euclid()
    print(reference)
    print(result)
    assert np.allclose(reference, result)


def test_rotation():
    points = np.eye(4, 3)
    quaternion = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    reference = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).apply(points)
    versor = cga3d.Vector.from_pose([0, 0, 0], quaternion)
    result = versor.apply(cga3d.Vector.from_euclid(points)).to_euclid()
    print(reference)
    print(result)
    assert np.allclose(reference, result)


def test_sqrt():
    m = cga3d.Vector.from_rotator(2 * e_1 ^ e_2) & cga3d.Vector.from_translator(1.3 * e_2 ^ e_inf)
    assert allclose(m.sqrt(is_rigid=True) ** 2, m)


# @pytest.mark.parametrize("t, r", [([0, 0, 0], [1, 0, 0]), ([2, 0, 0], [1, 4, 0]), np.random.random([2, 3])])
# @pytest.mark.parametrize("obj, offset", [(POSE_ORIGIN, -3), (FRAME, 0)])
# def test_pose_difference(t, r, obj, offset):
#     versor = cga3d.Vector.from_pos_and_rot_vector([*t, *r])
#     pose = versor.apply(obj)
#     distance = -(pose.checked() | obj).to_scalar().sum(-1)
#     reference = 2 + np.linalg.norm(t) ** 2 / 2 - 2 * np.cos(np.linalg.norm(r)) + offset
#     assert np.allclose(distance, reference)


def test_pose_distance():
    a = cga3d.Vector.from_pos_and_rot_vector(np.zeros([6]))
    b = cga3d.Vector.from_pos_and_rot_vector([0.3, 0, 0, 0.7, 0, 0])
    assert np.asarray(pose_distance(a, b, mode="position")) == pytest.approx(0.3**2)
    assert np.asarray(pose_distance(a, b, mode="orientation")) == pytest.approx(1 - np.cos(0.7))
    assert np.asarray(pose_distance(a, b, mode="both")) == pytest.approx(
        pose_distance(a, b, mode="position") + pose_distance(a, b, mode="orientation")
    )


@pytest.mark.parametrize(
    ("name", "versor", "target"),
    [
        ("rotation", cga3d.Vector.from_rotator(np.pi / 2 * e_1 ^ e_2), [-2, 1, 3]),
        (
            "translation",
            cga3d.Vector.from_translator(cga3d.Vector.from_direction(np.array([3, 0, 0])) ^ e_inf),
            [4, 2, 3],
        ),
        ("scaling", cga3d.Vector.from_scaling(10), [10, 20, 30]),
    ],
)
def test_transformation(name, versor, target):
    origin = cga3d.Vector.from_euclid([1, 2, 3])
    transformed_vector = versor.apply(origin)
    if name != "scaling":
        assert np.allclose((transformed_vector | e_inf).to_scalar(), -1)
    assert np.allclose(transformed_vector.to_euclid(), target)


@pytest.mark.parametrize("rotation", [False, [0, 0, 0], [1.1, 0.4, 0.1]])
@pytest.mark.parametrize("translation", [False, [0, 0, 0], [0.3, 0.7, 1.4]])
@pytest.mark.parametrize("dilation", [False, 0.31, 1, 7.3])
@pytest.mark.parametrize("only_2d", [False, True])
@pytest.mark.parametrize("shape", [(), (1,)])
def test_optimal_rotor(rotation, translation, dilation, only_2d, shape):
    np.random.seed(0)
    a = np.random.random([*shape, 5, 3])
    p = cga3d.Vector.from_euclid(a)
    versor = cga3d.Vector.from_identity()
    if only_2d:
        rotation = list(np.array([0, 0, 1]) * rotation)
        translation = list(np.array([1, 1, 0]) * translation)
    if rotation:
        versor = cga3d.Vector.from_rotator(cga3d.Vector.from_direction(rotation).dual(i3)) & versor
    if translation:
        versor = cga3d.Vector.from_translator(cga3d.Vector.from_direction(translation) ^ e_inf) & versor
    if dilation:
        versor = cga3d.Vector.from_scaling(dilation) & versor
    q = versor.apply(p)
    optimal_motor = cga3d.Vector.from_motor_estimation(
        p,
        q,
        translation=bool(translation),
        rotation=bool(rotation),
        dilation=bool(dilation),
        only_2d=only_2d,
    )
    assert np.allclose(optimal_motor.reverse_norm(), 1)
    print(versor)
    print(optimal_motor)
    assert allclose(versor, optimal_motor) or allclose(versor, -optimal_motor)


def test_optimal_rotor_from_frame():
    pose = cga3d.Vector.from_pos_and_rot_vector([6, 5, 4, 0.3, 0.2, 0.1])
    pose_test = cga3d.Vector.from_motor_estimation(FRAME, pose.apply(FRAME))
    assert allclose(pose, pose_test) or allclose(pose, -pose_test)


@pytest.mark.parametrize("rotation", [[0, 0, 0], [1.1, 0.4, 0.1]])
def test_optimal_rotation(rotation):
    np.random.seed(0)
    a = np.random.random([5, 3])
    p = cga3d.Vector.from_euclid(a)
    versor = cga3d.Vector.from_rotator(cga3d.Vector.from_direction(rotation).dual(i3))
    q = versor.apply(p)
    optimal_motor = _get_optimal_rotation(p, q)
    assert np.allclose(optimal_motor.reverse_norm(), 1)
    print(versor)
    print(optimal_motor)
    assert allclose(versor, optimal_motor) or allclose(versor, -optimal_motor)


@pytest.mark.parametrize("translation", [[0, 0, 0], [0.3, 0.7, 1.4]])
def test_optimal_translation(translation):
    np.random.seed(0)
    a = np.random.random([5, 3])
    p = cga3d.Vector.from_euclid(a)
    versor = cga3d.Vector.from_translator(cga3d.Vector.from_direction(translation) ^ e_inf)
    q = versor.apply(p)
    optimal_motor = _get_optimal_translation(p, q)
    assert np.allclose(optimal_motor.reverse_norm(), 1)
    print(versor)
    print(optimal_motor)
    assert allclose(versor, optimal_motor) or allclose(versor, -optimal_motor)


@pytest.mark.parametrize("type_", ["bivector", "line", "circle"])
@pytest.mark.parametrize("same", [True, False])
def test_point_pair_to_end_points(type_, same):
    if type_ == "bivector":
        a, b, c, d = cga3d.Vector.from_euclid(np.random.random([4, 3]))
        p = (a ^ b).normed()
        q = (c ^ d).normed()
    elif type_ == "line":
        a, b, c, d = cga3d.Vector.from_euclid([[-1, 0, 0], [1.5, 0, 0.1], [0, -0.2, 0], [1, 3, 0]])
        p = (a ^ b ^ e_inf).normed()
        q = (c ^ d ^ e_inf).normed()
    elif type_ == "circle":
        a, b, c, d, e, f = cga3d.Vector.from_euclid(
            [[-1, 0, 0], [1.5, 0, 0.1], [0, -0.2, 0], [1, 4, 0], [1, 3, 6], [7, 3, 0]]
        )
        p = (a ^ b ^ c).normed()
        q = (d ^ e ^ f).normed()
    else:
        raise NotImplementedError()
    if same:
        q = p
    motor = get_motor_from_object_pair(p, q)
    if same:
        assert allclose(motor, cga3d.Vector.from_identity())
    assert allclose(q, motor.apply(p))


def test_transformation_to_zero_mean_unit_variance():
    x = cga3d.Vector.from_euclid(100 * np.random.random([10, 3]) + [2, 40, 200])
    transformation = transformation_to_zero_mean_unit_variance(x)
    transformed_x = transformation.apply(x)
    assert np.allclose(np.mean(transformed_x.to_euclid(), axis=0), 0)
    assert np.allclose(sum(np.var(transformed_x.to_euclid(), axis=0)), 1)
    assert allclose(x, transformation.adjoint().apply(transformed_x))


@pytest.mark.parametrize("angle", [1, -1, 0.3, 1.3, np.pi / 2, np.pi - 1e-5])  # todo add 0
@pytest.mark.parametrize("v", [0, 1])
@pytest.mark.parametrize("t_gt", [0, 1, -2])
def test_motor_to_screw(angle, v, t_gt):
    versor = cga3d.Vector.from_rotator(angle * e_1 ^ e_2)
    versor = cga3d.Vector.from_translator(v * e_3 ^ e_inf).apply(versor)
    versor = cga3d.Vector.from_translator(t_gt * e_3 ^ e_inf) & versor
    line, t = versor.motor_to_screw()
    assert np.allclose(angle**2, line.square_norm())
    assert np.allclose(((line.normed() | (angle * (v * e_3).up() ^ e_3 ^ e_inf).normed()).to_scalar()), 1)
    assert allclose(line, angle * (v * e_3).up() ^ e_3 ^ e_inf)
    # assert np.allclose(t, t_gt)


@pytest.mark.parametrize("angle", [1, -1, 0.3, 1.3, np.pi / 2, np.pi - 1e-5])  # todo add 0
@pytest.mark.parametrize("pitch", [0, 0.1, -2])
def test_motor_from_and_to_screw(angle, pitch):
    line_gt = angle * cga3d.Vector.from_euclid([1.3, 0.7, 0]) ^ e_3 ^ e_inf
    versor = cga3d.Vector.from_screw(line_gt, pitch)
    line, t = versor.motor_to_screw()
    assert np.allclose(angle**2, line.square_norm())
    assert np.allclose(((line.normed() | line_gt.normed()).to_scalar()), 1)
    assert allclose(line, line_gt, 1e-4, 1e-4)


def test_motor_from_and_to_screw2():
    reference = [1, 0, 0, 1.2, 0, 0]
    tmp = cga3d.Vector.motor_to_screw(cga3d.Vector.from_pos_and_rot_vector(reference))
    print(tmp)
    result = cga3d.Vector.from_screw(*tmp).to_pos_and_rot_vector()
    assert np.allclose(reference, result)


def from_and_to_quaternion():
    quaternion = [1, 2, 3, 4]
    versor = cga3d.Vector.from_quaternion(quaternion)
    result = versor.to_quaternion()
    assert np.allclose(quaternion, result)


def test_to_transformation_matrix():
    v = cga3d.Vector.from_pos_and_rot_vector([4, 5, 6, 1, 2, 3])
    reference = v.apply(e_1.up()).to_euclid()
    result = np.asarray(v.to_transformation_matrix()) @ [1, 0, 0, 1]
    assert np.allclose(result, reference)


def test_from_and_to_rotation_matrix():
    quaternion = np.array([1, 2, 3, 4])
    matrix = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    versor = cga3d.Vector.from_rotation_matrix(matrix)
    result = versor.to_rotation_matrix()
    assert np.allclose(matrix, result)


def test_from_rotation_between_two_directions():
    p = cga3d.Vector.from_direction(np.random.random([10, 3])).normed()
    q = cga3d.Vector.from_direction(np.random.random([10, 3])).normed()
    v = cga3d.Vector.from_rotation_between_two_directions(p, q)
    assert allclose(v.apply(p).normed(), q.normed())
