# """Test that tensorflow is doing the same as numpy"""
# import pytest
# import tensorflow as tf
#
# from geometricalgebra.multivector import EagerTensor, MultiVectorTensor, VersorTensor, allclose, e_inf, i3
#
#
# @pytest.mark.parametrize(
#     "op",
#     [
#         lambda a, a_np, b: a + b,
#         lambda a, a_np, b: a_np + b,
#         lambda a, a_np, b: b + a_np,
#         lambda a, a_np, b: a - b,
#         lambda a, a_np, b: a / 2,
#         lambda a, a_np, b: a ^ b,
#         lambda a, a_np, b: a | b,
#         lambda a, a_np, b: a & b,
#         lambda a, a_np, b: a_np & b,
#         lambda a, a_np, b: b & a_np,
#         lambda a, a_np, b: a + 0,
#         lambda a, a_np, b: 1 + a,
#         lambda a, a_np, b: +a,
#         lambda a, a_np, b: -a,
#         lambda a, a_np, b: MultiVectorTensor.from_scalar(a.square_norm()),
#         lambda a, a_np, b: a.square(),
#         lambda a, a_np, b: sum([a, a, b, b], a),
#         lambda a, a_np, b: sum([a, a, b, b]),
#         lambda a, a_np, b: a.exp(),
#         lambda a, a_np, b: a.dual(),
#         lambda a, a_np, b: VersorTensor.from_translator(a ^ e_inf),
#         lambda a, a_np, b: VersorTensor.from_rotator(a & i3.inv()),
#     ],
# )
# def test_elementary_operations(op):
#     data_a = [1.0, 2.0, 3.0]
#     data_b = tf.constant([3.0, 4.0, 5.0])
#     a = MultiVectorTensor.from_direction(tf.constant(data_a))
#     a_np = MultiVectorTensor.from_direction(data_a)
#     b = MultiVectorTensor.from_direction(data_b)
#     assert isinstance(op(a, a_np, b)._values, EagerTensor)  # pylint: disable=protected-access
#     result_numpy = op(a.numpy(), a_np, b.numpy())
#     result_tensorflow = op(a, a_np, b).numpy()
#     print(result_numpy)
#     print(result_tensorflow)
#     assert allclose(result_numpy, result_tensorflow, 1e-6, 1e-6)
#
#
# def test_gradient():
#     a_data = tf.Variable([2, 4.0, 3])
#     b_data = tf.Variable([2, 2, 1.0])
#
#     with tf.GradientTape() as tape:
#         a = MultiVectorTensor.from_direction(a_data)
#         b = MultiVectorTensor.from_direction(b_data)
#         c = (b | a).square_norm()
#
#     dc_da = tape.gradient(c, a_data)
#     assert dc_da is not None
#
