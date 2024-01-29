"""A simultaneous  pose and correspondence registration via the covariant point drift"""

from geometricalgebra.cga import CGAVector


def e_step(p: CGAVector, q: CGAVector, v: CGAVector):
    """The E-step estimate the correspondences, i.e., the probabilities


    Args:
        p: tensor of shape (..., N) of vectors
        q: tensor of shape (..., M) of vectors
        v: tensor of shape (...,) of poses

    Returns:
        probabilities p of shape (..., N, M) where p[i, j] indicated whether the p[i] correspond to q[j]
    """
    logits = v[..., None].apply(p)[..., None].scalar_product(q[..., None, :])
    return p.framework().softmax(logits, axis=-1)


def m_step(p, q, prob, only_2d=False):
    """The M-step maximizes the likelihood by finding the optimal motor"""
    q_weighted = q[..., None, :] * prob
    motor = type(p).from_motor_estimation(p, q_weighted.sum(-1), only_2d=only_2d)
    variance = -2 * (motor[..., None].apply(p)[..., None].scalar_product(q_weighted(1))).sum(-1).mean(-1)
    variance_dir = -2 * ((motor[..., None].apply(p)[..., None].scalar_product(q_weighted(3))).sum(-1) - 1).mean(-1)
    variance = p.xnp().maximum(1e-6, variance)
    variance_dir = p.xnp().maximum(1e-6, variance_dir)
    return motor, (variance, variance_dir)


def pose_and_correspondence_registration(p: CGAVector, q: CGAVector, threshold=1e-3, max_iteration=50, only_2d=False):
    """A simultaneous pose and correspondence registration via the covariant point drift"""
    v = type(p).from_identity()
    variance = variance_dir = q.xnp().array(1e-6)
    last = variance
    for i in range(max_iteration):
        prob = e_step(p, (q(1) / variance[..., None] + 0.5 * q(3) / variance_dir[..., None]) * 2, v)
        v, (variance, variance_dir) = m_step(p, q, prob, only_2d)
        if variance * (1 + threshold) > last and i > 4:
            break
        last = variance
    return v, prob, variance
