"""Generating Caylay tables (for geometric algebras)"""

from itertools import combinations, groupby
from typing import Sequence, Tuple

import numpy as np


def _kendall_tau_distance(s: Sequence[int], t: Sequence[int], /) -> int:
    """Return the Kendall tau distance between two sequences

    Args:
        s: first sequence
        t: second sequence with the same length as values1

    Returns:
        the (unnormalized) Kendall tau distance

    Reference:
        https://en.wikipedia.org/wiki/Kendall_tau_distance
        (Also code adapted from there)

    Raises:
        ValueError: if both sequences do not have same length

    Example:
    >>> _kendall_tau_distance([1, 2, 3], [1, 3, 2])
    1
    >>> _kendall_tau_distance([1, 2, 3], [1, 2, 3])
    0
    >>> _kendall_tau_distance([1, 2, 3], [3, 2, 1])
    3
    """
    if len(s) != len(t):
        raise ValueError("Both sequences must have same length")
    i, j = np.meshgrid(np.arange(len(s)), np.arange(len(t)))
    a = np.argsort(s, kind="stable")
    b = np.argsort(t, kind="stable")
    result = np.sum((a[i] < a[j]) * (b[i] > b[j]) + (a[i] > a[j]) * (b[i] < b[j])) // 2
    return result.item()


def get_basis(size: int) -> Tuple[Tuple[int, ...], ...]:
    """Return the basis of the exterior algebra

    Args:
        size: the number of basis vectors

    Returns:
        All basis vectors for the multivector space

    Examples:
    >>> get_basis(3)
    ((), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3))
    """
    return tuple(indices for grade in range(size + 1) for indices in list(combinations(range(1, size + 1), grade)))


def _multiply(s1, s2, signature) -> Tuple[Tuple[int, ...], int]:
    merged = s1 + s2
    normed, sign = _normalize(merged)
    result = []
    for key, group in groupby(normed):
        a, b = divmod(len(list(group)), 2)
        if b:
            result.append(key)
        sign = sign * signature[key - 1] ** a
    return tuple(result), sign


def _normalize(s: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """Return a sorted sequence and +1 (-1) if the number of swaps of adjacent elements was even (odd), respectively

    Example:
    >>> _normalize((1,2,3))
    ((1, 2, 3), 1)
    >>> _normalize((1,3,2))
    ((1, 2, 3), -1)
    >>> _normalize((3,2,1))
    ((1, 2, 3), -1)
    """

    index = tuple(sorted(s))
    sign = (-1) ** _kendall_tau_distance(s, index)
    return index, sign


def get_cayley_table(signature: Tuple[int, ...]):
    """Create Caylay table from signature

    A description of a (similar) algorithm can be found in: https://rjw57.github.io/phd-thesis/rjw-thesis.pdf

    Args:
        signature: the signature of the algebra

    Returns:
        The Cayley table
    """
    basis_indices = get_basis(len(signature))
    n_dim = len(basis_indices)
    c = np.zeros([n_dim, n_dim, n_dim], dtype=np.int8)
    for i in range(n_dim):
        for j in range(n_dim):
            a = basis_indices[i]
            b = basis_indices[j]
            index, sign = _multiply(a, b, signature)
            c[i, basis_indices.index(index), j] = sign
    return c
