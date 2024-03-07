"""Encapsulates different algebras (of various dimensions, signatures, ...)"""

from enum import Enum
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, Optional, Tuple, Union

import numpy as np

from geometricalgebra import cayley

DEFAULT_DTYPE = np.int8
Grades = FrozenSet[int]


def as_grades(grade: Union[int, Iterable[int]]) -> Grades:
    """Convert to Grades"""
    assert isinstance(grade, int) or all(isinstance(g, int) for g in grade), f"Unexpected type: {type(grade)}"
    return frozenset([grade]) if isinstance(grade, int) else frozenset(grade)


class ProductType(Enum):
    """Different kinds of binary operations on multi vectors"""

    GEOMETRIC = "geometric"
    INNER = "inner"
    OUTER = "outer"
    COMMUTATOR = "commutator"
    ANTICOMMUTATOR = "anticommutator"


class GeometricAlgebra:
    """Holding the definition and derives all properties of a geometric algebra

    Reference: https://en.wikipedia.org/wiki/Geometric_algebra

    Args:
        signature: tuple of +1 and -1. It should start with euclid_dim of +1. The length must be `dim'

    Attributes:
        basis: the basis of the multi vector
        dim: the number basis vectors
    """

    signature: Tuple[int, ...]
    basis: Tuple[Tuple[int, ...], ...]
    dim: int
    grade_to_slice: Tuple[slice, ...]
    dims_of_grade: Tuple[int, ...]
    max_grade: int

    _instances: Dict[Tuple[int, ...], "GeometricAlgebra"] = {}

    def __repr__(self):
        return f"GeometricAlgebra({self.signature})"

    def __new__(cls, signature: Tuple[int, ...]):
        if signature not in cls._instances:
            self = object.__new__(GeometricAlgebra)
            self.signature = signature
            self.basis = cayley.get_basis(len(signature))
            self.dim = len(self.basis)
            grades = tuple(len(b) for b in self.basis)
            self.max_grade = len(signature)
            self.dims_of_grade = tuple(grades.count(i) for i in range(self.max_grade + 1))
            self.grade_to_slice = tuple(
                slice(grades.index(i), grades.index(i) + self.dims_of_grade[i]) for i in range(self.max_grade + 1)
            )
            cls._instances[signature] = self
        return cls._instances[signature]

    def __getnewargs__(self):
        return (self.signature,)

    @lru_cache(2**8)
    def grades_to_mask(self, grades: Grades) -> Union[slice, np.ndarray]:
        """Returns a mask or slice that masks all components of any given grade."""
        if len(grades) == 1:
            return self.grade_to_slice[next(iter(grades))]
        mask = np.zeros(self.dim, dtype=bool)
        for grade in grades:
            mask[self.grade_to_slice[grade]] = True
        return mask

    @lru_cache(2**8)
    def get_generator(self, product_type: ProductType = ProductType.GEOMETRIC, dtype=np.int8):
        """Get the matrix for product in desired dtype while all results are cached"""
        if dtype == np.int8:
            if product_type is ProductType.COMMUTATOR:
                return self.get_generator(ProductType.GEOMETRIC) - self.get_generator(ProductType.GEOMETRIC).transpose()
            if product_type is ProductType.ANTICOMMUTATOR:
                return self.get_generator(ProductType.GEOMETRIC) + self.get_generator(ProductType.GEOMETRIC).transpose()
            result = cayley.get_cayley_table(self.signature)
            if product_type is ProductType.GEOMETRIC:
                return result
            for i in range(self.max_grade + 1):
                for j in range(self.max_grade + 1):
                    for k in range(self.max_grade + 1):
                        condition1 = product_type is ProductType.OUTER and i + j != k
                        condition2 = product_type is ProductType.INNER and (abs(i - j) != k or not i or not j)
                        if condition1 or condition2:
                            result[self.grade_to_slice[i], self.grade_to_slice[k], self.grade_to_slice[j]] = 0
            return result
        return self.get_generator(product_type).astype(dtype)

    @lru_cache(2**8)
    def get_reverse_generator(self):
        """Generator to reverse an element (reverse order of each basis element)"""
        result = np.ones(self.dim, dtype=DEFAULT_DTYPE)
        for grade, mask in enumerate(self.grade_to_slice):
            result[mask] = (-1) ** (grade * (grade - 1) // 2)
        return result

    @lru_cache(2**8)
    def get_reversed_norm_generator(self):
        """Optimized generator to calculate (a & a.reverse())(0)"""
        return np.diag(self.get_generator().reshape(3 * [self.dim])[:, 0, :]) * self.get_reverse_generator()

    @lru_cache(2**8)
    def get_square_norm_generator(self):
        """Optimized generator to calculate (a & a)(0)"""
        return np.diag(cayley.get_cayley_table(self.signature).astype(DEFAULT_DTYPE).reshape(3 * [self.dim])[:, 0, :])

    @lru_cache(2**8)
    def get_check_operator_generator(self):
        """Generator to calculate the checkes operator"""
        result = np.ones([self.dim], DEFAULT_DTYPE)
        for i in [0, 2, 4]:
            result[self.grade_to_slice[i]] = -1
        return result

    @lru_cache(2**8)
    def get_adjoint_generator(self):
        """Generator for the adjoint operation"""
        if self.signature != (1, 1, 1, 1, -1):
            raise NotImplementedError()
        return np.pad(np.ones(20, DEFAULT_DTYPE), (6, 6), constant_values=-1)

    @lru_cache(2**8)
    def mask_size(self, grades: Grades):
        """The size of the elements which need to be stored in a multivector with given grades

        Args:
            grades: all grades with non-zero elements

        Returns:
            The number of masked elements

        Example:
        >>> CGA3D_test = GeometricAlgebra((1, 1, 1, 1, -1))
        >>> CGA3D_test.mask_size(frozenset({1}))
        5
        >>> CGA3D_test.mask_size(frozenset({0, 3}))
        11
        """
        return sum(self.dims_of_grade[i] for i in grades)

    @lru_cache(2**12)
    def mask_from_grades(self, a: Grades, other: Optional[Grades] = None) -> Union[slice, np.ndarray]:
        if other is None:
            other = frozenset(range(len(self.dims_of_grade)))
        else:
            other = frozenset([other]) if isinstance(other, int) else frozenset(other)  # type: ignore
        if set(other) == set(a):
            return slice(None)
        if not a.intersection(other):
            return slice(0)
        if not a.issubset(other):
            raise ValueError(f"Argument other ({other}) must be a subset of self ({a})")
        if len(a) == 1:
            grade = next(iter(a))
            start = sum(self.dims_of_grade[i] for i in other if i < grade)
            return slice(start, start + self.dims_of_grade[grade])
        mask = np.zeros([self.mask_size(other)], dtype=bool)
        i = 0
        for grade in other:
            if grade in a:
                mask[i : i + self.dims_of_grade[grade]] = True
            i += self.dims_of_grade[grade]
        return mask

    @lru_cache(2**14)
    def grades_of_product(
        self, grades_left: Grades, grades_right: Grades, product_type: ProductType = ProductType.GEOMETRIC
    ) -> Grades:
        """Return the grades which might have non-zero entries given the grades of the two inputs multivectors

        Note: It gives a superset of the grades since depending on the actual values some grades may vanish,
            e.g., a & a

        Args:
            grades_left: the grades of the left argument
            grades_right: the grades of the right argument
            product_type: type of the product (inner, outer, geometric)

        Returns:
            grades of the operation a & b, a ^ b, a | b, depending on product_type

        Raises:
            TypeError: if product_type is of wrong type
        """
        all_pairs = ((i, j) for i in grades_left for j in grades_right)
        if product_type in (ProductType.GEOMETRIC, ProductType.COMMUTATOR, ProductType.ANTICOMMUTATOR):
            # TODO(dv): commutator and anticommutator output grades can be narrowed down (e.g. based on the generator matrix)
            result = (i for (a, b) in all_pairs for i in range(abs(a - b), a + b + 1, 2))
        elif product_type is ProductType.INNER:
            result = (abs(a - b) for (a, b) in all_pairs)
        elif product_type is ProductType.OUTER:
            result = (a + b for (a, b) in all_pairs)
        else:
            raise TypeError(f"Unexpected type {product_type}")
        return frozenset(filter(lambda i: i < len(self.dims_of_grade), result))
