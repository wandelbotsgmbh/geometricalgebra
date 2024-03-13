"""Elementary data types for geometric algebra"""

from __future__ import annotations

import os
from functools import partialmethod, reduce
from typing import Any, FrozenSet, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Type, TypeVar, Union, cast

import numpy as np

from geometricalgebra.algebra import Grades, ProductType, as_grades


class Framework(NamedTuple):
    """A framework offering numpy API"""

    numpy: Any
    linalg: Any
    softmax: Any


ga_numpy = os.environ.get("GEOMETRICALGEBRA_NUMPY", "numpy")
if ga_numpy == "jax":
    import jax
    import jax.numpy as jnp
    from jax import Array
    from jax.nn import softmax
    from jax.typing import ArrayLike

    jax.config.update("jax_enable_x64", True)
    FRAMEWORK = Framework(jnp, jnp.linalg, softmax)
elif ga_numpy == "tensorflow":
    import tensorflow.experimental.numpy as tnp  # pylint: disable=import-error
    import tensorflow.linalg as tf_linalg  # pylint: disable=import-error

    Array = ArrayLike = Any
    tnp.experimental_enable_numpy_behavior()  # dtype_conversion_mode="safe")
    tnp.linalg = tf_linalg
    from tensorflow.nn import softmax  # pylint: disable=import-error

    FRAMEWORK = Framework(tnp, tf_linalg, softmax)

elif ga_numpy == "numpy":
    from scipy.special import softmax

    FRAMEWORK = Framework(np, np.linalg, softmax)
    Array = ArrayLike = Any
else:
    raise NotImplementedError(f"Unknown backend: {ga_numpy}")

Subtype = TypeVar("Subtype", bound="Vector")

os.environ["KERAS_BACKEND"] = "tensorflow"


USER = os.getenv("GEOMETRIC_ALGEBRA_NUMPY_BACKEND")


class Vector:  # pylint: disable=too-many-public-methods
    """An array of multi vectors in a clifford algebra

    The last dimension is the vector (e.g. 32 dimensional for conformal 3D space)

    Represented in basis {1, e1, e2, e3, e+, e-, ...}

    OBSTACLES:
        __radd__: see test_scalar_addition for documentation
    """

    __array_priority__ = 15  # Make sure numpy lets MultiVectorTensor the precedence for binary operations

    __slots__ = "_values", "_grades"

    @classmethod
    def framework(cls) -> Framework:
        return FRAMEWORK

    @classmethod
    def basis(cls) -> Vector:
        return cls(cls.xnp().eye(cls.algebra.dims_of_grade[1]), grade=1)  # type: ignore

    @classmethod
    def xnp(cls):
        return cls.framework().numpy

    @classmethod
    def xlinalg(cls):
        return cls.framework().linalg

    def __init__(self, values: ArrayLike, grade: Union[int, Iterable[int]]):
        self._grades = as_grades(grade)
        self._values: Array = self.xnp().asarray(values)
        if not (self._values.shape and self.algebra.mask_size(self._grades) == self._values.shape[-1]):
            raise ValueError(
                f"Last dimension of values must be {self.algebra.mask_size(self._grades)} but the shape is {self._values.shape}"
            )

    def __call__(self: Subtype, grade: Union[int, Iterable[int]]) -> Subtype:
        """Return projection to grade or a set of grades

        Args:
            grade: The set of grades or a grade to which the vector gets projected

        Returns:
            The projection
        """
        grades = as_grades(grade).intersection(self._grades)
        mask = self.algebra.mask_from_grades(grades, self._grades)
        return type(self)(self._values[..., mask], grades)

    @classmethod  # type: ignore
    @property
    def algebra(cls):
        raise NotImplementedError()

    @property
    def _algebra(self):
        return self.algebra

    # def framework(self):
    #     return jnp

    def numpy(self):
        return self

    def __array__(self):
        # This is called when numpy tries to treat this as an array.
        # We prevent this since its behavior would depend on the representation of the multivectors
        raise TypeError(
            f"{type(self)} can not be converted implicitly to an array. "
            f"If conversion is intended use values property instead"
        )

    @classmethod
    def from_zero(cls: Type[Subtype], *, dtype=np.int8) -> Subtype:
        """Return a zero multivector (of empty grades)

        Args:
            dtype: the data type to represent the values

        Returns:
            A vector representing a zero
        """
        return cls(cls.xnp().zeros([0], dtype=dtype), [])

    @property
    def values(self) -> Array:
        """Read-only view on all values

        Returns:
            The matrix representation of the vector in the basis provided by its algebra
        """
        return self._expanded_grades()._values  # pylint: disable=protected-access

    @property
    def grades(self) -> Grades:
        """All grades of the vector

        The vectors are stored in a sparse format, i.e., only the elements for some grades are stores. However, these
        stores values might be explicilty zero.

        Returns:
             set of grades with exist in this multivector (cumulated over all axis). Note this means that these values
             can still be explicitly zero, however
        """
        return self._grades

    @property
    def grade(self) -> int:
        """Return set of grades with exist in this multivector (cumulated over all axis)

        Returns:
            It checks that this vector has only a single grade and returns that grade

        Raises:
            ValueError: if vector has several grades
        """
        if len(self.grades) != 1:
            raise ValueError("Multivector has several grades")
        return next(iter(self._grades))

    def _mask(self, other: Optional[Grades] = None) -> Union[slice, np.ndarray]:
        """Return the minimal mask containing all nonzeros grades based on _grades

        Note: if _grades is None, slice(None) is returned

        Args:
            other: enforce some grades

        Returns:
            The mask
        """
        # TODO: this is a blueprint to later allow for full grade-aware computing
        return self._algebra.mask_from_grades(self._grades, other)

    def exp(self: Vector, max_order: int = 10) -> Vector:
        """Calculates exponential (with Taylor expansion)

        Note: not optimized for numerical stability

        Args:
            max_order: the Taylor expansion is runcated at this order

        Returns:
            The exponential of this vector

        Example:
            This is a exponent of a pure quaternion:
        >>> from math import cos, sin
        >>> from geometricalgebra import cga3d
        >>> q = cga3d.e_1 ^ cga3d.e_2
        >>> allclose(q.exp(), (q & sin(1)) + cos(1))
        True
        """
        shape = [1 if d is None else d for d in self.shape]
        summands = [type(self).from_scalar(self.xnp().ones(shape, dtype=self.dtype))]
        for i in range(1, max_order + 1):
            summands.append(summands[-1] & self / i)
        return type(self).stack(summands).sum()

    @classmethod
    def stack(cls: Type[Subtype], values: Sequence[Subtype]) -> Subtype:
        if not values:
            raise ValueError("Can't stack an empty list")
        algebra = set(v._algebra for v in values)  # pylint: disable=protected-access
        if len(algebra) > 2:
            raise TypeError(f"All values must have the same algebra, but they are {algebra}")
        algebra = next(iter(algebra))
        grades: Grades = reduce(frozenset.union, [v.grades for v in values], frozenset())
        data = [v._expanded_grades(grades)._values for v in values]  # pylint: disable=protected-access
        result = cls.xnp().stack(data)
        return cls(result, grades)

    @classmethod
    def concatenate(cls: Type[Subtype], tensors: Sequence[Subtype], axis=0) -> Subtype:
        """Concatenates several Vector (tensors)

        Args:
            tensors: the list of tensors which get concatenated along a specified axis. Expect for this axis these
                tensors must be broadcastable
            axis: The axis along which the tensors get concatenated

        Returns:
            A tensor which contains all the tensors

        Raises:
            TypeError: if the tensors have different algebras

        Example:
            >>> from geometricalgebra import cga3d
            >>> a = cga3d.Vector.from_euclid([[1, 2, 3], [4, 5, 6]])
            >>> np.asarray(cga3d.Vector.concatenate([a, a]).to_euclid())
            array([[1., 2., 3.],
                   [4., 5., 6.],
                   [1., 2., 3.],
                   [4., 5., 6.]])
        """
        algebras = set(v._algebra for v in tensors)  # pylint: disable=protected-access
        if len(algebras) > 1:
            raise TypeError(f"All values must have the same algebra, but they are {algebras}")
        grades: Grades = frozenset(reduce(set.union, [v.grades for v in tensors], set()))  # type: ignore
        values_list = []
        for v in tensors:
            values_list.append(v._expanded_grades(grades)._values)  # pylint: disable=protected-access
        return cls(cls.xnp().concatenate(values_list, axis=axis), grades)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self: Subtype, item) -> Subtype:
        """
        Note: negative axis indices are not working and are highy discouraged to use
        """
        if isinstance(item, tuple):
            item = (*item, slice(None))
        return type(self)(self._values[item], self.grades)

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        return self._values.shape[:-1]

    @property
    def ndim(self) -> int:
        """Number of dimension of the tensor"""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Number of dimension of the tensor"""
        shape = self.shape
        if None in shape:
            raise ValueError("No size information in not fully specified arrays")
        return np.prod(cast(List[int], shape)).astype(int).item()

    @property
    def dtype(self):
        """The data type (as numpy datatype)"""
        dtype = self._values.dtype
        if hasattr(dtype, "as_numpy_dtype"):
            dtype = np.dtype(dtype.as_numpy_dtype)
        return dtype

    def copy(self: Subtype) -> Subtype:
        return type(self)(self._values.copy(), self._grades)

    def __neg__(self: Subtype) -> Subtype:
        return type(self)(-self._values, self._grades)  # pylint: disable=invalid-unary-operand-type

    def __pos__(self: Subtype) -> Subtype:
        return self

    def __truediv__(self: Subtype, other) -> Subtype:
        if isinstance(other, Vector):
            raise NotImplementedError()
        other = self.xnp().asarray(
            other,
        )
        return type(self)(self._values / other[..., None], self._grades)

    def astype(self: Subtype, dtype) -> Subtype:
        return type(self)(self.xnp().asarray(self._values, dtype), self._grades)

    def __len__(self) -> int:
        if not self.shape:
            raise ValueError("Only a real tensor has a len but not a single (multi) vector")
        return len(self._values)

    def view(self, type_):
        if not issubclass(type_, Vector):
            raise TypeError(f"Unexpected type {type(self)}")
        return type_(self._values, self._grades)

    def product(
        self: Subtype,
        other: Union[Subtype, float, int],
        projection: Optional[Union[int, Iterable[int]]] = None,
        product_type: ProductType = ProductType.GEOMETRIC,
    ) -> Subtype:
        """The geometric, inner, or outer product"""
        if isinstance(other, (np.ndarray, np.generic, float, int)):
            other = type(self).from_scalar(other)
        if self._algebra != other.algebra:
            raise TypeError(f"{self._algebra} {other.algebra}")
        out_grades = self._algebra.grades_of_product(self.grades, other.grades, product_type)
        if projection is not None:
            out_grades = out_grades.intersection(as_grades(projection))
        out_mask = self.grades_to_mask(out_grades)

        generator = self._algebra.get_generator(product_type, self.dtype)
        out = self.xnp().einsum(
            "...i,ikj,...j->...k",
            self._values,
            generator[:, :, other._mask()][:, out_mask][self._mask()],  # pylint: disable=protected-access
            other._values,  # pylint: disable=protected-access
        )
        return type(self)(out, out_grades)

    def _add(self: Subtype, other: Union[Vector, float, int], subtract: bool = False, right=False) -> Subtype:
        if isinstance(other, (float, int)) and not other:
            other = type(self).from_zero()
        elif isinstance(other, (np.ndarray, np.generic, float, int)):
            other = type(self).from_scalar(other)
        elif isinstance(other, Vector):
            pass
        else:
            return NotImplemented  # type: ignore
        grades = self.grades.union(other.grades)
        self_values = self._expanded_grades(grades)._values  # pylint: disable=protected-access
        other_values = other._expanded_grades(grades)._values  # pylint: disable=protected-access
        if not subtract:
            result_values = self_values + other_values
        elif right:
            result_values = other_values - self_values
        else:
            result_values = self_values - other_values
        return type(self)(result_values, grades)

    def _inplace_add(self, other: Vector, subtract: bool = False) -> Vector:
        assert self.grades == other.grades
        if not subtract:
            self._values += other._values  # pylint: disable=protected-access
        else:
            self._values -= other._values  # pylint: disable=protected-access
        return self

    def outer_prod(self: Subtype, other, projection=None) -> Subtype:
        return self.product(other, projection, product_type=ProductType.OUTER)

    def inner_prod(self: Subtype, other, projection=None) -> Subtype:
        return self.product(other, projection, product_type=ProductType.INNER)

    def geometric_prod(self: Subtype, other, projection=None) -> Subtype:
        return self.product(other, projection, product_type=ProductType.GEOMETRIC)

    __xor__ = outer_prod
    __or__ = inner_prod
    __and__ = geometric_prod
    __rand__ = geometric_prod

    def __add__(self: Subtype, other) -> Subtype:
        return self._add(other, subtract=False)

    def __sub__(self: Subtype, other) -> Subtype:
        return self._add(other, subtract=True)

    __iadd__ = partialmethod(_inplace_add, subtract=False)
    __isub__ = partialmethod(_inplace_add, subtract=True)
    __radd__ = __add__
    __rsub__ = partialmethod(_add, subtract=True, right=True)

    def __pow__(self, other: int):
        if other == 2:
            return self & self
        return NotImplemented

    # __rand__ == __and__  TODO: supported from python 3.9

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.geometric_prod(other)
        other = self.xnp().asarray(other)
        return type(self)(self._values * other[..., None], self._grades)

    def __rmul__(self, other):
        if isinstance(other, Vector):
            return other.geometric_prod(self)
        other = self.xnp().asarray(other)
        return type(self)(self._values * other[..., None], self._grades)

    def grades_to_mask(self, grades: Grades):
        return self._algebra.grades_to_mask(grades)

    def _expanded_grades(self, grades: Optional[Grades] = None) -> Vector:
        """Augment the grades of the vector (an fill with zeros)

        Example:
            >>> from geometricalgebra import cga3d
            >>> a = cga3d.e_1
            >>> a.grades
            frozenset({1})
            >>> a._expanded_grades(frozenset([1, 3, 4])).grades
            frozenset({1, 3, 4})
            >>> a._expanded_grades(frozenset([1])) is a
            True
        """
        if grades is None:
            grades = frozenset(range(self._algebra.max_grade + 1))
        if grades == self.grades:
            return self
        if not self.grades.issubset(grades):
            raise ValueError(f"Current grades {self.grades} mus be a subset of argument grades {grades}")
        if grades == self.grades:
            return self
        if not self.grades:
            return type(self)(self.xnp().zeros([*self.shape, self._algebra.mask_size(grades)], self.dtype), grades)
        tmp = [
            (
                self._values[..., self._algebra.mask_from_grades(frozenset([g]), self.grades)]
                if g in self.grades
                else self.xnp().zeros([*self.shape, self._algebra.mask_size(frozenset([g]))], dtype=self._values.dtype)
            )
            for g in grades
        ]
        result = self.xnp().concatenate(tmp, axis=-1)  # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
        return type(self)(result, grades)

    def mean(self: Subtype) -> Subtype:
        return type(self)(self.xnp().sum(self._values, 0), self._grades) / len(self)  # TODO class check, axis

    def sum(self: Subtype, axis: Union[int, List[int]] = 0) -> Subtype:
        if isinstance(axis, int):
            axis = [axis]
        else:
            axis = list(axis)
        for i, a in enumerate(axis):
            if a < 0:
                axis[i] -= 1  # axis reflects the axis of the MultiVectorTensor not of the underlying np.ndarray
        return type(self)(self.xnp().sum(self._values, tuple(axis)), self._grades)

    def gather(self, indices, axis):
        return type(self)(self.xnp().take(self._values, indices, axis), self._grades)

    def split(self: Subtype) -> List[Subtype]:
        length = self.shape[0]  # len(self) does not work for symbolic tensors in tf
        if length is None:
            raise TypeError("Can not split tensors of unknown dimension")
        return [self[i] for i in range(length)]

    def sparsified(self):
        """Keep only the grades with non-zero elements"""
        new_grades = set()
        for grade in self.grades:
            if self.xnp().count_nonzero(abs(self(grade)._values) > 1e-9):
                new_grades.add(grade)
        return self(new_grades)

    @classmethod
    def from_scalar(cls: Type[Subtype], scalar: ArrayLike, pseudo: bool = False) -> Subtype:
        """Create a multivector (tensor) representing scalar(s)

        Args:
            scalar: tensor of scalar inputs of shape (no copy is made)
            pseudo: if True returns a pseudo-scalar
            pseudo: if True, a pseudo scalar is created instead

        Returns:
            array whose number of dimension is one larger than scalar input
        """
        result = cls(cls.xnp().asarray(scalar)[..., None], grade=0)
        if pseudo:
            return result.dual()  # type: ignore
        return result

    def square(self: Subtype) -> Subtype:
        """Square norm (might be negative)

        Is equivalent to (self & self) but much faster
        """
        return type(self).from_scalar(self.square_norm())

    def square_norm(self) -> Array:
        """Square norm (might be negative)

        Returns:
             array of shape equal to self.shape

        Is equivalent to (self & self).to_scalar() but much faster
        Is equivalent to self.square().to_scalar()
        Example:
        >>> from geometricalgebra import cga3d
        >>> a, b = cga3d.Vector(np.random.normal(size=[2, 32]), set(range(6)))
        >>> np.allclose((a & b).to_scalar(), sum(a.values * a._algebra.get_square_norm_generator() * b.values), atol=1e-4)
        True
        """
        return self.xnp().einsum(
            "...i,i->...", self._values**2, self._algebra.get_square_norm_generator()[self._mask()]
        )

    def inv(self: Subtype) -> Subtype:
        """The inverse of the vector"""
        return self / self.square_norm()

    def scalar_product(self: Subtype, other: Subtype) -> Array:
        return self.product(other, 0).to_scalar()

    def to_scalar(self, pseudo: bool = False) -> Array:
        """Return the scalar(s)

        Args:
            pseudo: if True return the pseudo scalar instead

        Returns:
            The scalar as plain array

        Example:
        >>> from geometricalgebra import cga3d
        >>> np.asarray(cga3d.Vector.from_scalar(1).to_scalar())
        array(1)
        """
        if (self._algebra.max_grade if pseudo else 0) in self.grades:
            result = self._values[..., -1 if pseudo else 0]
        else:
            result = self.xnp().zeros([1 if s is None else s for s in self.shape], self.dtype)
        return result

    def reverse_norm(self) -> Array:
        return self.xnp().einsum(
            "...i,i->...",
            self._values**2,
            self._algebra.get_reversed_norm_generator()[self._mask()],
        )

    @classmethod
    def where(cls, condition, x, y):
        """Counterpart to np.where

        Args:
            condition: where to take first
            x: is selected if condition is True
            y: is selected otherwise

        Returns:
            Elementwise either x or y depending on whether condition is True or not.

        Example:
        >>> from geometricalgebra import cga3d
        >>> cga3d.Vector.where([True, False], cga3d.e_1, cga3d.e_2)
        Vector([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        """
        condition = cls.xnp().asarray(condition)
        assert x.grades == y.grades
        assert x.algebra is y.algebra
        result_values = cls.xnp().where(condition[..., None], x._values, y._values)  # pylint: disable=protected-access
        return cls(result_values, x.grades, x.algebra)  # pylint: disable=too-many-function-args

    def ravel(self: Subtype) -> Subtype:
        return type(self)(self._values.reshape([-1, self._values.shape[-1]]), self._grades)

    def __setitem__(self: Subtype, key, value: Subtype):
        """
        Note: negative axis indices are not working and are highy discouraged to use
        """
        if isinstance(key, tuple):
            key = (*key, slice(None))
        assert self._grades == value._grades
        self._values[key] = value._values

    def sandwich(self: Subtype, other: Subtype) -> Subtype:
        """Return the sandwich operation"""
        return (self & other & self.inverse())(other.grades)  # type: ignore

    def __repr__(self):
        # if self.framework() is jnp:
        # return repr(self.numpy())
        #    return f"<{type(self).__name__} of shape {self.shape}>"
        return (
            repr(np.asarray(self.values))
            .replace(", dtype=float64", "")
            .replace(",      dtype=float64", "")
            .replace(",      dtype=float64", "")
            .replace("dtype=float64", "")
            .replace("array", type(self).__name__)
            .replace("       ", (len(type(self).__name__) + 2) * " ")
        )

    def __abs__(self: Subtype) -> Subtype:
        return self.square().sqrt()

    def left_product_matrix(
        self,
        product_type: ProductType,
        in_grade: Union[None, int, FrozenSet[int]] = None,
        out_grade: Union[None, int, FrozenSet[int]] = None,
    ) -> Array:
        """Return an array that perform a, e.g, geometric product by a matrix multiplication

        Args:
            product_type: type of the product
            in_grade: mask
            out_grade: mask

        Returns:
            The product matrix

        Example:
        >>> from geometricalgebra import cga3d
        >>> a, b = cga3d.Vector(np.random.normal(size=[2, 32]), grade=set(range(6)))
        >>> matrix = a.left_product_matrix(ProductType.GEOMETRIC)
        >>> np.allclose(matrix @ b.values, (a & b).values)
        True
        """
        in_mask = self._algebra.grades_to_mask(in_grade) if in_grade is not None else slice(None)
        out_mask = self._algebra.grades_to_mask(out_grade) if out_grade is not None else slice(None)
        return self.xnp().einsum(
            "...i,ijk->...jk",
            self._values,
            self._algebra.get_generator(product_type, self.dtype)[self._mask(), out_mask, in_mask],
        )

    def right_product_matrix(self, product_type: ProductType) -> Array:
        """Return an array that perform an, e.g, geometric product by a matrix multiplication

        Args:
            product_type: type of the product

        Returns:
            The product matrix

        Example:
        >>> from geometricalgebra import cga3d
        >>> a, b = cga3d.Vector(np.random.normal(size=[2, 32]), grade=set(range(6)))
        >>> matrix = a.right_product_matrix(ProductType.GEOMETRIC)
        >>> np.allclose(matrix @ b.values, (b & a).values)
        True
        """
        return self.xnp().einsum(
            "...i,kji->...jk", self._values, self._algebra.get_generator(product_type, self.dtype)[..., self._mask()]
        )

    def reshape(self, shape):
        """Reshape like in numpy

        Example:
        >>> from geometricalgebra import cga3d
        >>> cga3d.Vector.from_euclid(np.zeros([3, 4, 3])).reshape([2, -1]).shape
        (2, 6)
        """
        return type(self)(  # pylint: disable=too-many-function-args
            self._values.reshape([*shape, self._values.shape[-1]]), self._grades, self._algebra
        )

    def adjoint(self: Subtype) -> Subtype:
        return type(self)(self._algebra.get_adjoint_generator()[self._mask()] * self._values, self._grades)

    def reversed(self: Subtype) -> Subtype:
        """Reversed
        Reference:
            Dorst et al. "Geometric algebra for computer science", Chap. 2.9.4
        """
        return type(self)(self._algebra.get_reverse_generator()[self._mask()] * self._values, self._grades)

    def inverse(self: Subtype) -> Subtype:
        tmp = self.reversed()
        return tmp / (tmp & self).to_scalar()  # TODO: improve tmp & self like normal square norm

    def checked(self: Subtype) -> Subtype:
        """Return the check operator applied to self

        References:
            Leo Dorst, Guide to Geometruc Algebra in Practice, Chapter 2.4.1
        """
        return type(self)(
            self._algebra.get_check_operator_generator()[self._mask()] * self._values,
            self._grades,
        )

    def sqrt(self: Subtype, is_rigid=False) -> Subtype:
        """

        References:
            Dorst et al. "Geometric algebra for computer science", Chap. 5.2.9
        """
        if is_rigid:
            result = (1 + self) & (1 + self(0) - self(4) / 2)
            return result / self.xnp().sqrt(result.reverse_norm())
        if self._grades == {0}:
            return type(self)(self.xnp().sqrt(self._values), self._grades)
        raise NotImplementedError()

    def commutator(self: Subtype, other: Subtype, anti=False, projection=None) -> Subtype:
        """The commutator or anticommutator

        The commutator is: (a & b) - (b & a)
        The antocommutator is: (a & b) + (b & a)
        This is a more efficient implementation
        """
        product_type = ProductType.ANTICOMMUTATOR if anti else ProductType.COMMUTATOR
        return self.product(other, projection, product_type=product_type)

    def anticommutator(self: Subtype, other: Subtype, projection=None) -> Subtype:
        return self.commutator(other, True, projection)


def allclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    """Compare two vectors for equality within the numerical precision

    Args:
        a: first entities to compare
        b: second entities to compare
        rtol: see np.allcose
        atol: see np.allcose
        equal_nan: see np.allcose

    Returns:
         True if all vectors in `a` are close the corresponding one in `b`
    """

    return np.allclose(a.values, b.values, rtol, atol, equal_nan)
