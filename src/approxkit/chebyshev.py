from __future__ import annotations
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import Any, Literal, Sequence, Self

import numpy as np
from numpy.polynomial.chebyshev import chebval  # chebpts1, chebpts2
from numpy.polynomial.polyutils import RankWarning
from numpy.typing import ArrayLike, NDArray

from scipy.fft import dct


def map_from_interval(
    x: ArrayLike,
    a: float,
    b: float,
) -> NDArray:
    """F(x), where F: [a,b] -> [-1,1].

    Examples
    --------
    >>> x = np.array([-1, 0, 1])
    >>> np.allclose(
    ...     map_from_interval(map_to_interval(x, 2, 4), 2, 4),
    ...     x
    ... )
    True
    """
    return (x - (b + a) / 2.0) * (2.0 / (b - a))


def map_to_interval(
    x: ArrayLike,
    a: float,
    b: float,
) -> NDArray:
    """F(x), where F: [-1,1] -> [a,b]."""
    return (x * (b - a) + (b + a)) / 2.0


def chebyshev_lobatto_nodes(n: int) -> NDArray:
    """
    Return Chebyshev-Lobatto nodes (roots of the derivative of the
    Chebyshev polynomial of the first kind).

    Parameters
    ----------
    n : int
        Degree of Chebyshev polynomial T_n.

    Notes
    ------
    Because the extrema of Chebyshev polynomials of the first
    kind occur at ±1, these points are often used as initial
    nodes in minimax approximation algorithms.

    Examples
    --------
    >>> from numpy.polynomial.chebyshev import chebpts2, chebval
    >>> x = chebyshev_lobatto_nodes(4)
    >>> chebval(x, [0]*4 + [1])
    array([ 1., -1.,  1., -1.,  1.])

    Equals the Chebyshev points of the second kind.
    >>> y = chebpts2(5)
    >>> np.allclose(x, y)
    True

    References
    ----------
    http://en.wikipedia.org/wiki/Chebyshev_nodes
    http://en.wikipedia.org/wiki/Chebyshev_polynomials
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([0.0])
    return -np.cos((np.pi * np.arange(n + 1)) / n)


chebextr = chebyshev_lobatto_nodes  # alias


def chebyshev_nodes(n: int) -> NDArray:
    """
    Return Chebyshev nodes (roots of the Chebyshev polynomial of the first kind).

    Parameters
    ----------
    n : int
        Degree of Chebyshev polynomial T_n.

    Notes
    -----
    The roots of the Chebyshev polynomial of the first kind form a particularly
    good set of nodes for polynomial interpolation because the resulting
    interpolation polynomial minimizes the problem of Runge's phenomenon.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.chebyshev import chebpts1, chebval

    >>> x = chebyshev_nodes(3)
    >>> np.allclose(chebval(x, [0, 0, 0, 1]), [0, 0, 0])
    True

    Equals the Chebyshev points of the first kind.
    >>> x1 = chebpts1(3)
    >>> np.allclose(x, x1)
    True

    References
    ----------
    http://en.wikipedia.org/wiki/Chebyshev_nodes
    http://en.wikipedia.org/wiki/Chebyshev_polynomials
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return - np.cos(np.pi * (np.arange(n) + 0.5) / n)


chebroot = chebyshev_nodes  # alias


def _check_domain(
    domain: ArrayLike,
    ndim: int,
) -> NDArray:
    """Validate and normalize domain intervals."""
    domain = np.asarray(domain)
    if domain.size % 2:
        raise ValueError(
            "domain must contain pairs (a, b)"
        )
    domain = domain.reshape((-1, 2))
    if domain.shape[0] != ndim:
        raise ValueError(
            "domain must contain one interval per dimension"
        )
    return domain


@dataclass
class ChebyshevND:
    """
    N-dimensional Chebyshev approximation.

    Parameters
    ----------
    coef : ndarray
        Chebyshev coefficients.
    domain : array_like, optional
        Domain intervals [(a1, b1), ..., (an, bn)].

    Examples
    --------
    >>> approx = ChebyshevND.fit_dct(np.exp, n=9, domain=[(0, 2)])
    >>> approx
    ChebyshevND(degree=(8,), domain=[[0, 2]])

    >>> x = np.linspace(0, 2, 5)
    >>> y = approx(x)
    >>> np.allclose(y, np.exp(x), atol=1e-10)
    True

    2D:
    >>> fun = lambda x, y: np.tanh(x + y)
    >>> approx = ChebyshevND.fit_dct(fun, n=(12, 12))
    >>> approx
    ChebyshevND(degree=(11, 11), domain=None)

    >>> x = np.linspace(-1, 1, 5)
    >>> X, Y = np.meshgrid(x, x, indexing="ij")
    >>> Z = approx(X, Y)
    >>> np.allclose(fun(X,Y), Z, atol=1e-10)
    True
    """
    coef: ArrayLike
    domain: ArrayLike | None = None

    def __post_init__(self) -> None:
        self.coef = np.asarray(self.coef)
        if self.domain is not None:
            self.domain = _check_domain(
                self.domain,
                self.coef.ndim
            )

    @property
    def degree(self) -> tuple[int, ...]:
        """Degree of polynomial(s)"""
        return tuple(n - 1 for n in self.coef.shape)

    @property
    def ndim(self) -> int:
        return self.coef.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.coef.shape

    def _normalize(
        self,
        *xi: ArrayLike,
    ) -> tuple[NDArray, ...]:
        if len(xi) != self.ndim:
            raise ValueError(
                f"expected {self.ndim} coordinates, got {len(xi)}"
            )
        if self.domain is None:
            return tuple(np.asarray(x) for x in xi)

        return tuple(
            map_from_interval(x, d[0], d[1])
            for x, d in zip(xi, self.domain)
        )

    def eval_normalized(self, *xi: ArrayLike) -> NDArray:
        return chebvalnd(self.coef, *xi)

    def __call__(self, *xi: ArrayLike) -> NDArray:
        """
        Evaluate the approximation.
        """
        xi = self._normalize(*xi)
        return chebvalnd(self.coef, *xi)

    def copy(self) -> Self:
        return type(self)(
            self.coef.copy(),
            None if self.domain is None else self.domain.copy(),
        )

    def grid(self, *xi: ArrayLike) -> NDArray:
        """
        Evaluate on Cartesian product grid.

        Notes
        -----
        xi are physical coordinates.
        """
        xi = self._normalize(*xi)
        return chebgridnd(self.coef, *xi)

    def truncate(self, *degrees: int) -> Self:
        """
        Return lower-order approximation.
        """
        if len(degrees) != self.ndim:
            raise ValueError(
                f"expected {self.ndim} degrees, got {len(degrees)}"
            )
        if any(d < 0 for d in degrees):
            raise ValueError(
                "degrees must be non-negative"
            )
        slices = tuple(slice(0, d + 1) for d in degrees)
        return type(self)(
            self.coef[slices],
            None if self.domain is None else self.domain.copy(),
        )

    def __repr__(self) -> str:
        return (
            f"ChebyshevND("
            f"degree={self.degree}, "
            f"domain={self.domain.tolist() if self.domain is not None else None}"
            f")"
        )

    @classmethod
    def fit_dct(
        cls,
        f: Callable[..., ArrayLike] | ArrayLike,
        n: int | Sequence[int] = (10,),
        domain: ArrayLike | None = None,
        **kwargs: Any,
    ) -> Self:
        coef = chebfit_dct(
            f,
            n=n,
            domain=domain,
            **kwargs,
        )
        return cls(coef, domain)

    @classmethod
    def fit(
        cls,
        xi: tuple[ArrayLike, ...],
        values: ArrayLike,
        deg: Sequence[int] | NDArray,
        **kwargs: Any,
    ) -> Self:
        coef = chebfitnd(
            xi,
            values,
            deg,
            **kwargs,
        )
        return cls(coef)


def chebfit_dct(
    f: Callable[..., ArrayLike] | ArrayLike,
    n: int | Sequence[int] = (10,),
    domain: ArrayLike | None = None,
    args: tuple[Any, ...] = (),
    indexing: Literal["ij", "xy"] = "ij"
) -> NDArray:
    """
    Fit Chebyshev series to N-dimensional function
    so that f(x1, x2,..., xn) can be approximated by:

    .. math:: f(x_1, x_2,...,x_n) =
                    \\sum_{i,j,...k} c_i T_i(x_1)*...*c_k T_k(x_n) ,

    where Tk is the k'th Chebyshev polynomial of the first kind.

    Parameters
    ----------
    f : callable or array_like
        Function to approximate, or function values sampled at
        Chebyshev nodes.
    n : int or sequence of int, optional
        Number of Chebyshev nodes used in each dimension.
        Default n=10. Values larger than about 50 may lead to noisy
        high-order coefficients.
    domain : array_like, optional
        Domain intervals [(a1, b1), ..., (an, bn)].
        (default domain = [(-1, 1)] * len(n))
    args :
        additional arguments to pass to f.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy') or matrix ('ij', default) indexing of output.

    Returns
    -------
    ck : ndarray
        polynomial coefficients in Chebyshev form.

    Notes
    -----
    If `f` is callable, it is evaluated at Chebyshev nodes.

    If `f` is array_like, it is interpreted as values already
    sampled at Chebyshev nodes and transformed directly.

    For large n, higher-order coefficients may become dominated by
    numerical noise. These noisy coefficients do not improve the
    approximation and can reduce accuracy.

    Examples
    --------
    Fit exponential function

    >>> x = chebyshev_nodes(9)
    >>> c9 = chebfit_dct(lambda x: np.tanh(x) + 0.5, 9)
    >>> np.allclose(c9, [5.00000000e-01,   8.11675684e-01,  -9.86864911e-17,
    ...                 -5.42457905e-02,  -2.71387850e-16,   4.51658839e-03,
    ...                  2.46716228e-17,  -3.79694221e-04,  -3.26899002e-16])
    True
    >>> np.allclose(chebvalnd(c9, x), np.tanh(x)+0.5)
    True

    >>> x1,x2 = np.meshgrid(x, x, indexing="ij")
    >>> c99 = chebfit_dct(lambda x,y: np.tanh(x+y) + 0.5, n=(9, 9))
    >>> np.allclose(chebvalnd(c99, x1, x2), np.tanh(x1+x2)+0.5)
    True

    >>> domain = (0, 2)
    >>> ck7 = chebfit_dct(np.exp, 7, domain)
    >>> np.allclose(ck7, [3.44152387e+00,   3.07252345e+00,   7.38000848e-01,
    ...                   1.20520053e-01,   1.48805268e-02,   1.47579673e-03,
    ...                   1.21719524e-04])
    True
    >>> x7 = map_to_interval(chebyshev_nodes(7), *domain)
    >>> ck7b = chebfit_dct(np.exp(x7))
    >>> np.allclose(ck7, ck7b)
    True

    >>> x9 = map_to_interval(chebyshev_nodes(9), *domain)
    >>> ck9 = chebfit_dct(np.exp(x9))
    >>> np.allclose(ck9, [3.44152387e+00, 3.07252345e+00, 7.38000848e-01,
    ...                   1.20520053e-01, 1.48805283e-02, 1.47582673e-03,
    ...                   1.22261037e-04, 8.69418381e-06, 5.40019009e-07])
    True
    >>> ck49 = chebfit_dct(np.exp, 49, domain)
    >>> len(ck49)
    49
    >>> ck49m = np.array(ck49)
    >>> ck49m[np.abs(ck49) < 1e-14] = 0  # Truncate noisy coefs

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 4)
    >>> xn = map_from_interval(x, *domain)
    >>> y = np.exp(x)

    >>> h1 = plt.plot(x, y - chebvalnd(ck7, xn), 'g.', label='ck7')
    >>> h2 = plt.plot(x, y - chebval(xn, ck9),'b.', label='ck9')
    >>> h3 = plt.plot(x, y - chebval(xn, ck49),'r.', label='ck49')
    >>> h4 = plt.plot(x, y - chebval(xn, ck49m),'m.', label='ck49m')
    >>> h5 = plt.legend()
    >>> h6 = plt.title('Errors for approximating np.exp')
    >>> plt.close()

    See also
    --------
    chebval, chebvalnd

    References
    ----------
    http://en.wikipedia.org/wiki/Chebyshev_nodes

    Weisstein, Eric W. "Chebyshev Approximation Formula."
    From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/ChebyshevApproximationFormula.html

    W. Fraser (1965)
    "A Survey of Methods of Computing Minimax and Near-Minimax Polynomial
    Approximations for Functions of a Single Independent Variable"
    Journal of the ACM (JACM), Vol. 12 ,  Issue 3, pp 295 - 314
    """
    n = np.asarray(np.atleast_1d(n), dtype=int)
    if np.any(n <= 0):
        raise ValueError("n must contain positive integers")
    if np.any(n > 50):
        warnings.warn(
            "n > 50 may lead to noisy high-order coefficients",
            stacklevel=2
        )

    if callable(f):
        if domain is None:
            domain = [(-1, 1)] * len(n)
        domain = _check_domain(domain, len(n))
        xi = [map_to_interval(chebyshev_nodes(ni), d[0], d[1])
              for ni, d in zip(n, domain)]
        Xi = tuple(np.meshgrid(*xi, indexing=indexing))
        ck = np.asarray(f(*(Xi + args))) / np.prod(n)
        expected_shape = Xi[0].shape

        if ck.shape != expected_shape:
            raise ValueError(
                f"expected function values with shape "
                f"{expected_shape}, got {ck.shape}"
            )
    else:
        n = np.shape(f)
        ck = np.asarray(f) / np.prod(n)


    ndim = len(n)

    for i in range(ndim):
        ck = dct(ck[..., ::-1])
        # Adjust the constant term for the Chebyshev/DCT-I normalization.
        ck[..., 0] /= 2.0
        if i < ndim - 1 or indexing == "ij":
            ck = np.rollaxis(ck, axis=-1)
    return ck


def chebfitnd(
    xi: tuple[ArrayLike, ...],
    f: ArrayLike,
    deg: Sequence[int] | NDArray,
    rcond: float | None = None,
    full: bool = False,
    w: ArrayLike | None = None,
) -> NDArray | tuple[NDArray, list[Any]]:
    """
    Least squares fit of Chebyshev series to N-dimensional data.
    Return the coefficients of a Chebyshev series of degree `deg` that is the
    least squares fit to the data values `f` given at points
    `x1`, `x2`,..., `xn`

    The fitted polynomial(s) are in the form
    .. math::  p(x,y) = c_00 + c_11 * T_1(x)*T_1(y) + ..c_ij * T_i(x)*T_j(y).
                        + c_nm * T_n(x)*T_m(y),
    where `n`, `m` is `deg`.

    Parameters
    ----------
    xi: tuple of array_like
        x1-, x2-,....xn-coordinates of the sample points.
    f : array_like
        function values at the sample points ``(x1[i], x2[i], ..., xn[i])``.
    deg : sequence of int
        Degrees of the fitting series in the x1, x2, ..., xn directions,
        respectively.
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is size(x1)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, optional
        Weights. If not None, the contribution of each point
        ``(x1[i], x2[i], ..., xn[i])`` to the fit is weighted by `w[i]`.
        Ideally the weights are chosen so that the errors of the products
        ``w[i]*f[i]`` all have the same variance.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (M1, M2,..., Mn)
        Chebyshev coefficients ordered from low to high.
    [residuals, rank, singular_values, rcond] : list
        These values are only returned if `full` = True
        resid -- sum of squared residuals of the least squares fit
        rank -- the numerical rank of the scaled Vandermonde matrix
        sv -- singular values of the scaled Vandermonde matrix
        rcond -- value of `rcond`.
        For more details, see `linalg.lstsq`.
    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if `full` = False.  The
        warnings can be turned off by
        >>> import warnings
        >>> warnings.simplefilter('ignore', RankWarning)

    See Also
    --------
    chebvalnd, chebgridnd

    Notes
    -----
    The solution is the coefficients of the Chebyshev series `p` that
    minimizes the sum of the weighted squared errors
    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,
    where :math:`w_j` are the weights. This problem is solved by setting up
    as the (typically) overdetermined matrix equation
    .. math:: V(x, y) * c = w * y,
    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.
    If some of the singular values of `V` are so small that they are
    neglected, then a `RankWarning` will be issued. This means that the
    coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.
    Fits using Chebyshev series are usually better conditioned than fits
    using power series, but much can depend on the distribution of the
    sample points and the smoothness of the data. If the quality of the fit
    is inadequate splines may be a good alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           http://en.wikipedia.org/wiki/Curve_fitting
    Examples
    --------
    """
    def _check_shapes(z, xi):
        ndims = np.array([np.ndim(x) for x in xi])
        sizes = np.array([np.size(x) for x in xi])
        ndim = len(ndims)
        if np.any(ndims != ndim) or z.ndim != ndim:
            msg = f"expected {ndim}-dimensional arrays for all xi and f"
            raise TypeError(msg)
        if np.any(sizes == 0):
            raise TypeError("expected non-empty vector for xi")

    def _check_size(w, n):
        if n != len(w):
            raise TypeError("expected x and w to have same length")

    def _scale(lhs):
        if issubclass(lhs.dtype.type, np.complexfloating):
            scl = np.sqrt((np.square(lhs.real) +
                           np.square(lhs.imag)).sum(axis=0))
        else:
            scl = np.sqrt(np.square(lhs).sum(axis=0))
        scl[scl == 0] = 1
        return scl

    def _init(xi, z, w, degrees, order):
        lhs = chebvandernd(degrees, *xi).reshape((-1, order))
        rhs = z.ravel()
        if w is not None:
            w = np.asarray(w).ravel() + 0.0
            _check_size(w, len(lhs))
            lhs = lhs * w
            rhs = rhs * w
        scl = _scale(lhs)
        return lhs, rhs, scl

    # xi = np.array(xi, copy=0) + 0.0
    z = np.array(f)
    _check_shapes(z, xi)

    degrees = np.asarray(deg, dtype=int)
    orders = degrees + 1
    order = int(np.prod(orders))

    lhs, rhs, scl = _init(xi, z, w, degrees, order)

    if rcond is None:
        rcond = np.asarray(xi[0]).size * np.finfo(float).eps

    # Solve the least squares problem.
    c, resids, rank, s = np.linalg.lstsq(lhs / scl, rhs, rcond)
    c = (c / scl).reshape(orders)

    if full:
        return c, [resids, rank, s, rcond]
    if rank != order:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg, RankWarning)
    return c


def chebvalnd(
    c: ArrayLike,
    *xi: ArrayLike,
) -> NDArray:
    """
    Evaluate a N-D Chebyshev series at points (x1, x2, ..., xn).

    This function returns the values:

    .. math:: p(x1,x2,...,xn) =
            \\sum_{i,j,...,k} c_{i,j,...,k} * T_i(x1) * T_j(x2)*...* T_k(xn)

    The parameters `x1`, `x2`, ...., `xn` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x1`, `x2`, ..., `xn` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than N dimensions, ones are implicitly appended to its
    shape to make it N-D. The shape of the result will be c.shape[3:] +
    x1.shape.

    Parameters
    ----------
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,...,k is contained in ``c[i,j,...,k]``. If `c` has
        dimension greater than N the remaining indices enumerate multiple sets
        of coefficients.
    x1, x2,..., xn : array_like, compatible object
        The N dimensional series is evaluated at the points
        `(x1, x2,...,xn)`, where `x1`, `x2`,..., `xn` must have the same shape.
        If any of `x1`, `x2`, ..., `xn` is a list or tuple, it is first
        converted to an ndarray, otherwise it is left unchanged and if it isn't
        an ndarray it is  treated as a scalar.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    chebval, chebgridnd, chebfitnd
    """
    try:
        xi = np.asarray(xi)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation coordinates have incompatible shapes") from exc
    if len(xi) == 0:
        raise ValueError(
            "expected at least one coordinate"
        )
    chebval = np.polynomial.chebyshev.chebval
    c = chebval(xi[0], c)
    for x in xi[1:]:
        c = chebval(x, c, tensor=False)
    return c


def chebvandernd(
    deg: Sequence[int],
    *xi: ArrayLike,
) -> NDArray:
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points `(x1, x2, ..., xn)`. If `l, m, n` are the given degrees in
    `x1, x2, x3`, then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = T_i(x1)*T_j(x2)*T_k(x3),

    where `0 <= i <= l`, `0 <= j <= m`, and `0 <= k <= n`.  The leading
    indices of `V` index the points `(x, y, z)` and the last index encodes
    the degrees of the Chebyshev polynomials.

    If ``V = chebvandernd([xdeg, ydeg, zdeg], x, y, z)``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and ``np.dot(V, c.flat)`` and ``chebvalnd(c, x, y, z)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of N-D Chebyshev
    series of the same degrees and sample points.

    Parameters
    ----------
    deg : sequence of int
        Sequence of maximum degrees of the form [x1_deg, x2_deg, ...,xn_deg].
    x1, x2, ..., xn : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.

    Returns
    -------
    vander : ndarray
        The shape of the returned matrix is ``x1.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg([1]+1)*...*(deg[n-1]+1)`.  The dtype
        will be the same as the converted `x1`, `x2`, ... `xn`.

    See Also
    --------
    chebvander, chebvalnd, chebfitnd
    """
    def _check_deg(ideg, is_valid, ndim):
        if np.any(is_valid != 1):
            raise ValueError("degrees must be non-negative integers")
        if len(ideg) != ndim:
            msg = 'length of deg must be the same as number of dimensions'
            raise ValueError(msg)

    ideg = [int(d) for d in deg]
    is_valid = np.array([di == d and di >= 0 for di, d in zip(ideg, deg)])
    ndim = len(xi)
    _check_deg(ideg, is_valid, ndim)

    xi = np.asarray(xi, dtype=float)
    chebvander = np.polynomial.chebyshev.chebvander
    shape0 = xi[0].shape
    s0 = (1,) * ndim
    vxi = [chebvander(x, d).reshape(shape0 + s0[:i] + (-1,) + s0[i + 1::])
           for i, (d, x) in enumerate(zip(ideg, xi))]

    v = reduce(np.multiply, vxi)

    return v.reshape(v.shape[:-ndim] + (-1,))


def chebgridnd(
    c: ArrayLike,
    *xi: ArrayLike,
) -> NDArray:
    """
    Evaluate a N-D Chebyshev series on the Cartesian product of x1, x2,..., xn.

    This function returns the values:

    .. math:: p(a,b,...) = \\sum_{i,j,...} c_{i,j,...} * T_i(a) * T_j(b) *  ...

    where the points `(a, b, ...)` consist of all points formed by taking
    `a` from `x1`, `b` from `x2`, and so on. The resulting points form
    a grid with `x1` in the first dimension, `x2` in the second, and so on.

    The parameters `x1`, `x2`, ... and `xn` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars. In
    either case, either `x1`, `x2`,... and `xn` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than N dimensions, ones are implicitly appended to
    its shape to make it N-D. The shape of the result will be c.shape[3:] +
    x1.shape + x2.shape + ... + xn.shape

    Parameters
    ----------
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.
    x1, x2,..., xn : ndarray, compatible object
        1-D arrays representing the coordinates of a grid.
        The N dimensional series is evaluated at the points in the
        Cartesian product of `x1`, `x2`, ... and `xn`.  If `xi`, is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.

    Returns
    -------
    values : ndarray, compatible object
        The values of the N dimensional polynomial at points in the Cartesian
        product of `x1`, `x2`, ... and `xn`.

    Examples
    --------
    >>> c = np.zeros((3, 3))
    >>> c[0, 0] = 1

    >>> x = np.linspace(-1, 1, 4)
    >>> y = np.linspace(-1, 1, 5)

    >>> np.allclose(
    ...    chebgridnd(c, x, y),
    ...    np.ones((4, 5))
    ... )
    True


    See Also
    --------
    chebval, chebvalnd, chebfitnd
    """
    for x in xi:
        c = chebval(x, c)
    return c


if __name__ == '__main__':
    from timeit import default_timer as timer
    import doctest
    print("Running docstests .....")

    t0 = timer()
    result = doctest.testmod(
        optionflags=(doctest.NORMALIZE_WHITESPACE
                     | doctest.ELLIPSIS
        )
    )
    dt = timer() - t0

    print(
        f"Attempted: {result.attempted}, "
        f"Failed: {result.failed}, "
        f"Elapsed: {dt:.3f}s"
    )
