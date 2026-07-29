from typing import cast, Callable
import warnings
from dataclasses import dataclass
import numpy as np

from numpy.polynomial import Polynomial
from numpy.polynomial.chebyshev import chebpts1, chebpts2
from numpy.typing import ArrayLike, NDArray

from mpmath import pade

from approxkit.utils import map_to_interval

__all__ = [
    "PadeApproximation",
    "padefit",
    "padefitlsq",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PadeApproximation:
    """
    Padé approximation.

    Parameters
    ----------
    numerator : Polynomial
        Numerator polynomial.

    denominator : Polynomial
        Denominator polynomial.

    domain : tuple[float, float] | None
        Interval used when fitting the approximation.

    max_error : float | None
        Maximum absolute fitting error.
    """

    numerator: Polynomial
    denominator: Polynomial
    domain: tuple[float, float] | None = None
    max_error: float | None = None

    def __call__(self, x: ArrayLike) -> FloatArray:
        x_arr: FloatArray = np.asarray(x, dtype=float)
        return cast(
            FloatArray,
            self.numerator(x_arr) / self.denominator(x_arr),
        )

    def __repr__(self) -> str:
        attrs = [
            f"m={self.numerator.degree()}",
            f"n={self.denominator.degree()}",
        ]
        if self.domain is not None:
            attrs.append(f"domain={self.domain}")
        if self.max_error is not None:
            attrs.append(f"max_error={self.max_error:.3e}")
        return f"PadeApproximation({', '.join(attrs)})"

    @property
    def has_domain(self) -> bool:
        return self.domain is not None

    @property
    def has_error_estimate(self) -> bool:
        return self.max_error is not None

    @property
    def num(self) -> Polynomial:
        """Alias for numerator."""
        return self.numerator

    @property
    def den(self) -> Polynomial:
        """Alias for denominator."""
        return self.denominator

    @property
    def poles(self) -> NDArray[np.complex128]:
        """Roots of the denominator polynomial (poles)."""
        return cast(
            NDArray[np.complex128],
            self.denominator.roots(),
        )

    @property
    def zeros(self) -> NDArray[np.complex128]:
        """Roots of the numerator polynomial (zeros)."""
        return cast(
            NDArray[np.complex128],
            self.numerator.roots(),
        )

    @property
    def degrees(self) -> tuple[int, int]:
        """Return (numerator_degree, denominator_degree)."""
        return (
            self.numerator.degree(),
            self.denominator.degree(),
        )


def padefit(
    c: ArrayLike,
    m: int | None = None,
    n: int | None = None,
) -> PadeApproximation:
    """
    Compute a Padé approximant from Taylor coefficients.

    Parameters
    ----------
    c : array_like
        Taylor coefficients ordered from low degree to high degree:

            c0 + c1*x + c2*x**2 + ...

    m : int, optional
        Degree of numerator polynomial. (Default ``(len(c)-1)//2 + 1``)

    n : int, optional
        Degree of denominator polynomial. (Default ``len(c)-1-m``)

    Returns
    -------
    PadeApproximation
        Padé approximation represented by numerator and denominator
        polynomials.


    If the function is well approximated by
              m+n+1
       f(x) = sum c(i)*x^i
              i=0

    then the pade approximation is given by
               m
              sum c1(i)*x^i
              i=0
    f(x) = ------------------------
              n
              sum c2(i)*x^i
              i=0


    Examples
    --------
    Pade approximation to exp(x)
    >>> from numpy.polynomial import Polynomial
    >>> import scipy.special as sp

    >>> c = Polynomial(1./sp.gamma(np.arange(1, 8)))
    >>> p = padefit(c.coef)

    >>> np.allclose(p.numerator.coef,
    ...             [1.0, 0.66666667, 0.2, 0.03333333, 0.00277778])
    True
    >>> np.allclose(p.denominator.coef,
    ...             [1.0, -0.33333333, 0.03333333])
    True

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 4)
    >>> h0 = plt.plot(x, np.exp(x), 'r.', label='exp')
    >>> h1 = plt.plot(x, p(x), 'g', label=f'pade {len(p.num.coef)} {len(p.den.coef)}')
    >>> h2 = plt.plot(x, c(x), label=f'taylor {len(c.coef)}')
    >>> h3 = plt.legend()
    >>> plt.close()

    See also
    --------
    scipy.interpolate.pade

    """
    coef: FloatArray = np.asarray(c, dtype=float)
    if m is None:
        # Use a near-diagonal Pade approximant.
        # If the total degree is odd, assign the extra degree
        # to the numerator.
        m = (len(coef) - 1) // 2 + 1
    if not (0 <= m < len(coef)):
        raise ValueError(f"expected 0 <= m < {len(coef)}")

    if n is None:
        n = len(coef) - 1 - m
    elif n < 0:
        raise ValueError("n must be non-negative")

    if m + n + 1 > len(coef):
        raise ValueError(f"require m + n + 1 <= {len(coef)}")

    # mpmath.pade expects (coefficients, numerator_degree,
    # denominator_degree).
    num, den = pade(coef, m, n)
    return PadeApproximation(Polynomial(num), Polynomial(den))


def padefitlsq(
    fun: Callable[[ArrayLike], ArrayLike] | ArrayLike,
    m: int,
    n: int,
    a: float = -1,
    b: float = 1,
    trace: bool = False,
    x: ArrayLike | None = None,
    end_points: bool = True,
) -> PadeApproximation:
    """
    Rational polynomial fitting. A minimax solution by least squares.

    Parameters
    ----------
    fun : callable or a vector
        function to approximate. If fun and x are supplied as vectors the
        vectors must satisfy: len(fun)=len(x) > (m+n+1)*8.
    m, n : int
        Degrees of the numerator and denominator polynomials.
    a, b : real scalars
        evaluation limits, (default a=-1,b=1)
    trace : bool
        if True plot values and fitted function.
    end_points : bool
        If True, use Chebyshev points of the second kind.
        Otherwise, use Chebyshev points of the first kind.
        Note set end_points to True if there are singularities close to the
        endpoints.

    Returns
    -------
    PadeApproximation
        Padé approximation represented by numerator and denominator
        polynomials.

    Notes
    -----
    The pade approximation is given by
               m
              sum c1[i]*x**i
              i=0
    f(x) = ------------------------
               n
              sum c2[i]*x**i
              i=0

    Examples
    --------

    Pade approximation to exp(x) between 0 and 2

    >>> p = padefitlsq(np.exp, 3, 3, 0, 2)
    >>> np.allclose(p.num.coef, [0.99999962, 0.55284547, 0.128842, 0.01443847])
    True
    >>> np.allclose(p.den.coef, [1.0, -0.44716929, 0.07610473, -0.0049658])
    True

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0,4)
    >>> y = p(x)
    >>> h0 = plt.plot(x, y, 'g', label=f'pade {len(p.num.coef)} {len(p.den.coef)}')
    >>> h1 = plt.plot(x, np.exp(x), 'r', label='exp')
    >>> plt.close()


    See also
    --------
    padefit

    References
    ----------
    William H. Press, Saul Teukolsky,
    William T. Wetterling and Brian P. Flannery (1997)
    "Numerical recipes in Fortran 77", Vol. 1, pp 197-20
    """

    def _points(npt: int, end_points: bool) -> FloatArray:
        if end_points:
            # Use the location of the local extreme values of
            # the Chebychev polynomial of the first kind of degree NPT-1.
            # return chebextr(npt - 1)
            # This equals the Chebyshev points of the second kind.
            return chebpts2(npt)
        # Use the roots of the Chebychev polynomial of the first kind of
        # degree NPT. Note this is useful if there are singularities close
        # to the endpoints.
        # return chebroot(npt, kind=1)
        # This equals the Chebyshev points of the first kind.
        return chebpts1(npt)

    def _check_size(fs: FloatArray, x: FloatArray, npt: int) -> None:
        if len(fs) != len(x):
            raise ValueError("x and function values must have the same length")
        if len(fs) < npt:
            warnings.warn(f"expected at least {npt} sample points", stacklevel=2)

    def _init(
        fun: Callable[[ArrayLike], ArrayLike] | ArrayLike,
        a: float,
        b: float,
        x: ArrayLike | None,
        end_points: bool,
        npt: int,
    ) -> tuple[FloatArray, FloatArray]:
        if x is None:
            x_arr = map_to_interval(_points(npt, end_points), a, b)
        else:
            x_arr = np.asarray(x, dtype=float)

        fs_arr = (
            np.asarray(fun(x_arr), dtype=float)
            if callable(fun)
            else np.asarray(fun, dtype=float)
        )
        _check_size(fs_arr, x_arr, npt)
        return x_arr, fs_arr

    def _cond_plot1(trace: bool, x: FloatArray, fs: FloatArray) -> None:
        if trace:
            import matplotlib.pyplot as plt

            plt.plot(x, fs, "+")

    def _cond_plot2(
        x: FloatArray,
        fs: FloatArray,
        y_fit: FloatArray,
        ix: int,
        devmax: float,
    ) -> None:
        if trace:
            import matplotlib.pyplot as plt

            print(f"Iteration={ix}, max error={devmax:g}")
            plt.plot(x, fs, x, y_fit)

    NFAC = 8
    MAXIT = 5

    smallest_devmax: float = np.inf
    ncof = m + n + 1
    # Number of points where function is evaluated, i.e. fineness of mesh
    npt_min = NFAC * ncof

    x, fs = _init(fun, a, b, x, end_points, npt_min)
    npt = len(x)
    _cond_plot1(trace, x, fs)

    wt: FloatArray = np.ones(npt, dtype=float)
    ee: FloatArray = np.ones(npt, dtype=float)
    mad: float = 0.0
    u: FloatArray = np.zeros((npt, ncof), dtype=float)
    best_num: Polynomial | None = None
    best_den: Polynomial | None = None

    for ix in range(MAXIT):
        # Set up design matrix for least squares fit.
        pow1 = wt
        bb = pow1 * (fs + abs(mad) * np.sign(ee))

        for jx in range(m + 1):
            u[:, jx] = pow1
            pow1 = pow1 * x

        pow1 = -bb
        for jx in range(m + 1, ncof):
            pow1 = pow1 * x
            u[:, jx] = pow1

        cof, *_ = np.linalg.lstsq(u, bb, rcond=None)

        # Tabulate the deviations and revise the weights
        num = Polynomial(cof[: m + 1])
        den = Polynomial(np.r_[1.0, cof[m + 1 :]])
        ee = num(x) / den(x) - fs

        wt = np.abs(ee)
        devmax: float = np.max(wt)
        mad = wt.mean()  # mean absolute deviation

        # Save only the best coefficients found
        if devmax <= smallest_devmax:
            smallest_devmax = devmax
            best_num = num
            best_den = den

        _cond_plot2(x, fs, ee + fs, ix, devmax)

    assert best_num is not None
    assert best_den is not None

    return PadeApproximation(
        best_num,
        best_den,
        domain=(a, b),
        max_error=smallest_devmax,
    )


if __name__ == "__main__":
    from approxkit.testing import test_docstrings

    test_docstrings()
