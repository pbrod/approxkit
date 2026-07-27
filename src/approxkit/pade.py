import warnings
from dataclasses import dataclass
import numpy as np

from numpy.polynomial import Polynomial
from numpy.polynomial.chebyshev import chebpts1, chebpts2
from numpy.typing import ArrayLike, NDArray

from mpmath import pade

from approxkit.utils import map_to_interval


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

    def __call__(self, x: ArrayLike) -> NDArray:
        return self.numerator(x) / self.denominator(x)

    @property
    def has_domain(self) -> bool:
        return self.domain is not None

    @property
    def has_error_estimate(self) -> bool:
        return self.max_error is not None

    @property
    def num(self)-> Polynomial:
        """Alias for numerator."""
        return self.numerator

    @property
    def den(self)-> Polynomial:
        """Alias for denominator."""
        return self.denominator

    @property
    def poles(self)-> NDArray:
        """Roots of the denominator polynomial (poles)."""
        return self.denominator.roots()

    @property
    def zeros(self)-> NDArray:
        """Roots of the numerator polynomial (zeros)."""
        return self.numerator.roots()


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
    c = np.asarray(c)
    if m is None:
        # Use a near-diagonal Pade approximant.
        # If the total degree is odd, assign the extra degree
        # to the numerator.
        m = (len(c) - 1) // 2 + 1
    if not (0 <= m < len(c)):
        raise ValueError(f"expected 0 <= m < {len(c)}")

    if n is None:
        n = len(c) - 1 - m
    elif n < 0:
        raise ValueError('n must be non-negative')

    if m + n + 1 > len(c):
        raise ValueError(f'require m + n + 1 <= {len(c)}')

    # mpmath.pade expects (coefficients, numerator_degree,
    # denominator_degree).
    num, den = pade(c, m, n)
    return PadeApproximation(Polynomial(num), Polynomial(den))


def padefitlsq(
    fun,
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
    def _points(npt, end_points):
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

    def _check_size(fs, x, npt):
        if len(fs) != len(x):
            raise ValueError(
                "x and function values must have the same length"
            )
        if len(fs) < npt:
            warnings.warn(
                f'expected at least {npt} sample points',
                stacklevel=2)

    def _init(fun, a, b, x, end_points, npt):
        if x is None:
            x = map_to_interval(_points(npt, end_points), a, b)
        if callable(fun):
            fs = fun(x)
        else:
            fs = fun
        _check_size(fs, x, npt)
        return x, fs

    def _cond_plot1(trace, x, fs):
        if trace:
            import matplotlib.pyplot as plt
            plt.plot(x, fs, '+')

    def _cond_plot2(x, fs, ys, ix, devmax):
        if trace:
            import matplotlib.pyplot as plt
            print(f"Iteration={ix}, max error={devmax:g}")
            plt.plot(x, fs, x, ys)

    NFAC = 8
    MAXIT = 5

    smallest_devmax = np.inf
    ncof = m + n + 1
    # Number of points where function is evaluated, i.e. fineness of mesh
    npt_min = NFAC * ncof

    x, fs = _init(fun, a, b, x, end_points, npt_min)
    npt = len(x)
    _cond_plot1(trace, x, fs)

    wt = np.ones(npt)
    ee = np.ones(npt)
    mad = 0

    u = np.zeros((npt, ncof))
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

        #[u1, w, v] = np.linalg.svd(u, full_matrices=False)
        #cof = np.dot(np.where(w == 0, 0.0, np.dot(bb, u1) / w), v)
        cof, *_ = np.linalg.lstsq(u, bb, rcond=None)

        # Tabulate the deviations and revise the weights
        num = Polynomial(cof[: m + 1])
        den = Polynomial(np.r_[1.0, cof[m + 1 :]])
        ee = num(x) / den(x) - fs
        # ee = polyval(cof[m::-1], x) / \
        #     polyval(cof[ncof:m:-1].tolist() + [1, ], x) - fs

        wt = np.abs(ee)
        devmax = np.max(wt)
        mad = wt.mean()  # mean absolute deviation

        # Save only the best coefficients found
        if devmax <= smallest_devmax:
            smallest_devmax = devmax
            c1 = cof[: m + 1]
            c2 = np.r_[1.0, cof[m + 1 :]]

        _cond_plot2(x, fs, ee + fs, ix, devmax)
    return PadeApproximation(
        Polynomial(c1),
        Polynomial(c2),
        domain=(a, b),
        max_error=smallest_devmax,
    )



if __name__ == '__main__':
    from approxkit.testing import test_docstrings
    test_docstrings()
