from dataclasses import FrozenInstanceError

import pytest
import numpy as np
from numpy.polynomial import Polynomial
from approxkit.pade import PadeApproximation, padefit, padefitlsq


def test_padeapproximation_properties():
    p = PadeApproximation(
        Polynomial([1, 2]),
        Polynomial([1, -1]),
        domain=(0, 1),
        max_error=0.5,
    )

    assert p.has_domain
    assert p.has_error_estimate
    assert p.num is p.numerator
    assert p.den is p.denominator

def test_padeapproximation_default_metadata():
    p = PadeApproximation(
        Polynomial([1]),
        Polynomial([1]),
    )

    assert not p.has_domain
    assert not p.has_error_estimate


def test_padeapproximation_call():
    p = PadeApproximation(
        Polynomial([1, 1]),
        Polynomial([1]),
    )

    x = np.array([0, 1, 2])

    assert np.allclose(p(x), 1 + x)


def test_padeapproximation_poles_and_zeros():
    p = PadeApproximation(
        Polynomial([-1, 1]),  # x - 1
        Polynomial([2, 1]),   # x + 2
    )

    assert np.allclose(p.zeros, [1])
    assert np.allclose(p.poles, [-2])


def test_padeapproximation_is_frozen():
    p = PadeApproximation(
        Polynomial([1]),
        Polynomial([1]),
    )

    with pytest.raises(FrozenInstanceError):
        p.domain = (0, 1)  # type: ignore[misc]


def test_padefit_rejects_negative_n():
    with pytest.raises(ValueError, match="non-negative"):
        padefit([1, 2, 3], m=1, n=-1)


def test_padefit():
    cof = np.array([1, 1, 1/2, 1/6, 1/24])
    p = padefit(cof)

    t = np.arange(0, 2, 0.1)

    assert np.max(np.abs(p(t) - np.exp(t))) < 0.3


def test_padefit_rejects_invalid_m():
    with pytest.raises(ValueError):
        padefit([1, 2, 3], m=3)


def test_padefit_requires_enough_coefficients():
    with pytest.raises(ValueError):
        padefit([1, 2, 3], m=2, n=1)


def test_padefit_default_degrees():
    p = padefit([1, 1, 0.5, 1/6, 1/24])

    assert isinstance(p, PadeApproximation)
    assert p.num.degree() == 3
    assert p.den.degree() == 1



def test_padefit_exact_rational_function():
    # 1 / (1 - x)
    coeffs = [1] * 6

    p = padefit(coeffs, m=0, n=1)

    x = np.linspace(-0.5, 0.5, 20)

    assert np.allclose(
        p(x),
        1 / (1 - x),
        atol=1e-12,
    )


def test_padefitlsq_returns_metadata():
    p = padefitlsq(np.exp, 2, 2, a=0, b=1)

    assert p.domain == (0, 1)
    assert p.max_error is not None


def test_padefitlsq_accepts_sample_values():
    x = np.linspace(0, 1, 100)
    y = np.exp(x)

    p = padefitlsq(y, 2, 2, x=x)

    assert isinstance(p, PadeApproximation)


def test_padefitlsq_requires_matching_lengths():
    x = np.linspace(0, 1, 20)
    y = np.ones(10)

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        padefitlsq(y, 2, 2, x=x)


def test_padefitlsq_warns_for_too_few_points():
    x = np.linspace(0, 1, 10)
    y = np.exp(x)

    with pytest.warns(
        UserWarning,
        match="sample points",
    ):
        padefitlsq(y, 2, 2, x=x)


def test_padefitlsq_callable_length_mismatch():
    x = np.linspace(0, 1, 20)

    def fun(x):
        return np.ones(10)

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        padefitlsq(fun, 2, 2, x=x)
