import pytest
import numpy as np
from approxkit import chebvandernd, chebvalnd


def test_chebvandernd_matches_chebvander():
    """Test 1D consistency with NumPy's chebvander"""
    x = np.linspace(-1, 1, 5)

    v1 = np.polynomial.chebyshev.chebvander(x, 4)
    v2 = chebvandernd([4], x)

    assert np.allclose(v1, v2)


def test_chebvandernd_meshgrid_matches_chebvalnd():
    """Test 2D meshgrid consistency with chebvalnd

    This validates:

    coefficient ordering
    flattening convention
    tensor-product construction
    """
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 5)

    X, Y = np.meshgrid(x, y, indexing="ij")

    coef = np.arange(12).reshape(3, 4)

    V = chebvandernd([2, 3], X, Y)

    lhs = np.dot(V, coef.ravel())
    rhs = chebvalnd(coef, X, Y)

    assert np.allclose(lhs, rhs)


def test_chebvandernd_preserves_input_shape():
    """Test Nontrivial 2D shape"""
    X = np.random.default_rng(1234).uniform(-1, 1, (3, 4))
    Y = np.random.default_rng(5678).uniform(-1, 1, (3, 4))

    V = chebvandernd([2, 3], X, Y)

    assert V.shape == (3, 4, 12)


def test_chebvandernd_3d_matches_chebvalnd():
    """Test 3D consistency

    This catches axis-order mistakes.
    """
    rng = np.random.default_rng(123)

    X = rng.uniform(-1, 1, (2, 3, 4))
    Y = rng.uniform(-1, 1, (2, 3, 4))
    Z = rng.uniform(-1, 1, (2, 3, 4))

    coef = rng.normal(size=(3, 2, 4))

    V = chebvandernd([2, 1, 3], X, Y, Z)

    lhs = np.dot(V, coef.ravel())
    rhs = chebvalnd(coef, X, Y, Z)

    assert np.allclose(lhs, rhs)


def test_chebvandernd_scalar_inputs():
    """Test scalar inpurts"""
    V = chebvandernd([2, 3], 0.25, -0.5)

    assert V.shape == (12,)


def test_chebvandernd_rejects_mismatched_shapes():
    """Test that mismatched shapes fail"""
    x = np.zeros((3, 4))
    y = np.zeros((3, 5))

    with pytest.raises(ValueError):
        chebvandernd([2, 2], x, y)