import pytest
import numpy as np
from approxkit.chebyshev import (
    _check_domain,
    chebvandernd,
    chebvalnd,
    chebyshev_nodes,
    chebfit_dct,
    chebfitnd
)


def test_check_domain_tuple_pairs():
    domain = _check_domain(
        [(0, 1), (2, 3)],
        ndim=2,
    )

    assert np.array_equal(
        domain,
        np.array([[0, 1], [2, 3]])
    )


def test_check_domain_rejects_odd_length_input():
    with pytest.raises(
        ValueError,
        match=r"domain must contain pairs \(a, b\)",
    ):
        _check_domain([0, 1, 2], ndim=1)


def test_check_domain_requires_one_interval_per_dimension():
    with pytest.raises(
        ValueError,
        match="domain must contain one interval per dimension",
    ):
        _check_domain([(0, 1)], ndim=2)


def test_check_domain_normalizes_shape():
    domain = _check_domain([0, 1, 2, 3], ndim=2)

    expected = np.array([
        [0, 1],
        [2, 3],
    ])

    assert np.array_equal(domain, expected)


def test_chebyshev_nodes_invalid_degree():
    with pytest.raises(ValueError):
        chebyshev_nodes(0)

    with pytest.raises(ValueError):
        chebyshev_nodes(-1)


def test_chebfit_dct_non_square_grid():
    c = chebfit_dct(
        lambda x, y: x + y,
        n=(3, 5),
        indexing="xy"
    )
    expected = np.zeros((3, 5))
    expected[0, 1] = 1.0
    expected[1, 0] = 1.0

    assert np.allclose(c, expected)


def test_chebfit_dct_rejects_wrong_callable_shape():
    with pytest.raises(
        ValueError,
        match="expected function values with shape",
    ):
        chebfit_dct(
            lambda x, y: np.zeros((2, 2)),
            n=(5, 7),
        )


def test_chebfit_dct_basis_order_2d():
    f = lambda x, y: x + 2 * y

    c = chebfit_dct(f, n=(9, 9))

    expected = np.zeros_like(c)
    expected[1, 0] = 1.0
    expected[0, 1] = 2.0

    assert np.allclose(c, expected, atol=1e-14)


def test_chebfit_dct_basis_order_3d():
    f = lambda x, y, z: x + 2 * y + 3 * z

    c = chebfit_dct(f, n=(9, 9, 9))

    expected = np.zeros_like(c)
    expected[1, 0, 0] = 1.0
    expected[0, 1, 0] = 2.0
    expected[0, 0, 1] = 3.0

    assert np.allclose(c, expected, atol=1e-14)


def test_chebfit_dct_reconstructs_linear_function():
    f = lambda x, y: x + 2 * y

    c = chebfit_dct(f, n=(9, 9))

    x = chebyshev_nodes(9)
    y = chebyshev_nodes(9)

    X, Y = np.meshgrid(x, y, indexing="ij")

    assert np.allclose(
        chebvalnd(c, X, Y),
        f(X, Y),
    )


def test_chebfit_dct_xy_vs_ij():
    f = lambda x, y: x + 2 * y

    x = chebyshev_nodes(9)
    y = chebyshev_nodes(9)

    Xij, Yij = np.meshgrid(x, y, indexing="ij")
    Xxy, Yxy = np.meshgrid(x, y, indexing="xy")

    c_ij = chebfit_dct(f, n=(9, 9), indexing="ij")
    c_xy = chebfit_dct(f, n=(9, 9), indexing="xy")

    assert np.allclose(
        chebvalnd(c_ij, Xij, Yij),
        f(Xij, Yij),
    )

    assert np.allclose(
        chebvalnd(c_xy, Xxy, Yxy),
        f(Xxy, Yxy),
    )


def test_chebfitnd_basis_order_2d():
    x = chebyshev_nodes(9)
    y = chebyshev_nodes(9)

    X, Y = np.meshgrid(x, y, indexing="ij")

    f = X + 2 * Y

    c = chebfitnd((X, Y), f, deg=[1, 1])

    expected = np.zeros((2, 2))
    expected[1, 0] = 1.0
    expected[0, 1] = 2.0

    assert np.allclose(c, expected, atol=1e-14)


def test_chebfitnd_reconstructs_linear_function_2d():
    x = chebyshev_nodes(9)
    y = chebyshev_nodes(9)

    X, Y = np.meshgrid(x, y, indexing="ij")

    f = X + 2 * Y

    c = chebfitnd((X, Y), f, deg=[1, 1])

    assert np.allclose(
        chebvalnd(c, X, Y),
        f,
    )


def test_chebfitnd_basis_order_3d():
    x = chebyshev_nodes(5)
    y = chebyshev_nodes(6)
    z = chebyshev_nodes(7)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    c = chebfitnd(
        (X, Y, Z),
        X + 2 * Y + 3 * Z,
        deg=[1, 1, 1],
    )

    expected = np.zeros((2, 2, 2))
    expected[1, 0, 0] = 1.0
    expected[0, 1, 0] = 2.0
    expected[0, 0, 1] = 3.0

    assert np.allclose(c, expected)


def test_chebfitnd_reconstructs_linear_function_3d():
    x = chebyshev_nodes(5)
    y = chebyshev_nodes(6)
    z = chebyshev_nodes(7)

    X, Y, Z = np.meshgrid(
        x, y, z,
        indexing="ij",
    )

    f = X + 2 * Y + 3 * Z

    c = chebfitnd((X, Y, Z), f, deg=[1, 1, 1])

    assert np.allclose(
        chebvalnd(c, X, Y, Z),
        f,
    )


def test_chebvalnd_requires_at_least_one_coordinate():
    c = [1.0]

    with pytest.raises(
        ValueError,
        match="at least one coordinate",
    ):
        chebvalnd(c)


def test_chebvalnd_basis_order():
    x = 0.3
    y = 0.4

    c = np.zeros((2, 2))
    c[1, 0] = 1

    assert np.allclose(chebvalnd(c, x, y), x)

    c = np.zeros((2, 2))
    c[0, 1] = 1

    assert np.allclose(chebvalnd(c, x, y), y)


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

    coef = np.arange(12, dtype=float).reshape(3, 4)

    V = chebvandernd([2, 3], X, Y)

    lhs = np.dot(V, coef.ravel())
    rhs = chebvalnd(coef, X, Y)

    assert np.allclose(lhs, rhs)


def test_chebvandernd_preserves_input_shape():
    """Test Nontrivial 2D shape"""
    rng = np.random.default_rng(1234)
    X = rng.uniform(-1, 1, (3, 4))
    Y = rng.uniform(-1, 1, (3, 4))

    V = chebvandernd([2, 3], X, Y)

    assert V.shape == (3, 4, 12)


@pytest.mark.parametrize(
    ("deg", "expected"),
    [
        ([0], 1),
        ([1], 2),
        ([4], 5),
        ([1, 1], 4),
        ([2, 3], 12),
        ([2, 1, 3], 24),
    ],
)
def test_chebvandernd_output_size(deg, expected):
    shape = (3, 4)

    xi = tuple(np.zeros(shape) for _ in deg)

    V = chebvandernd(deg, *xi)

    assert V.shape == shape + (expected,)


def test_chebvandernd_zero_degree():
    x = np.linspace(-1, 1, 5)

    V = chebvandernd([0], x)

    assert np.allclose(V, np.ones((5, 1)))


def test_chebvandernd_rejects_negative_degree():
    x = np.linspace(-1, 1, 5)

    with pytest.raises(ValueError):
        chebvandernd([-1], x)


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
    """Test scalar inputs"""
    V = chebvandernd([2, 3], 0.25, -0.5)

    assert V.shape == (12,)


def test_chebvandernd_rejects_mismatched_shapes():
    """Test that mismatched shapes fail"""
    x = np.zeros((3, 4))
    y = np.zeros((3, 5))

    with pytest.raises(ValueError):
        chebvandernd([2, 2], x, y)