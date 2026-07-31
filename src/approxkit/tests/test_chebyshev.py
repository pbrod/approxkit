import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebval, chebvander

from approxkit.chebyshev import (
    ChebyshevND,
    _check_domain,
    chebfit1d,
    chebfit_dct,
    chebfitnd,
    chebgridnd,
    chebvalnd,
    chebvandernd,
    chebyshev_lobatto_nodes,
    chebyshev_nodes,
    select_degree_aic,
)
from approxkit.utils import map_to_interval


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
def test_chebyshev_lobatto_nodes_formula(n):
    expected = -np.cos(np.pi * np.arange(n + 1) / n)

    assert np.allclose(
        chebyshev_lobatto_nodes(n),
        expected,
    )


def test_chebyshev_lobatto_nodes_zero():
    assert np.array_equal(
        chebyshev_lobatto_nodes(0),
        np.array([0.0]),
    )


@pytest.mark.parametrize("n", [-1, -2])
def test_chebyshev_lobatto_nodes_negative_degree(n):
    with pytest.raises(ValueError, match="n must be non-negative"):
        chebyshev_lobatto_nodes(n)


def test_check_domain_rejects_zero_width_interval():
    with pytest.raises(
        ValueError,
        match="nonzero width",
    ):
        _check_domain([(1, 1)], 1)


def test_check_domain_tuple_pairs():
    domain = _check_domain(
        [(0, 1), (2, 3)],
        ndim=2,
    )

    assert np.array_equal(domain, np.array([[0, 1], [2, 3]]))


def test_check_domain_rejects_odd_length_input():
    with pytest.raises(
        ValueError,
        match="domain must contain pairs",
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

    expected = np.array(
        [
            [0, 1],
            [2, 3],
        ]
    )

    assert np.array_equal(domain, expected)


def test_chebyshev_nodes_invalid_degree():
    with pytest.raises(ValueError):
        chebyshev_nodes(0)

    with pytest.raises(ValueError):
        chebyshev_nodes(-1)


def test_chebfit1d_domain():
    x = np.linspace(0, 2, 50)
    y = np.exp(x)

    p = chebfit1d(
        x,
        y,
        deg=8,
        domain=(0, 2),
    )

    assert np.allclose(
        p(x),
        y,
        atol=1e-2,
    )


def test_chebfit1d_exact_polynomial():
    x = np.linspace(-1, 1, 50)
    y = 1 + 2 * x + 3 * x**2

    p = chebfit1d(x, y, deg=2)

    assert np.allclose(p(x), y)


def test_chebfit_dct_non_square_grid():
    c = chebfit_dct(lambda x, y: x + y, n=(3, 5), indexing="xy")
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


def test_chebfit_dct_accepts_scalar_callable():
    c = chebfit_dct(lambda x: 1.0, n=9)

    expected = np.zeros(9)
    expected[0] = 1.0

    assert np.allclose(c, expected)


def test_chebfit_dct_accepts_scalar_callable_2d():
    c = chebfit_dct(
        lambda x, y: 1.0,
        n=(5, 7),
    )

    expected = np.zeros((5, 7))
    expected[0, 0] = 1.0

    assert np.allclose(c, expected)


def test_chebfit_dct_accepts_scalar_callable_3d():
    c = chebfit_dct(
        lambda x, y, z: 1.0,
        n=(3, 4, 5),
    )

    expected = np.zeros((3, 4, 5))
    expected[0, 0, 0] = 1.0

    assert np.allclose(c, expected)


def test_chebfit_dct_basis_order_2d():
    def f(x, y):
        return x + 2 * y

    c = chebfit_dct(f, n=(9, 9))

    expected = np.zeros_like(c)
    expected[1, 0] = 1.0
    expected[0, 1] = 2.0

    assert np.allclose(c, expected, atol=1e-14)


def test_chebfit_dct_basis_order_3d():

    c = chebfit_dct(
        lambda x, y, z: x + 2 * y + 3 * z,
        n=(9, 9, 9),
    )

    expected = np.zeros_like(c)
    expected[1, 0, 0] = 1.0
    expected[0, 1, 0] = 2.0
    expected[0, 0, 1] = 3.0

    assert np.allclose(c, expected, atol=1e-14)


def test_chebfit_dct_reconstructs_linear_function():
    def f(x, y):
        return x + 2 * y

    c = chebfit_dct(f, n=(9, 9))

    x = chebyshev_nodes(9)
    y = chebyshev_nodes(9)

    X, Y = np.meshgrid(x, y, indexing="ij")

    assert np.allclose(
        chebvalnd(c, X, Y),
        f(X, Y),
    )


def test_chebfit_dct_xy_vs_ij():
    def f(x, y):
        return x + 2 * y

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


def test_chebfit_dct_warns_for_large_n():
    with pytest.warns(
        UserWarning,
        match="n > 50",
    ):
        chebfit_dct(np.exp, n=51)


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


def test_chebfitnd_coordinate_dimension_mismatch():
    x = np.arange(5)
    y = np.arange(5)

    z = np.zeros(5)

    with pytest.raises(
        ValueError,
        match="expected .*dimensional arrays",
    ):
        chebfitnd((x, y), z, deg=[2, 2])


def test_chebfitnd_empty_coordinate():
    x = np.array([])
    z = np.array([])

    with pytest.raises(
        ValueError,
        match="non-empty vector",
    ):
        chebfitnd((x,), z, deg=[1])


def test_chebfitnd_weight_length_mismatch():
    x = np.linspace(-1, 1, 5)
    y = x**2

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        chebfitnd(
            (x,),
            y,
            deg=[2],
            w=np.ones(4),
        )


def test_chebfitnd_weights_influence_fit():
    x = np.array([-1.0, 0.0, 1.0])

    y = np.array([0.0, 100.0, 0.0])

    coef_unweighted = chebfitnd(
        (x,),
        y,
        deg=[0],
    )

    coef_weighted = chebfitnd(
        (x,),
        y,
        deg=[0],
        w=np.array([1.0, 1000.0, 1.0]),
    )

    p_unweighted = chebvalnd(coef_unweighted, 0.0)
    p_weighted = chebvalnd(coef_weighted, 0.0)

    assert abs(p_weighted - 100) < abs(p_unweighted - 100)


def test_chebfitnd_constant_fit_with_weights():
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([0.0, 100.0, 0.0])

    w = np.array([1.0, 100.0, 1.0])

    coef = chebfitnd(
        (x,),
        y,
        deg=[0],
        w=w,
    )

    expected = np.sum(w**2 * y) / np.sum(w**2)

    assert np.allclose(
        coef[0],
        expected,
    )


def test_chebfitnd_reconstructs_linear_function_3d():
    x = chebyshev_nodes(5)
    y = chebyshev_nodes(6)
    z = chebyshev_nodes(7)

    X, Y, Z = np.meshgrid(
        x,
        y,
        z,
        indexing="ij",
    )

    f = X + 2 * Y + 3 * Z

    c = chebfitnd((X, Y, Z), f, deg=[1, 1, 1])

    assert np.allclose(
        chebvalnd(c, X, Y, Z),
        f,
    )


def test_chebfitnd_rejects_non_integer_degree():
    x = chebyshev_nodes(5)
    X = x

    with pytest.raises(
        ValueError,
        match="degrees must be non-negative integers",
    ):
        chebfitnd((X,), X, deg=[1.5])  # type: ignore[list-item]


def test_chebgridnd_constant():
    c = np.zeros((3, 3))
    c[0, 0] = 1.0

    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 5)

    result = chebgridnd(c, x, y)

    assert np.allclose(result, np.ones((4, 5)))


def test_chebgridnd_t1_t1():
    c = np.zeros((2, 2))
    c[1, 1] = 1.0

    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([-1.0, 0.0, 1.0])

    expected = np.outer(x, y)

    result = chebgridnd(c, x, y)

    assert np.allclose(result, expected)


def test_chebgridnd_matches_chebvalnd():
    rng = np.random.default_rng(1234)

    c = rng.normal(size=(3, 4))

    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 6)

    xx, yy = np.meshgrid(x, y, indexing="ij")

    expected = chebvalnd(c, xx, yy)

    result = chebgridnd(c, x, y)

    assert np.allclose(result, expected)


def test_chebgridnd_1d_matches_chebval():
    c = np.array([1.0, 2.0, 3.0])

    x = np.linspace(-1, 1, 10)

    assert np.allclose(
        chebgridnd(c, x),
        chebval(x, c),
    )


def test_chebgridnd_accepts_lists():
    c = np.array([1.0, 2.0])

    result = chebgridnd(c, [-1, 0, 1])

    expected = chebval(
        np.array([-1.0, 0.0, 1.0]),
        c,
    )

    assert np.allclose(result, expected)


def test_chebvalnd_supports_broadcastable_coordinates():
    c = np.zeros((2, 2))
    c[0, 0] = 1.0  # constant
    c[1, 0] = 2.0  # 2*T1(x)
    c[0, 1] = 3.0  # 3*T1(y)

    x = np.linspace(-1, 1, 5).reshape(5, 1)
    y = np.linspace(-1, 1, 7).reshape(1, 7)

    result = chebvalnd(c, x, y)
    expected = 1.0 + 2.0 * x + 3.0 * y

    assert result.shape == (5, 7)
    assert np.allclose(result, expected)


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


@pytest.mark.parametrize(
    ("deg", "coords"),
    [
        ([2], ([0.0], [0.0])),  # 1 degree, 2 dimensions
        ([2, 3, 4], ([0.0], [0.0])),  # 3 degrees, 2 dimensions
        ([1, 2], ([0.0], [0.0], [0.0])),  # 2 degrees, 3 dimensions
    ],
)
def test_chebvandernd_degree_length_mismatch(
    deg,
    coords,
):
    with pytest.raises(
        ValueError,
        match="length of deg must be the same as number of dimensions",
    ):
        chebvandernd(deg, *coords)


def test_chebvandernd_matches_chebvander():
    """Test 1D consistency with NumPy's chebvander"""
    x = np.linspace(-1, 1, 5)

    v1 = chebvander(x, 4)
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


def test_chebvandernd_rejects_non_integer_degree():
    x = np.linspace(-1, 1, 5)

    with pytest.raises(
        ValueError,
        match="degrees must be non-negative integers",
    ):
        chebvandernd([1.5], x)  # type: ignore[list-item]


def test_chebvandernd_requires_at_least_one_coordinate():
    with pytest.raises(
        ValueError,
        match="at least one coordinate",
    ):
        chebvandernd([])


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

    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        chebvandernd([2, 2], x, y)


def test_chebyshevnd_degree():
    coef = np.zeros((3, 4, 5))

    approx = ChebyshevND(coef)

    assert approx.degree == (2, 3, 4)


def test_chebyshevnd_shape():
    coef = np.zeros((3, 4))

    approx = ChebyshevND(coef)

    assert approx.shape == (3, 4)


def test_chebyshevnd_eval_normalized():
    coef = np.array([1.0, 2.0])

    approx = ChebyshevND(
        coef,
        domain=[(0, 2)],
    )

    x = np.array([-1.0, 0.0, 1.0])

    expected = chebvalnd(coef, x)

    assert np.allclose(
        approx.eval_normalized(x),
        expected,
    )


def test_chebyshevnd_copy():
    coef = np.array([1.0, 2.0])

    approx = ChebyshevND(
        coef,
        domain=[(0, 2)],
    )

    copied = approx.copy()

    assert copied is not approx
    assert copied._domain is not None
    assert approx._domain is not None
    assert np.allclose(copied._coef, approx._coef)
    assert np.allclose(copied._domain, approx._domain)

    copied._coef[0] = 99

    assert approx._coef[0] == 1.0


def test_chebyshevnd_grid():
    coef = np.zeros((2, 2))
    coef[1, 1] = 1.0

    approx = ChebyshevND(coef)

    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([-1.0, 0.0, 1.0])

    expected = chebgridnd(coef, x, y)

    assert np.allclose(
        approx.grid(x, y),
        expected,
    )


def test_chebyshevnd_grid_uses_domain():
    coef = np.array([0.0, 1.0])

    approx = ChebyshevND(
        coef,
        domain=[(0, 2)],
    )

    x = np.array([0.0, 1.0, 2.0])

    assert np.allclose(
        approx.grid(x),
        [-1.0, 0.0, 1.0],
    )


def test_chebyshevnd_truncate():
    coef = np.arange(12.0).reshape(3, 4)

    approx = ChebyshevND(
        coef,
        domain=[(-1, 1), (-1, 1)],
    )

    truncated = approx.truncate(1, 2)

    assert truncated.shape == (2, 3)

    assert np.allclose(
        truncated._coef,
        coef[:2, :3],
    )

    assert truncated._domain is not None
    assert approx._domain is not None
    assert np.allclose(
        truncated._domain,
        approx._domain,
    )


@pytest.mark.parametrize(
    "degrees",
    [
        (),
        (1,),
        (1, 2, 3),
    ],
)
def test_chebyshevnd_truncate_wrong_length(degrees):
    approx = ChebyshevND(np.zeros((3, 3)))

    with pytest.raises(ValueError):
        approx.truncate(*degrees)


def test_chebyshevnd_truncate_negative_degree():
    approx = ChebyshevND(np.zeros((3, 3)))

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        approx.truncate(-1, 1)


def test_chebyshevnd_repr_without_domain():
    approx = ChebyshevND(np.zeros((3, 4)))

    assert repr(approx) == ("ChebyshevND(degree=(2, 3), domain=None)")


def test_chebyshevnd_repr_with_domain():
    approx = ChebyshevND(
        np.zeros((3,)),
        domain=[(0, 2)],
    )

    assert repr(approx) == ("ChebyshevND(degree=(2,), domain=[[0.0, 2.0]])")


def test_chebyshevnd_domain_mapping():
    approx = ChebyshevND.fit_dct(
        np.exp,
        n=9,
        domain=[(0, 2)],
    )

    x = np.linspace(0, 2, 20)

    assert np.allclose(
        approx(x),
        np.exp(x),
        atol=1e-10,
    )


def test_chebyshevnd_rejects_zero_width_domain():
    with pytest.raises(
        ValueError,
        match="nonzero width",
    ):
        ChebyshevND(
            [1.0],
            domain=[(1, 1)],
        )


def test_chebyshevnd_fit_preserves_domain():
    x = np.linspace(0, 2, 50)
    y = np.exp(x)

    approx = ChebyshevND.fit(
        (x,),
        y,
        deg=[8],
        domain=[(0, 2)],
    )
    assert approx.domain is not None
    assert np.array_equal(approx.domain, np.array([[0, 2]]))


def test_chebyshevnd_fit_domain_used_for_evaluation():

    xp = map_to_interval(chebyshev_nodes(9), 0, 2)
    y = np.exp(xp)

    approx = ChebyshevND.fit(
        (xp,),
        y,
        deg=[8],
        domain=[(0, 2)],
    )

    t = np.linspace(0, 2, 20)

    assert np.allclose(
        approx(t),
        np.exp(t),
        atol=1e-6,
    )


def test_select_degree_aic_exact_polynomial():
    x = np.linspace(-1, 1, 100)
    y = 1 + 2 * x + 3 * x**2

    deg = select_degree_aic(x, y)

    assert deg == 2


def test_select_degree_aic_length_mismatch():
    x = np.arange(5)
    y = np.arange(4)

    with pytest.raises(ValueError, match="same length"):
        select_degree_aic(x, y)


def test_select_degree_aic_identical_x():
    x = np.ones(10)
    y = np.arange(10)

    with pytest.raises(
        ValueError,
        match="must not all be identical",
    ):
        select_degree_aic(x, y)


def test_select_degree_aic_too_few_points():
    x = [0, 1]
    y = [0, 1]

    with pytest.raises(
        ValueError,
        match="at least 3 points",
    ):
        select_degree_aic(x, y)


def test_select_degree_aic_constant_function():
    x = np.linspace(-1, 1, 50)
    y = np.full_like(x, 3.0)

    deg = select_degree_aic(x, y)

    assert deg == 0


def test_select_degree_aic_small_dataset():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])

    deg = select_degree_aic(x, y)

    assert deg == 0


@pytest.mark.parametrize("tol", [0.0, -1e-12, -1.0])
def test_select_degree_aic_invalid_tol(tol):
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 4.0]

    with pytest.raises(
        ValueError,
        match="tol must be positive",
    ):
        select_degree_aic(x, y, tol=tol)