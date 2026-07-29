import pytest
import numpy as np
from approxkit.utils import map_from_interval, map_to_interval


def test_map_from_interval_rejects_zero_width():
    with pytest.raises(
        ValueError,
        match="interval endpoints must differ",
    ):
        map_from_interval([0], 1, 1)


def test_map_to_interval_rejects_zero_width():
    with pytest.raises(
        ValueError,
        match="interval endpoints must differ",
    ):
        map_to_interval([0], 1, 1)


def test_interval_mapping_roundtrip():
    x = np.linspace(-1, 1, 11)
    y = map_to_interval(x, 2, 5)
    xr = map_from_interval(y, 2, 5)
    assert np.allclose(xr, x)


def test_map_to_interval():
    x = np.array([-1, 0, 1])

    assert np.allclose(
        map_to_interval(x, 2, 4),
        [2, 3, 4],
    )


def test_map_from_interval():
    x = np.array([2, 3, 4])

    assert np.allclose(
        map_from_interval(x, 2, 4),
        [-1, 0, 1],
    )
