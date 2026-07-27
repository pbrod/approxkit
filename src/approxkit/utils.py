import numpy as np
from numpy.typing import ArrayLike, NDArray


def map_from_interval(
    x: ArrayLike,
    a: float,
    b: float,
) -> NDArray:
    """Map values from [a, b] to [-1, 1].

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([-1, 0, 1])
    >>> np.allclose(
    ...     map_from_interval(map_to_interval(x, 2, 4), 2, 4),
    ...     x
    ... )
    True
    """
    if a == b:
        raise ValueError(
            "interval endpoints must differ"
        )
    x = np.asarray(x)
    return (x - (b + a) / 2.0) * (2.0 / (b - a))


def map_to_interval(
    x: ArrayLike,
    a: float,
    b: float,
) -> NDArray:
    """Map values from [-1, 1] to [a, b]."""
    if a == b:
        raise ValueError(
            "interval endpoints must differ"
        )
    x = np.asarray(x)
    return (x * (b - a) + (b + a)) / 2.0


if __name__ == '__main__':
    from approxkit.testing import test_docstrings
    test_docstrings()