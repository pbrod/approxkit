from typing import Any

from ._version import version as __version__
from .chebyshev import (
    ChebyshevND,
    chebextr,
    chebfit1d,
    chebfit_dct,
    chebfitnd,
    chebgridnd,
    chebroot,
    chebvalnd,
    chebvandernd,
    chebyshev_lobatto_nodes,
    chebyshev_nodes,
    select_degree_aic,
)
from .pade import PadeApproximation, padefit, padefitlsq
from .utils import map_from_interval, map_to_interval

_PACKAGE_NAME = __name__
__all__ = [
    "ChebyshevND",
    "PadeApproximation",
    "__version__",
    "chebextr",
    "chebfit1d",
    "chebfit_dct",
    "chebfitnd",
    "chebgridnd",
    "chebroot",
    "chebvalnd",
    "chebvandernd",
    "chebyshev_lobatto_nodes",
    "chebyshev_nodes",
    "map_from_interval",
    "map_to_interval",
    "padefit",
    "padefitlsq",
    "select_degree_aic",
    "test",
]


def test(*options: str, plugins: Any | None = None) -> int:
    """
    Run tests for module using pytest.

    Parameters
    ----------
    *options : optional
        options to pass to pytest. The most important ones include:
        '-v', '--verbose':
            increase verbosity.
        '-q', '--quiet':
            decrease verbosity.
        '--doctest-modules':
            run doctests in all .py modules
        '--cov':
            measure coverage for .py modules (requires pytest-cov plugin)
        '-h', '--help':
            show full help message and display all possible options to use.

    Returns
    -------
    exit_code: int
        Exit code is 0 if all tests passed without failure.

    Examples
    --------
    {super}

    """
    try:
        import pytest
    except ImportError as exc:
        raise ImportError(
            "pytest is required to run approxkit.test(). "
            "Install it with: pip install 'approxkit[test]'"
        ) from exc

    return pytest.main(["--pyargs", _PACKAGE_NAME] + list(options), plugins=plugins)
