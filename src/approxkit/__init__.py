from typing import Any, Optional

from .chebyshev import (
    chebyshev_lobatto_nodes,
    chebextr,
    chebyshev_nodes,
    chebroot,
    ChebyshevND,
    chebfit1d,
    chebfit_dct,
    chebfitnd,
    chebvalnd,
    chebvandernd,
    chebgridnd,
    select_degree_aic,
)
from .pade import padefit, padefitlsq, PadeApproximation
from .utils import map_from_interval, map_to_interval
from ._version import version as __version__

_PACKAGE_NAME = __name__
__all__ = [
    "ChebyshevND",
    "PadeApproximation",
    "chebyshev_lobatto_nodes",
    "chebextr",
    "chebyshev_nodes",
    "chebroot",
    "chebfit1d",
    "chebfit_dct",
    "chebfitnd",
    "chebvalnd",
    "chebvandernd",
    "chebgridnd",
    "padefit",
    "padefitlsq",
    "map_from_interval",
    "select_degree_aic",
    "map_to_interval",
    "test",
    "__version__",
]


def test(*options: str, plugins: Optional[Any] = None) -> int:
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
