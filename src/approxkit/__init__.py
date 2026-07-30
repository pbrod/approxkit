from functools import wraps
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
from .testing import test as _test
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


@wraps(_test)
def test(*options: str, plugins: Any | None = None) -> int:
    return _test(__name__, *options, plugins=plugins)
