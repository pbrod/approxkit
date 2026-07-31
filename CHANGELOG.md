# CHANGELOG

## v0.2.0 (2026-07-31)

## Added

- Added N-dimensional Chebyshev approximation support via `ChebyshevND`.
- Added domain-aware approximation objects with callable evaluation.
- Added `PadeApproximation` as a unified object for rational approximations.
- Added approximation metadata including domains and error estimates.
- Added Chebyshev and Chebyshev-Lobatto node utilities.
- Added automatic polynomial degree selection via `select_degree_aic()`.
- Added comprehensive test suite covering Chebyshev, Padé, and utility functions.
- Added package documentation, release notes, CI workflows, and packaging configuration.

## Changed

- Overhauled the README with:
  - N-dimensional quick-start examples
  - API overview
  - method selection guide
  - NumPy comparison
  - practical examples and usage patterns
- Improved rational approximation documentation and examples.
- Modernized the Padé API using `numpy.polynomial.Polynomial`.
- Standardized polynomial coefficient ordering to NumPy conventions.
- Improved typing throughout the codebase and enabled mypy validation.
- Refactored test infrastructure and moved testing utilities into `approxkit.testing`.
- Updated GitHub Actions workflows and CI configuration.
- Improved package metadata and development tooling.

## Improved

- Improved numerical stability of `select_degree_aic()`.
- Improved weighting support and numerical robustness in Chebyshev fitting routines.
- Improved validation and error handling throughout Chebyshev, Padé, and utility modules.
- Expanded regression testing and edge-case coverage.
- Improved documentation, examples, docstrings, and terminology consistency.

## Fixed

- Prevented overfitting in `select_degree_aic()` by introducing a residual floor in AIC calculations.
- Fixed least-squares Padé fitting to return the best solution rather than the final iterate.
- Fixed validation and error-reporting issues across approximation utilities.
- Resolved typing issues and achieved clean mypy checks.

## Build & Infrastructure

- Require Python 3.11 or newer.
- Added Python 3.14 CI support.
- Added coverage tooling (`pytest-cov`).
- Added license-file packaging support.
- Applied Ruff formatting and style cleanup.