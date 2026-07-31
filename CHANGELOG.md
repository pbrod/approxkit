# CHANGELOG

## v0.2.0 (2026-07-31)

### Added

- Added automatic polynomial degree selection via `select_degree_aic()`.
- Added approximation metadata to `PadeApproximation`, including:
  - domain information
  - error estimates
  - poles and zeros
- Added comprehensive test coverage for Chebyshev, Padé, and utility functionality.
- Added release workflow documentation and changelog generation procedures.
- Added Python 3.14 CI support and coverage tooling.

### Changed

- Expanded ApproxKit's documentation with:
  - a reorganized README
  - N-dimensional quick-start examples
  - API overview
  - method-selection decision guide
  - practical usage examples
  - improved rational approximation documentation
- Modernized the Padé approximation API using `numpy.polynomial.Polynomial`.
- Standardized coefficient handling and improved API consistency.
- Refactored test infrastructure into `approxkit.testing`.
- Improved package metadata, packaging configuration, and development tooling.

### Improved

- Improved numerical stability and robustness of `select_degree_aic()`.
- Improved Chebyshev fitting stability, weighting support, typing, and validation.
- Improved validation and error handling across Chebyshev, Padé, and utility modules.
- Expanded regression testing and edge-case coverage.
- Improved documentation, examples, docstrings, and terminology consistency.
- Applied Ruff formatting and general code-quality improvements.

### Fixed

- Prevented overfitting in `select_degree_aic()` through a residual-error floor.
- Fixed least-squares Padé fitting to return the best approximation found.
- Resolved typing issues and achieved clean mypy validation.

### Build & Infrastructure

- Require Python 3.11 or newer.
- Updated GitHub Actions workflows to current major versions.
- Added license file packaging support.
- Improved CI, build validation, and release workflows.
