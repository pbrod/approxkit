# ApproxKit

[![PyPI version](https://img.shields.io/pypi/v/approxkit.svg)](https://pypi.org/project/approxkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/approxkit.svg)](https://pypi.org/project/approxkit/)
[![License](https://img.shields.io/pypi/l/approxkit.svg)](LICENSE.txt)
[![CI Status](https://github.com/pbrod/approxkit/actions/workflows/tests.yml/badge.svg)](https://github.com/pbrod/approxkit/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/pbrod/approxkit/branch/main/graph/badge.svg)](https://codecov.io/gh/pbrod/approxkit)
[![Ruff](https://img.shields.io/badge/lint-ruff-blueviolet)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/type--checked-mypy-blue)](http://mypy-lang.org/)
[![Downloads](https://pepy.tech/badge/approxkit/month)](https://pepy.tech/project/approxkit)

ApproxKit extends NumPy's Chebyshev approximation tools to arbitrary dimensions and provides fast polynomial and rational approximation methods for scientific computing.

It supports:

- Fast Chebyshev approximation using discrete cosine transforms (DCT)
- N-dimensional Chebyshev fitting
- N-dimensional Chebyshev evaluation
- N-dimensional Chebyshev Vandermonde matrices
- Chebyshev and Chebyshev-Lobatto node generation
- Automatic polynomial-degree selection using AIC
- Padé approximation
- Rational least-squares fitting
- Utility functions for interval transformations

---

# Installation

```bash
pip install approxkit
```

Install with testing support:

```bash
pip install "approxkit[test]"
```

## Requirements

- Python 3.11+
- NumPy
- SciPy
- mpmath

---

# Quick Start

Approximate a four-dimensional function on
`[(0, 4), (0, 4), (0, 4), (0, 4)]`:

```python
from approxkit import ChebyshevND

approx = ChebyshevND.fit_dct(
    lambda x, y, z, w: x + y * z + w**2,
    n=(8, 8, 8, 8),
    domain=[
        (0, 4),
        (0, 4),
        (0, 4),
        (0, 4),
    ],
)

value = approx(0.1, 0.2, 0.3, 0.4)
```

The approximation behaves like a regular Python function while automatically handling the mapping between the physical domain and the Chebyshev interval `[-1, 1]`.

Chebyshev polynomials are naturally defined on `[-1, 1]`.
Internally, the approximation is evaluated on `[-1, 1]` while users work directly in physical coordinates.

## 1D Chebyshev Approximation

Approximate `exp(x)` on `[0, 2]`:

```python
import numpy as np
from approxkit import ChebyshevND

approx = ChebyshevND.fit_dct(
    np.exp,
    n=9,
    domain=[(0, 2)],
)

x = np.linspace(0, 2, 50)

assert np.allclose(
    approx(x),
    np.exp(x),
    atol=1e-10,
)
```

---

# Why ApproxKit?

NumPy provides excellent Chebyshev support for one-, two-, and three-dimensional problems.

ApproxKit generalizes these capabilities to **arbitrary dimensions** while adding:

- Domain-aware approximation objects in arbitrary dimensions
- Fast DCT-based fitting
- Padé approximation
- Rational least-squares fitting
- Unified N-dimensional APIs

This makes ApproxKit useful for:

- Surrogate modeling
- Scientific computing
- Reduced-order models
- Numerical integration
- High-dimensional approximation problems


## Approximation Objects

Unlike lower-level fitting routines that only return coefficients,
ApproxKit provides high-level approximation objects.

```python
import numpy as np
from approxkit import ChebyshevND

approx = ChebyshevND.fit_dct(
    np.exp,
    n=9,
)

y = approx(x)
```

These objects combine:

- model coefficients
- domain metadata
- callable evaluation
- approximation diagnostics
- utility methods

into a single callable interface.

---


# Key Features

## N-Dimensional Chebyshev Approximation

Fit and evaluate Chebyshev approximations in any number of dimensions.

```python
from approxkit import chebfitnd
coef = chebfitnd(
    (x1, x2, x3, x4, x5),
    values,
    deg=[4, 4, 4, 4, 4],
)
```

## Domain-Aware Approximation Objects

```python
import numpy as np
from approxkit import ChebyshevND
approx = ChebyshevND.fit_dct(
    np.exp,
    n=9,
    domain=[(0, 2)],
)

y = approx(1.5)  # ≈ exp(1.5)
```

Physical coordinates are automatically mapped to the Chebyshev interval.

## Fast DCT-Based Fitting

```python
import numpy as np
from approxkit import chebfit_dct
c = chebfit_dct(
    np.tanh,
    n=25,
)
```

Uses discrete cosine transforms to compute coefficients efficiently.

## Rational Approximation

```python
import numpy as np
from numpy.polynomial import Polynomial
from approxkit import padefit, padefitlsq

x = np.linspace(0, 2, 100)

# Taylor polynomial for exp
p = Polynomial(
    [1, 1, 1 / 2, 1 / 6, 1 / 24]
)

assert np.allclose(
    p(x),
    np.exp(x),
    atol=1e-1,
)

# Classical Padé approximation from Taylor coefficients
pade = padefit(p.coef)

assert np.allclose(
    pade(x),
    np.exp(x),
    atol=1e-2,
)

# Rational least-squares fit from sampled values
rational = padefitlsq(
    np.exp,
    m=3,
    n=3,
    a=0,
    b=2,
)

assert np.allclose(
    rational(x),
    np.exp(x),
    atol=1e-6,
)
```

Compared with a Taylor polynomial of the same order, Padé and rational least-squares approximations often achieve substantially higher accuracy over a finite interval.

Supports both classical Padé approximation and least-squares rational fitting.


## Chebyshev Node Generation

Generate interpolation and quadrature nodes for interpolation,
quadrature, and spectral methods.

```Python
from approxkit import chebyshev_nodes, chebyshev_lobatto_nodes

x = chebyshev_nodes(16)

x = chebyshev_lobatto_nodes(16)
```

## Automatic Degree Selection

Select a suitable polynomial degree using Akaike's Information Criterion.

```python
from approxkit import select_degree_aic

deg = select_degree_aic(x, y)
```

This helps balance approximation accuracy and model complexity.


---

# NumPy vs ApproxKit

| Capability | NumPy | ApproxKit |
|------------|--------|-----------|
| 1D fitting | ✓ | ✓ |
| 2D fitting | ✓ | ✓ |
| 3D fitting | ✓ | ✓ |
| N-dimensional fitting (N > 3) | ✗ | ✓ |
| N-dimensional evaluation (N > 3) | ✗ | ✓ |
| N-dimensional Vandermonde matrices (N > 3) | ✗ | ✓ |
| Domain-aware approximation objects (1D) | ✓ | ✓ |
| Domain-aware approximation objects (ND) | ✗ | ✓ |
| Automatic degree selection (AIC) | ✗ | ✓ |
| Padé approximation | ✗ | ✓ |
| Rational least-squares approximation | ✗ | ✓ |

## Familiar NumPy-Style API

ApproxKit extends NumPy's Chebyshev tools:

| NumPy | ApproxKit |
|--------|-----------|
| `chebval()` | `chebvalnd()` |
| `chebvander()` | `chebvandernd()` |
| `chebfit2d()` / `chebfit3d()` | `chebfitnd()` |
| `chebpts1()` | `chebyshev_nodes()` |
| `chebpts2()` | `chebyshev_lobatto_nodes()` |

---

# Choosing the Right Method

## Decision Guide

```text
What do you want to do?

├─ Approximate a callable function on Chebyshev nodes
│  ├─ 1D or ND approximation object → ChebyshevND.fit_dct()
│  └─ Coefficients only → chebfit_dct()
│
├─ Fit Chebyshev polynomials to arbitrary sampled data
│  ├─ Approximation object → ChebyshevND.fit()
│  ├─ N-dimensional coefficients → chebfitnd()
│  └─ 1D Chebyshev polynomial → chebfit1d()
│
├─ Construct a rational approximation
│  ├─ Taylor coefficients available → padefit()
│  └─ Sampled values available → padefitlsq()
│
└─ Utilities
   ├─ Choose interpolation nodes → chebyshev_nodes()
   ├─ Need endpoints included → chebyshev_lobatto_nodes()
   └─ Unknown polynomial degree → select_degree_aic()
```

## DCT Fitting

Use `chebfit_dct()` when:

- Function values are available on Chebyshev nodes
- The function is inexpensive to evaluate on Chebyshev grids
- Maximum fitting speed is desired

```python
import numpy as np
from approxkit import chebfit_dct
c = chebfit_dct(
    np.exp,
    n=25,
)
```

### Note on `n`

`n` specifies the number of Chebyshev nodes used to construct the approximation.

---

## Least-Squares Chebyshev Fitting

Use `chebfitnd()` when:

- Data are sampled at arbitrary locations
- Experimental or simulation data must be fitted
- Weighted least-squares fitting is desired

```python
from approxkit import chebfitnd

coef = chebfitnd(
    (temperature, pressure),
    efficiency,
    deg=[6, 6],
)
```

### Note on `deg`

`deg` specifies the polynomial degree in each dimension.

---

## 1D Convenience Fitting

```python
p = chebfit1d(
    x,
    y,
    deg=8,
)
```

Returns a fitted `numpy.polynomial.Chebyshev` object and serves as a convenience wrapper around `numpy.polynomial.Chebyshev.fit()`.

---

### Choosing Interpolation Nodes

Use `chebyshev_nodes()` when:

- Building interpolation polynomials
- Sampling smooth functions
- Minimizing Runge phenomena

```python
x = chebyshev_nodes(32)
```

Use `chebyshev_lobatto_nodes()` when:

- Endpoints must be included
- Spectral methods are used
- Minimax approximation workflows are implemented

```python
x = chebyshev_lobatto_nodes(32)
```

---

### Choosing a Polynomial Degree

Use `select_degree_aic()` when:

- The polynomial degree is unknown
- Data contain noise
- Overfitting should be avoided

```python
degree = select_degree_aic(
    x,
    y,
)

p = chebfit1d(
    x,
    y,
    deg=degree,
)
```

The selected degree minimizes Akaike's Information Criterion (AIC), balancing model complexity against residual error.

---

## Classical Padé Approximation

Use `padefit()` when Taylor-series coefficients are known.

```python
coeffs = [1, 1, 1 / 2, 1 / 6, 1 / 24]

p = padefit(coeffs)
```

---

## Rational Least-Squares Approximation

Use `padefitlsq()` when sampled values are available.

```python
import numpy as np
from approxkit import padefitlsq
p = padefitlsq(
    np.exp,
    m=3,
    n=3,
    a=0,
    b=2,
)
```

---

# API Overview

ApproxKit provides both low-level fitting utilities and high-level approximation objects.

```python
from approxkit import (
    ChebyshevND,
    PadeApproximation,
    chebfit_dct,
    chebfit1d,
    chebfitnd,
    chebvalnd,
    chebvandernd,
    chebyshev_nodes,
    chebyshev_lobatto_nodes,
    select_degree_aic,
    padefit,
    padefitlsq,
    map_to_interval,
    map_from_interval,
)
```

## Approximation Objects

ApproxKit provides high-level approximation objects for both polynomial and rational approximation:

```python
ChebyshevND
PadeApproximation
```

These objects are callable and carry approximation metadata such as domains, coefficients, poles, zeros, and error estimates.

## ChebyshevND

Represents a Chebyshev approximation together with optional domain metadata.

### Common Methods

```python
approx(x)

approx.grid(x, y)

approx.truncate(5)

approx.copy()
```

### Features

- Automatic domain mapping
- N-dimensional fitting
- N-dimensional evaluation
- Cartesian-grid evaluation
- Truncation support
- Callable interface

---

## PadeApproximation

Represents a rational approximation

```text
f(x) ≈ P(x) / Q(x)
```

### Common Properties

**Mathematical properties**

Numerator, denominator, poles, and zeros of the rational approximation.

```python
p.num
p.den
p.zeros
p.poles
```

**Stored metadata**

Optional information associated with the approximation.

```python
p.max_error
p.domain
```

**Convenience predicates**

Check whether optional metadata is available.

```python
p.has_error_estimate
p.has_domain
```

### Features

- Classical Padé approximation
- Rational least-squares fitting
- Pole and zero analysis
- Domain metadata
- Optional error estimates
---

## Utility Functions

ApproxKit provides utility functions for node generation,
degree selection, and domain transformations.

### Chebyshev Nodes (Roots of Tₙ)

```python
from approxkit import chebyshev_nodes

x = chebyshev_nodes(16)
```

These are the roots of the Chebyshev polynomial of the first kind and are commonly used for interpolation because they minimize Runge oscillations. Equivalent to NumPy's `chebpts1()`.

### Chebyshev-Lobatto Nodes

```python
from approxkit import chebyshev_lobatto_nodes

x = chebyshev_lobatto_nodes(16)
```

These are the extrema of the Chebyshev polynomial of the first kind and include the endpoints `-1` and `1`.
Equivalent to NumPy's `chebpts2()`.

Typical applications:

- Polynomial interpolation
- Spectral methods
- Numerical quadrature
- Minimax approximation algorithms


### Automatic Degree Selection

ApproxKit can estimate an appropriate polynomial degree using Akaike's Information Criterion (AIC).

```python
import numpy as np
from approxkit import (
    chebfit1d,
    select_degree_aic,
)

x = np.linspace(0, 10, 300)

y = np.sin(x**3 / 100) ** 2

degree = select_degree_aic(
    x,
    y,
)

p = chebfit1d(
    x,
    y,
    deg=degree,
)
```

This is useful when the appropriate polynomial degree is not known in advance.


### Interval Mapping

```python
map_to_interval(x, a, b)
map_from_interval(x, a, b)
```

Convert values between physical domains and the Chebyshev interval `[-1, 1]`.

---

# Examples

## 1D Chebyshev Approximation

```python
import numpy as np
from approxkit import ChebyshevND, chebfit_dct, chebvalnd

c = chebfit_dct(
    np.exp,
    n=9,
)

x = np.linspace(-1, 1, 100)

y = chebvalnd(c, x)

approx = ChebyshevND.fit_dct(
    np.exp,
    n=9,
)
y1 = approx(x)

approx2 = approx.truncate(5)

y2 = approx2(x)
```

## 2D Approximation

```python
import numpy as np
from approxkit import ChebyshevND

approx = ChebyshevND.fit_dct(
    lambda x, y: np.tanh(x + y),
    n=(12, 12),
)

u = np.linspace(-1, 1, 50)

X, Y = np.meshgrid(
    u,
    u,
    indexing="ij",
)

Z = approx(X, Y)
```

## 4D Least-Squares Fit

```python
coef = chebfitnd(
    (x1, x2, x3, x4),
    values,
    deg=[4, 4, 4, 4],
)
```

---

## Padé Approximation

```python
from approxkit import padefit

coeffs = [
    1,
    1,
    1 / 2,
    1 / 6,
    1 / 24,
]

p = padefit(coeffs)

y = p(1.0)
```

## Rational Least-Squares Approximation

```python
import numpy as np
from approxkit import padefitlsq

p = padefitlsq(
    np.exp,
    m=3,
    n=3,
    a=0,
    b=2,
)
```

---

## Chebyshev Nodes

```python
from approxkit import chebyshev_nodes

x = chebyshev_nodes(8)
```

## Chebyshev-Lobatto Nodes

```python
from approxkit import chebyshev_lobatto_nodes

x = chebyshev_lobatto_nodes(8)
```

## Automatic Degree Selection

```python
import numpy as np

from approxkit import (
    chebfit1d,
    select_degree_aic,
)

x = np.linspace(0, 10, 300)
y = np.sin(x**3 / 100) ** 2

deg = select_degree_aic(
    x,
    y,
)

p = chebfit1d(
    x,
    y,
    deg=deg,
)
```

## Interval Mapping

```python
from approxkit import (
    map_to_interval,
    map_from_interval,
)

x = [-1, 0, 1]

y = map_to_interval(
    x,
    2,
    4,
)

z = map_from_interval(
    y,
    2,
    4,
)
```

## Running Tests

```python
import approxkit

approxkit.test()
```

or

```bash
pytest --pyargs approxkit
```

---

# Development

```bash
git clone https://github.com/pbrod/approxkit.git

cd approxkit

pip install -e .
```

Install testing support:

```bash
pip install -e ".[test]"
```

Run tests:

```bash
pytest --pyargs approxkit
```

---

# License

BSD 3-Clause License.

## Author

Per A. Brodtkorb