# ApproxKit

ApproxKit is a lightweight Python package for approximation theory and numerical analysis.

ApproxKit extends NumPy's Chebyshev approximation tools from one, two, and three dimensions to arbitrary dimensions.

It provides tools for:

- Fast Chebyshev approximation using discrete cosine transforms (DCT)
- Multidimensional Chebyshev fitting
- N-dimensional Chebyshev evaluation
- N-dimensional Chebyshev Vandermonde matrices
- Padé approximation
- Rational least-squares fitting
- Utility functions for interval transformations

## Installation

```bash
pip install approxkit
```

Install with testing support:

```bash
pip install "approxkit[test]"
```

## Requirements

- Python 3.10+
- NumPy
- SciPy
- mpmath

## Quick Start

Approximate \(e^x\) on the interval \([0, 2]\):

```python
import numpy as np
from approxkit import ChebyshevND

approx = ChebyshevND.fit_dct(
    np.exp,
    n=9,
    domain=[(0, 2)],
)

x = np.linspace(0, 2, 5)

print(approx(x))
print(np.exp(x))
```

The approximation behaves like a regular Python function while automatically handling the mapping between the physical domain and the Chebyshev interval \([-1, 1]\).

## Main API

```python
from approxkit import (
    ChebyshevND,
    PadeApproximation,
    chebfit_dct,
    chebfitnd,
    chebvalnd,
    chebvandernd,
    padefit,
    padefitlsq,
    map_to_interval,
    map_from_interval,
)
```

# Why ApproxKit?

ApproxKit complements NumPy's Chebyshev polynomial tools by extending them to arbitrary dimensions.

NumPy provides specialized routines for one-, two-, and three-dimensional Chebyshev fitting, evaluation, and Vandermonde matrix construction.

ApproxKit generalizes these capabilities to arbitrary dimensions.

## NumPy vs ApproxKit

| Capability | NumPy | ApproxKit |
|------------|--------|-----------|
| 1D fitting | ✓ | ✓ |
| 2D fitting | ✓ | ✓ |
| 3D fitting | ✓ | ✓ |
| N-dimensional fitting (N > 3) | ✗ | ✓ |
| N-dimensional evaluation (N > 3) | ✗ | ✓ |
| N-dimensional Vandermonde matrices (N > 3) | ✗ | ✓ |
| Domain-aware approximation objects | ✗ | ✓ |
| Padé approximation | ✗ | ✓ |
| Rational least-squares approximation | ✗ | ✓ |

## Familiar NumPy-style API

ApproxKit extends several NumPy Chebyshev functions:

| NumPy | ApproxKit |
|--------|-----------|
| `chebval()` | `chebvalnd()` |
| `chebvander()` | `chebvandernd()` |
| `chebfit2d()` / `chebfit3d()` | `chebfitnd()` |

This allows users familiar with NumPy's Chebyshev API to work naturally with higher-dimensional problems.

## N-Dimensional Fitting

NumPy provides:

```python
chebfit()
chebfit2d()
chebfit3d()
```

ApproxKit generalizes Chebyshev fitting to arbitrary dimensions:

```python
coef = chebfitnd(
    (x1, x2, x3, x4, x5),
    values,
    deg=[4, 4, 4, 4, 4],
)
```

## N-Dimensional Evaluation

NumPy provides:

```python
chebval()
chebval2d()
chebval3d()
```

ApproxKit generalizes evaluation to arbitrary dimensions:

```python
values = chebvalnd(
    coef,
    x1,
    x2,
    x3,
    x4,
    x5,
)
```

## N-Dimensional Vandermonde Matrices

NumPy provides:

```python
chebvander()
chebvander2d()
chebvander3d()
```

ApproxKit generalizes Vandermonde matrix construction to arbitrary dimensions:

```python
V = chebvandernd(
    [4, 4, 4, 4, 4],
    x1,
    x2,
    x3,
    x4,
    x5,
)
```

## High-Level Approximation Objects

ApproxKit also provides high-level approximation classes that combine fitting, evaluation, domain handling, and approximation metadata.

### Key ChebyshevND Features

- Automatic mapping between physical domains and `[-1, 1]`
- N-dimensional fitting
- N-dimensional evaluation
- Cartesian-grid evaluation
- Approximation truncation
- Convenient callable interface

### ChebyshevND

```python
approx = ChebyshevND.fit_dct(
    f,
    n=(12, 12, 12, 12),
)

y = approx(x1, x2, x3, x4)
```

### PadeApproximation

```python
p = padefit(coeffs)

y = p(x)
```

## Summary

ApproxKit is particularly useful when working with:

- Four or more dimensions
- Tensor-product Chebyshev approximations
- Surrogate models
- Scientific computing
- Reduced-order models
- High-dimensional interpolation
- High-dimensional regression

In short, ApproxKit aims to provide for arbitrary dimensions what NumPy's Chebyshev module provides for one, two, and three dimensions.

# API Overview

ApproxKit provides two high-level approximation objects.

## ChebyshevND

Represents a one-dimensional or multi-dimensional Chebyshev approximation.

A `ChebyshevND` instance stores Chebyshev coefficients together with optional domain information and behaves like a callable function.

```python
import numpy as np
from approxkit import ChebyshevND

approx = ChebyshevND.fit_dct(
    np.exp,
    n=9,
    domain=[(0, 2)],
)

y = approx(1.5)
```

### Main Features

- 1D, 2D, and N-dimensional approximations
- Automatic domain mapping
- Fast evaluation
- Degree truncation
- Cartesian-grid evaluation

### Common Methods

```python
approx(x)
approx.grid(x, y)
approx.truncate(5)
approx.copy()
```

### Typical Applications

- Surrogate modeling
- Numerical integration
- Interpolation
- Scientific computing
- Reduced-order models

---

## PadeApproximation

Represents a rational approximation

\[
f(x) \approx \frac{P(x)}{Q(x)}
\]

where \(P(x)\) and \(Q(x)\) are polynomials.

```python
from approxkit import padefit

coeffs = [1, 1, 1 / 2, 1 / 6, 1 / 24]

p = padefit(coeffs)

y = p(1.0)
```

### Main Features

- Classical Padé approximants
- Rational least-squares fitting
- Pole and zero analysis
- Optional domain metadata
- Optional error estimates

### Common Properties

```python
p.num
p.den
p.poles
p.zeros
```

### Typical Applications

- Extending Taylor approximations
- Rational-function modeling
- Asymptotic analysis
- Engineering approximations
- Reduced-order models

# Choosing the Right Fitting Method

ApproxKit provides several fitting methods optimized for different situations.

## Use chebfit_dct When...

You have values sampled on Chebyshev nodes or can efficiently evaluate a function on Chebyshev nodes.

### Advantages

- Fastest fitting method
- Uses a discrete cosine transform (DCT)
- No least-squares solve required
- Produces Chebyshev coefficients directly
- Supports multidimensional tensor-product grids

### Typical Use Cases

- Smooth callable functions
- Values already sampled at Chebyshev nodes
- Spectral methods
- Numerical quadrature
- Surrogate model generation

### Callable Example

```python
import numpy as np
from approxkit import chebfit_dct

c = chebfit_dct(np.tanh, n=25)
```

### Sampled-Value Example

```python
import numpy as np
from approxkit import (
    chebfit_dct,
    chebyshev_nodes,
)

x = chebyshev_nodes(25)
y = np.tanh(x)

c = chebfit_dct(y)
```

### Note

When passing sampled values, the data must already be sampled on a Chebyshev grid. For arbitrary sample locations, use `chebfitnd()`.

---

## Use chebfitnd When...

You have data sampled at arbitrary locations and want a least-squares Chebyshev approximation.

### Advantages

- Works with arbitrary sample locations
- Supports weighted least-squares fitting
- Does not require Chebyshev nodes
- Suitable for measured or simulated data

### Typical Use Cases

- Experimental measurements
- Simulation outputs
- Data-driven surrogate models
- Curve fitting
- Surface fitting

### Example

```python
coef = chebfitnd(
    (temperature, pressure),
    efficiency,
    deg=[6, 6],
)
```

---

## Use padefit When...

You know the Taylor-series coefficients of a function.

### Advantages

- Computes a classical Padé approximant
- Often converges beyond the Taylor radius of convergence
- Naturally represents poles
- Produces an exact rational approximation from the supplied series coefficients

### Typical Use Cases

- Symbolic mathematics
- Power-series acceleration
- Analytic approximations
- Asymptotic analysis

### Example

```python
coeffs = [1, 1, 1 / 2, 1 / 6, 1 / 24]

p = padefit(coeffs)
```

---

## Use padefitlsq When...

You have function evaluations or sampled data and want a rational approximation.

### Advantages

- Fits directly from sampled values
- Often achieves high accuracy using relatively few coefficients
- Useful on wide intervals
- Includes fitting-error metadata

### Typical Use Cases

- Wide-domain approximation
- Near-singular behavior
- Reduced-order models
- Compact rational approximations

### Example

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

# Performance Guidelines

| Situation | Recommended Method |
|------------|-------------------|
| Smooth callable function | `chebfit_dct` |
| Values sampled at Chebyshev nodes | `chebfit_dct` |
| Arbitrary sampled or measured data | `chebfitnd` |
| Taylor coefficients available | `padefit` |
| Rational approximation from sampled values | `padefitlsq` |
| Multidimensional approximation | `ChebyshevND.fit_dct()` or `ChebyshevND.fit()` |
| Need poles and zeros | `padefit()` or `padefitlsq()` |

### Rule of Thumb

- Use `chebfit_dct()` whenever Chebyshev-node samples are available.
- Use `chebfitnd()` for arbitrary sampled data.
- Use `padefit()` when Taylor coefficients are available.
- Use `padefitlsq()` when a rational approximation is desired from sampled values.

# Examples

## 1D Chebyshev Approximation

```python
import numpy as np
from approxkit import chebfit_dct, chebvalnd

c = chebfit_dct(np.exp, n=9)

x = np.linspace(-1, 1, 100)
y = chebvalnd(c, x)
```

## 2D Chebyshev Approximation

```python
import numpy as np
from approxkit import ChebyshevND

f = lambda x, y: np.tanh(x + y)

approx = ChebyshevND.fit_dct(
    f,
    n=(12, 12),
)

x = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, x, indexing="ij")

Z = approx(X, Y)
```

## 4D Least-Squares Approximation

```python
coef = chebfitnd(
    (x1, x2, x3, x4),
    values,
    deg=[4, 4, 4, 4],
)
```

## Padé Approximation

```python
from approxkit import padefit

coeffs = [1, 1, 1 / 2, 1 / 6, 1 / 24]

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

## Interval Mapping

```python
from approxkit import (
    map_to_interval,
    map_from_interval,
)

x = [-1, 0, 1]

y = map_to_interval(x, 2, 4)
z = map_from_interval(y, 2, 4)
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

## Development

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

## License

BSD 3-Clause License.

## Author

Per A. Brodtkorb