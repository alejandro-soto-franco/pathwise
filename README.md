# pathwise

High-performance SDE simulation toolkit: Rust core, Python API.

[![CI](https://github.com/alejandro-soto-franco/pathwise/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandro-soto-franco/pathwise/actions/workflows/ci.yml)

## Overview

`pathwise` provides composable, high-performance SDE simulation in Python backed by a Rust
core. Paths are returned as NumPy arrays. The Rust-level API simulates paths in parallel via
Rayon; the Python `simulate()` runs serially (one path at a time) because Python callables
require the GIL. Built-in processes (`bm`, `gbm`, `ou`) avoid this constraint.

**Key differentiator:** first-class support for non-Markovian and fractional processes
(fractional BM, Volterra SDEs, rough volatility) is on the roadmap. No other Python SDE
library covers this gap well.

## Features (v0.1)

- Euler-Maruyama and Milstein schemes
- Built-in processes: Brownian motion, GBM, Ornstein-Uhlenbeck
- Composable API: define any SDE from Python callables
- Parallel CPU simulation via Rayon (Rust API); serial for Python callables (GIL)
- NumPy array output

## Install

```bash
pip install pathwise
```

Build from source:

```bash
git clone https://github.com/alejandro-soto-franco/pathwise.git
cd pathwise
pip install maturin
python -m venv .venv && source .venv/bin/activate
maturin develop
```

## Quick start

```python
import pathwise as pw

# 10,000 GBM paths, 252 daily steps
paths = pw.simulate(
    pw.gbm(mu=0.05, sigma=0.2),
    scheme=pw.euler(),
    n_paths=10_000,
    n_steps=252,
    t1=1.0,
    x0=100.0,
)
# paths: np.ndarray, shape (10_000, 253), includes t=0

# Milstein scheme (higher strong order on state-dependent diffusion)
paths = pw.simulate(pw.gbm(0.05, 0.4), pw.milstein(),
                    n_paths=5_000, n_steps=500, t1=1.0, x0=1.0)

# Custom SDE from any Python callable
ou_custom = pw.sde(
    drift=lambda x, t: 2.0 * (1.0 - x),
    diffusion=lambda x, t: 0.3,
)
paths = pw.simulate(ou_custom, pw.euler(), n_paths=1000, n_steps=200, t1=1.0)
```

## API

### Processes

| Function | SDE |
|----------|-----|
| `pw.bm()` | $dX = dW$ |
| `pw.gbm(mu, sigma)` | $dX = \mu X \, dt + \sigma X \, dW$ |
| `pw.ou(theta, mu, sigma)` | $dX = \theta(\mu - X) \, dt + \sigma \, dW$ |
| `pw.sde(drift, diffusion)` | $dX = f(X,t) \, dt + g(X,t) \, dW$ |

### Schemes

| Function | Strong order | Weak order |
|----------|:---:|:---:|
| `pw.euler()` | $\tfrac{1}{2}$ | $1$ |
| `pw.milstein()` | $1$ | $1$ |

### simulate

```python
pw.simulate(
    sde,           # SDE object (bm, gbm, ou, or custom)
    scheme,        # euler() or milstein()
    n_paths,       # number of independent paths
    n_steps,       # time steps per path
    t1,            # end time
    x0=0.0,        # initial value
    t0=0.0,        # start time
    device="cpu",  # "cpu" only in v0.1
)
# returns: np.ndarray of shape (n_paths, n_steps + 1)
```

### Exceptions

```python
pw.NumericalDivergence   # reserved for future use (non-finite values are stored as NaN)
pw.ConvergenceError      # raised on inference failures (future versions)
```

## Numerical validation

The test suite verifies convergence orders against analytic benchmarks:

- **Euler strong order** measured at 0.49 (expected $\tfrac{1}{2}$) on GBM via common-noise regression
- **Milstein strong order** measured at 0.98 (expected $1$) on GBM
- **GBM mean/variance** match $\mathbb{E}[X_T] = x_0 e^{\mu T}$ and the exact second moment formula
- **OU conditional moments** match analytic formulas
- **BM increments** pass Kolmogorov-Smirnov normality test ($\mathcal{N}(0, \Delta t)$)
- **BM quadratic variation** converges to $T$
- **GBM log-normality** verified via KS test on $\log X_T$
- **European call price** matches Black-Scholes within 1%

Run the full suite:

```bash
cargo test -p pathwise-core            # Rust unit and integration tests
cargo test -p pathwise-core --test convergence
pytest tests/                          # Python API and numerical tests
```

## Roadmap

- v0.2: fractional BM, Volterra processes, Bayer-Friz-Gatheral scheme
- v0.3: CUDA GPU acceleration
- v0.4: MLE inference, particle filter
- v0.5: rough Heston / rough Bergomi surface calibration
- v1.0: stable API, JOSS paper

## License

MIT
