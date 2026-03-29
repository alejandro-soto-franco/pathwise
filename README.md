# pathwise

High-performance SDE simulation toolkit: Rust core, Python API.

[![CI](https://github.com/alejandro-soto-franco/pathwise/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandro-soto-franco/pathwise/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/pathwise-core.svg)](https://crates.io/crates/pathwise-core)
[![docs.rs](https://docs.rs/pathwise-core/badge.svg)](https://docs.rs/pathwise-core)
[![PyPI](https://img.shields.io/pypi/v/pathwise-sde.svg)](https://pypi.org/project/pathwise-sde/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

`pathwise` provides composable, high-performance SDE simulation in Python backed by a Rust core. Paths are returned as NumPy arrays with shape `(n_paths, n_steps+1)` for scalar processes and `(n_paths, n_steps+1, N)` for N-dimensional processes. Built-in processes (bm, gbm, ou, cir, heston, corr_ou) execute in parallel via Rayon with the GIL released; custom Python callables run serially.

**Key differentiator:** `pathwise` ships a Taylor 1.5 (SRI) strong-order 1.5 scheme alongside standard Euler and Milstein, and extends to geodesic SDE simulation on Riemannian manifolds via the `pathwise-geo` crate.

## Install

```bash
pip install pathwise-sde
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

# 10,000 GBM paths, 252 daily steps, Euler-Maruyama
paths = pw.simulate(
    pw.gbm(mu=0.05, sigma=0.2),
    scheme=pw.euler(),
    n_paths=10_000,
    n_steps=252,
    t1=1.0,
    x0=100.0,
)
# paths: np.ndarray shape (10_000, 253), includes t=0

# Milstein scheme (strong order 1 on state-dependent diffusion)
paths = pw.simulate(
    pw.gbm(0.05, 0.4),
    pw.milstein(),
    n_paths=5_000,
    n_steps=500,
    t1=1.0,
    x0=1.0,
)

# SRI scheme (strong order 1.5, Kloeden-Platen Taylor 1.5)
paths = pw.simulate(
    pw.gbm(0.05, 0.2),
    pw.sri(),
    n_paths=5_000,
    n_steps=252,
    t1=1.0,
    x0=1.0,
)

# Cox-Ingersoll-Ross (requires Feller condition: 2*kappa*theta > sigma^2)
paths = pw.simulate(
    pw.cir(kappa=1.0, theta=0.04, sigma=0.1),
    pw.euler(),
    n_paths=5_000,
    n_steps=252,
    t1=1.0,
    x0=0.04,
)

# Heston stochastic volatility
# x0 = log(S0); initial variance defaults to theta
# returns shape (n_paths, n_steps+1, 2): [:,:,0] = log-price, [:,:,1] = variance
paths = pw.simulate(
    pw.heston(mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7),
    pw.euler(),
    n_paths=5_000,
    n_steps=252,
    t1=1.0,
    x0=0.0,   # log(S0) = log(1.0) = 0
)
log_price = paths[:, :, 0]
variance  = paths[:, :, 1]
```

## API reference

### Processes

| Function | SDE | Notes |
|----------|-----|-------|
| `pw.bm()` | $dX = dW$ | Standard Brownian motion |
| `pw.gbm(mu, sigma)` | $dX = \mu Xdt + \sigma X dW$ | Geometric Brownian motion |
| `pw.ou(theta, mu, sigma)` | $dX = \theta(\mu - X) dt + \sigmadW$ | Ornstein-Uhlenbeck |
| `pw.cir(kappa, theta, sigma)` | $dX = \kappa(\theta - X) dt + \sigma\sqrt{X}dW$ | Cox-Ingersoll-Ross; raises `ValueError` if Feller violated |
| `pw.heston(mu, kappa, theta, xi, rho)` | coupled log-price + CIR variance | Returns `(n_paths, n_steps+1, 2)` array |
| `pw.corr_ou(theta, mu_vec, sigma_flat)` | N-dim correlated OU | `mu_vec`: list of length N; `sigma_flat`: flattened N×N covariance; Python bindings: N=2 only |
| `pw.sde(drift, diffusion)` | $dX = f(X,t)dt + g(X,t) dW$ | Custom Python callables, runs serially |

### Schemes

| Function | Strong order | Weak order | Notes |
|----------|:---:|:---:|-------|
| `pw.euler()` | $\tfrac{1}{2}$ | $1$ | Euler-Maruyama |
| `pw.milstein()` | $1$ | $1$ | Requires diffusion gradient |
| `pw.sri()` | $\tfrac{3}{2}$ | $2$ | Kloeden-Platen Taylor 1.5; not supported for Heston or CorrOU |

### simulate

```python
pw.simulate(
    sde,           # SDE object (bm, gbm, ou, cir, heston, corr_ou, or custom sde())
    scheme,        # euler(), milstein(), or sri()
    n_paths,       # number of independent paths (int)
    n_steps,       # time steps per path (int)
    t1,            # end time (float)
    x0=0.0,        # initial value (float)
                   #   for Heston: x0 = log(S0); initial variance = theta
    t0=0.0,        # start time (float)
    device="cpu",  # "cpu" only in v0.2
) -> np.ndarray
# Scalar SDE: shape (n_paths, n_steps + 1)
# N-dim SDE:  shape (n_paths, n_steps + 1, N)
#   Heston N=2: [:,:,0] = log-price, [:,:,1] = variance
```

**Parameter notes:**

- `x0` for Heston is `log(S0)`, not `S0`. To simulate starting from price 100, pass `x0=math.log(100)`.
- `sigma_flat` for `corr_ou` is the covariance matrix (not correlation) laid out in row-major order. For N=2 pass `[var1, cov12, cov12, var2]`.
- SRI raises `ValueError` if used with `heston` or `corr_ou`.
- CIR raises `ValueError` (message contains "Feller") if `2 * kappa * theta <= sigma ** 2`.

### Exceptions

```python
pw.NumericalDivergence   # reserved; non-finite values are stored as NaN
pw.ConvergenceError      # raised on inference failures (future versions)
# ValueError             # raised by cir (Feller), sri+heston, sri+corr_ou
```

## Numerical validation

The test suite verifies convergence orders against analytic benchmarks:

- **Euler strong order** measured at 0.49 (expected $\tfrac{1}{2}$) on GBM via common-noise regression
- **Milstein strong order** measured at 0.98 (expected $1$) on GBM
- **SRI strong order** measured at ~1.5 (expected $\tfrac{3}{2}$) on GBM
- **GBM mean/variance** match $\mathbb{E}[X_T] = x_0 e^{\mu T}$ and the exact second moment formula
- **OU conditional moments** match analytic formulas
- **CIR non-negativity** verified; moments checked against closed-form
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

## pathwise-geo (Rust only)

`pathwise-geo` is a companion crate for geodesic SDE simulation on Riemannian manifolds, backed by the [cartan](https://github.com/alejandro-soto-franco/cartan) geometry library.

**Supported manifolds:** $S^n$ (hypersphere), $SO(n)$ (rotation group), $SPD(n)$ (symmetric positive-definite matrices).

**Schemes:** `GeodesicEuler`, `GeodesicMilstein`, `GeodesicSRI`.

Add to your `Cargo.toml`:

```toml
[dependencies]
pathwise-geo = "0.2"
```

Python bindings for `pathwise-geo` are planned for v0.3. See the [crate docs](https://docs.rs/pathwise-geo) for the Rust API.

## Roadmap

| Version | Status | Items |
|---------|--------|-------|
| v0.1 | released | Euler, Milstein; BM, GBM, OU; custom Python SDEs |
| v0.2 | released | SRI (Taylor 1.5); CIR, Heston, CorrOU; pathwise-geo (geodesic SDEs on S^n/SO(n)/SPD(n)) |
| v0.3 | planned | Python bindings for pathwise-geo; fractional BM; Volterra SDEs |
| v0.4 | planned | CUDA GPU acceleration |
| v0.5 | planned | MLE inference, particle filter |
| v0.6 | planned | Rough Heston / rough Bergomi surface calibration |
| v1.0 | planned | Stable API, JOSS paper |

## License

MIT
