# pathwise — Design Spec

**Date:** 2026-03-23
**Status:** Draft — pending user review
**Remote:** https://github.com/alejandro-soto-franco/pathwise.git

---

## Overview

`pathwise` is a high-performance, composable toolkit for simulation, inference, and calibration of stochastic differential equations (SDEs), including non-Markovian and fractional processes. It is implemented as a Rust core with PyO3 Python bindings and GPU acceleration via CUDA.

**Key differentiators:**
- Non-Markovian SDE support (fractional BM, Volterra processes with arbitrary kernels)
- GPU-accelerated path simulation (CUDA primary, wgpu fallback)
- Full toolkit: simulation + inference + surface calibration
- Functional/composable API in both Rust and Python
- Usable standalone as a Rust crate or as a Python package (`pip install pathwise`)

**Target audiences:**
- Quant finance practitioners (rough vol models, calibration)
- ML researchers (score-based diffusion, stochastic optimal control)
- Mathematical physicists (SPDEs, active matter)

---

## Repository Layout

```
pathwise/
  pathwise-core/         # Rust crate: numerics, GPU kernels, inference
    src/
      lib.rs
      process/           # SDE primitives (bm, fbm, gbm, ou, rough_heston, etc.)
      scheme/            # Numerical schemes (EM, Milstein, RK1.5, NV)
      inference/         # MLE, particle filter, surface calibration
      gpu/               # CUDA kernels (cudarc) + wgpu fallback
      kernel/            # Volterra kernel definitions
  pathwise-py/           # PyO3 bindings crate
    src/
      lib.rs             # #[pymodule] entry point
      bindings/          # Per-module PyO3 wrappers
  python/pathwise/       # Python package (thin wrapper)
    __init__.py
    simulate.py
    calibrate.py
    process.py
  benches/               # Criterion benchmarks
  examples/              # Jupyter notebooks + .py examples
    rough_heston_calibration.ipynb
    fbm_paths.py
    particle_filter_ou.py
  docs/
    superpowers/specs/
      2026-03-23-pathwise-design.md   # this file
  Cargo.toml             # workspace
  pyproject.toml         # maturin build config
  README.md
```

---

## Core Rust Traits

The composable foundation — all SDEs are built from these:

```rust
// State: clone + thread-safe + 'static. Blanket impl covers all concrete types.
trait State: Clone + Send + Sync + 'static {}
impl<T: Clone + Send + Sync + 'static> State for T {}

// Marker traits with blanket impls — any matching closure satisfies them automatically.
trait Drift<S: State>: Fn(&S, f64) -> S + Send + Sync {}
impl<S: State, F: Fn(&S, f64) -> S + Send + Sync> Drift<S> for F {}

trait Diffusion<S: State>: Fn(&S, f64) -> S + Send + Sync {}
impl<S: State, F: Fn(&S, f64) -> S + Send + Sync> Diffusion<S> for F {}

trait Kernel: Fn(f64, f64) -> f64 + Send + Sync {}
impl<F: Fn(f64, f64) -> f64 + Send + Sync> Kernel for F {}

// Markovian SDE: dX = f(X,t)dt + g(X,t)dW
struct SDE<S: State, D: Drift<S>, G: Diffusion<S>> {
    drift: D,
    diffusion: G,
    _state: PhantomData<S>,
}

// Non-Markovian Volterra SDE: dX = f(X,t)dt + g(X,t) int K(t,s) dW(s)
struct VolterraSDE<S: State, D: Drift<S>, G: Diffusion<S>, K: Kernel> {
    drift: D,
    diffusion: G,
    kernel: K,
    _state: PhantomData<S>,
}
```

**Dimensionality scope:** v0.1 and v0.2 target scalar SDEs only (`S = f64`). For scalar processes, `Diffusion` returning `f64` is correct (scalar noise coefficient). Multi-dimensional support (`S = nalgebra::DVector<f64>`, diffusion as a matrix) is deferred to v0.3. The Rust API returns `ndarray::Array2<f64>` of shape `(n_paths, n_steps+1)` from `simulate`; the PyO3 layer converts this to a NumPy array.

Users compose SDEs from plain closures. In Python, drift and diffusion accept Python callables via PyO3.

---

## Built-in Process Primitives

Constructors for common processes (extracted/generalized from volterra + malliavin):

| Constructor | Process | Source |
|---|---|---|
| `bm()` | Standard Brownian motion | fresh |
| `gbm(mu, sigma)` | Geometric Brownian motion | fresh |
| `ou(theta, mu, sigma)` | Ornstein-Uhlenbeck | fresh |
| `fbm(hurst)` | Fractional BM (Hosking/Cholesky) | volterra |
| `rough_heston(H, lambda, nu, rho, V0)` | Rough Heston vol model | malliavin |
| `rough_bergomi(H, xi, eta, rho)` | Rough Bergomi vol model | malliavin |
| `volterra(kernel, drift, diffusion)` | General Volterra SDE | volterra |

---

## Numerical Schemes

Scheme is passed as a parameter — never baked into the model:

| Scheme | Order | Best for |
|---|---|---|
| `euler()` | Strong 0.5 | Fast baseline, any SDE |
| `milstein()` | Strong 1.0 | Markovian, smooth coefficients |
| `rk15()` | Strong 1.5 | Smooth Markovian |
| `ninomiya_victoir()` | Weak 2.0 | Smooth Markovian (e.g., rough Heston after Markovianization via auxiliary fields) |
| `bfg()` (Bayer-Friz-Gatheral) | Strong H+0.5 | Non-Markovian Volterra SDEs; Bennedsen et al. hybrid scheme |

---

## GPU Backend

Path simulation is embarrassingly parallel — each path is independent.

- **Primary:** CUDA via `cudarc` crate. RTX 5060 is the reference device.
- **Fallback:** `wgpu` (Metal/Vulkan/WebGPU-compatible)
- **CPU:** `rayon` thread pool (always available)

Feature flags in `Cargo.toml` (Cargo 2021 edition `dep:` syntax to avoid name collision):
```toml
[features]
default = []
backend-cuda = ["dep:cudarc"]
backend-wgpu = ["dep:wgpu"]
```

GPU is used for:
- Monte Carlo path batch simulation
- Quasi-Monte Carlo Sobol sequence generation
- Particle filter resampling step

---

## Inference + Calibration

Three inference modes:

### 1. MLE
Applies to **Markovian SDEs only** (bm, gbm, ou, and Markovianized forms). Log-likelihood via Euler discretization of the Gaussian transition density, optimized with L-BFGS via the `argmin` crate. For non-Markovian processes (fBm, Volterra SDEs), MLE is not available — use particle filter or surface calibration instead.

### 2. Particle Filter (Sequential Monte Carlo)
Online state estimation for filtering/smoothing. Resampling step GPU-accelerated. Observation model: Gaussian `p(y_t | x_t) = N(x_t, sigma_obs^2)` with user-supplied `sigma_obs`. Future versions will accept a callable likelihood `p(y | x)` for non-Gaussian observations.

### 3. Surface Calibration
Fit rough Heston or rough Bergomi to a market implied vol surface. Extracted from malliavin's calibration module. Uses Monte Carlo pricing + L-BFGS via `argmin` (gradient-free Nelder-Mead fallback for non-smooth surfaces). Input: `(strikes, maturities, market_vols)` arrays. NV scheme uses auxiliary field Markovianization following Gatheral-Radoicic-Rosenbaum.

---

## Python API

Functional and composable — no class hierarchy required for standard use:

```python
import pathwise as pw
import numpy as np

# 1. Compose a custom SDE from callables
drift     = lambda x, t: -0.5 * x
diffusion = lambda x, t: 1.0
sde = pw.sde(drift, diffusion)

# 2. Use a built-in process
model = pw.rough_heston(H=0.1, lambda_=0.3, nu=0.3, rho=-0.7, V0=0.02)

# 3. Simulate paths (GPU if available)
paths = pw.simulate(model, scheme=pw.euler(), n_paths=50_000, n_steps=1_000, T=1.0, device="cuda")
# paths: np.ndarray of shape (n_paths, n_steps+1) — includes t=0

# 4. Calibrate to market data
result = pw.calibrate(model, market_vols=vols, strikes=K, maturities=T)

# 5. Particle filter
filtered = pw.particle_filter(sde, observations=obs, n_particles=1000)
```

Drift/diffusion callables can be Python functions or lambdas. NumPy arrays are the standard return type. PyTorch tensor support is deferred to v0.4 to avoid scope creep in v0.1.

---

## Error Handling

`pathwise-core` uses `thiserror` for a typed error enum:

```rust
#[derive(thiserror::Error, Debug)]
pub enum PathwiseError {
    #[error("invalid SDE parameters: {0}")]
    InvalidParameters(String),
    #[error("numerical divergence at step {step}: value={value}")]
    NumericalDivergence { step: usize, value: f64 },
    #[error("GPU error: {0}")]
    GpuError(String),          // wraps cudarc/wgpu errors
    #[error("inference failed to converge: {0}")]
    ConvergenceFailure(String),
}
```

In PyO3 bindings, `PathwiseError` maps to Python exception types:
- `InvalidParameters` -> `ValueError`
- `NumericalDivergence` -> `ArithmeticError` (custom subclass `pathwise.NumericalDivergence`)
- `GpuError` -> `RuntimeError`
- `ConvergenceFailure` -> `RuntimeError` (custom subclass `pathwise.ConvergenceError`)

GPU errors are non-recoverable at the path level; the simulation aborts and returns the error. CPU fallback is the user's responsibility (pass `device="cpu"`).

---

## CI/CD

- **Platform:** GitHub Actions
- **Python versions tested:** 3.10, 3.11, 3.12
- **Build tool:** `maturin` (generates wheels via `maturin build --features backend-cuda` on GPU runner)
- **CUDA toolkit:** 12.x (matches RTX 5060 driver requirements)
- **Test matrix:**
  - CPU-only tests: run on every PR (ubuntu-latest, no GPU required)
  - CUDA tests: run on self-hosted GPU runner (RTX 5060), triggered on push to `main`
  - wgpu tests: run on ubuntu-latest with software renderer (Lavapipe)
- **Publish:** `maturin publish` to PyPI on version tag; `cargo publish` for `pathwise-core` to crates.io

---

## Publishing + Versioning

- `pathwise-core` published to crates.io as a standalone Rust crate (SemVer)
- `pathwise` published to PyPI as a Python package (`pip install pathwise`) built via maturin
- Rust and Python API versions are kept in sync (same version tag)
- SemVer: breaking changes to the Rust trait API or Python function signatures bump the minor version until v1.0, then major

---

## Data Sources / Extraction Plan

No private code is open-sourced directly. The math and algorithms are extracted, re-implemented cleanly, and generalized:

| Private source | What is extracted (as concept, not code) |
|---|---|
| `~/volterra/` | Volterra kernel math, fBm simulation via Hosking method |
| `~/malliavin/` | Rough Heston parameterization, implied vol surface calibration objective |

Both private repos remain unchanged.

---

## Testing Strategy

- Unit tests: each scheme on scalar OU process with known moments
- Integration tests: rough Heston calibration recovers synthetic parameters
- Benchmarks: Criterion for CPU/GPU path throughput (paths/sec at N=10k, 100k, 1M)
- Property tests: strong convergence order verified numerically for each scheme

---

## Release Plan

1. `v0.1.0` — CPU simulation only: EM + Milstein, standard processes, Python bindings
2. `v0.2.0` — fBm + Volterra processes, Ninomiya-Victoir scheme
3. `v0.3.0` — CUDA GPU backend
4. `v0.4.0` — Inference: MLE + particle filter
5. `v0.5.0` — Surface calibration (rough Heston + rough Bergomi)
6. `v1.0.0` — Stable API, JOSS paper submission

---

## Publication Plan

- **JOSS** (*Journal of Open Source Software*): short software paper on `pathwise` as a tool. Fast peer review, guaranteed citation.
- **SIAM Journal on Scientific Computing** or **Quantitative Finance**: longer methods paper on the non-Markovian simulation + calibration algorithms.
