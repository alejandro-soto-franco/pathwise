# Changelog

All notable changes to pathwise are documented here.

---

## [0.2.0] - 2026-03-25

### Added

- **SRI scheme** (`pathwise::sri()`): strong-order 1.5 Stochastic Runge-Kutta (Kloeden-Platen Taylor 1.5) for scalar SDEs. Produces lower strong error than Milstein at the same step count for smooth diffusion coefficients.
- **CIR process** (`pathwise::cir(kappa, theta, sigma)`): Cox-Ingersoll-Ross mean-reverting positive process. Enforces strict Feller condition `2*kappa*theta > sigma^2`; raises `ValueError` otherwise. Full-truncation ensures non-negative paths.
- **Heston stochastic volatility** (`pathwise::heston(mu, kappa, theta, xi, rho)`): correlated log-price/variance system. Returns `(n_paths, n_steps+1, 2)` array where `[:, :, 0]` is log-price and `[:, :, 1]` is variance.
- **Correlated OU** (`pathwise::corr_ou(theta, mu, sigma_flat)`): N-dimensional Ornstein-Uhlenbeck with full covariance matrix. Python bindings currently support N=2; use the Rust API (`pathwise_core::corr_ou`) for higher dimensions.
- **`pathwise-geo` crate**: Riemannian manifold SDE simulation backed by [cartan](https://github.com/alejandro-soto-franco/cartan). Provides `GeodesicEuler`, `GeodesicMilstein`, `GeodesicSRI` schemes and `manifold_simulate` for SDEs on S^n, SO(n), and SPD(n). Rust-only; Python bindings planned for v0.3.
- **`simulate_nd`**: multi-dimensional simulation returning `Array3<f64>` of shape `(n_paths, n_steps+1, N)`.
- **`MilsteinNd`**: diagonal Levy area correction for nD SDEs.
- **`Increment<B>` / `NoiseIncrement` / `State` traits**: foundation for the unified scheme API. `Increment` carries both `dW` and `dZ` (the Levy area), enabling SRI without resampling.
- Error variants: `FellerViolation`, `DimensionMismatch`.

### Changed

- `Scheme<S>` trait redesigned: `step` now takes `&Increment<Self::Noise>` instead of a bare `dw: f64`, enabling higher-order schemes.
- `EulerMaruyama` is now generic over both `f64` and `SVector<f64, N>` state types.
- `simulate` return type is `PyObject` in the Python layer to accommodate both 2D scalar and 3D nD outputs.

---

## [0.1.0] - 2026-03-01

Initial release.

- `pathwise-core`: `EulerMaruyama`, `Milstein` schemes; `bm`, `gbm`, `ou` processes; `simulate` returning `Array2<f64>`.
- `pathwise-py`: Python bindings via PyO3; `euler()`, `milstein()`, `bm()`, `gbm()`, `ou()`, `sde()`, `simulate()`.
- Parallel CPU simulation via Rayon for built-in processes; GIL-bound serial path for custom Python callables.
- Statistical test suite: convergence orders, BM/GBM/OU moment verification, Black-Scholes pricing.
