# pathwise Batch Update Design

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Extend pathwise with multi-dimensional SDE support, the SRI strong-order-1.5 scheme, new processes (CIR, Heston, CorrOU), and a new `pathwise-geo` crate for Riemannian manifold SDEs backed by cartan.

**Architecture:** Generic `Scheme<S>` trait over state type `S`; const-generic nalgebra types for nD state and noise; `pathwise-geo` holds the cartan dependency so flat users pay no geometry overhead.

**Tech stack:** Rust 2021, nalgebra 0.33, ndarray, rayon, rand/rand_distr, PyO3; cartan-core (pathwise-geo only).

---

## Section 1: Core Trait Redesign

### Increment struct

```rust
/// Raw Brownian increments passed to every scheme step.
/// `dz` = integral_0^h W(s) ds — used by SRI; silently ignored by Euler and Milstein.
pub struct Increment<B> {
    pub dw: B,
    pub dz: B,
}
```

`dz` generation formula (given `dw = z1 * sqrt_dt`, independent `z2 ~ N(0,1)`):

```
dz = (dt / 2) * dw - sqrt(dt^3 / 12) * z2
```

`simulate` always generates both `dw` and `dz`; schemes that do not use `dz` receive it and ignore it (zero runtime cost after inlining).

### Scheme trait

```rust
pub trait Scheme<S: State>: Send + Sync {
    type Noise: Send + Sync;

    fn step<D, G>(
        &self,
        drift: &D,
        diffusion: &G,
        x: &S,
        t: f64,
        dt: f64,
        inc: &Increment<Self::Noise>,
    ) -> S
    where
        D: Drift<S>,
        G: Diffusion<S, Self::Noise>;
}
```

### Diffusion trait

`Diffusion<S, B>` returns a diffusion coefficient type that implements `Mul<B, Output = S>`:

| Case | S | B | Diffusion return |
|---|---|---|---|
| Scalar | `f64` | `f64` | `f64` |
| nD diagonal | `SVector<f64, N>` | `SVector<f64, N>` | `SVector<f64, N>` (element-wise) |
| nD full correlated | `SVector<f64, N>` | `SVector<f64, M>` | `SMatrix<f64, N, M>` |

### Scheme-order policy

| Scheme | Scalar | nD diagonal | nD full correlated |
|---|---|---|---|
| EulerMaruyama | order 0.5 | order 0.5 | order 0.5 |
| Milstein | order 1.0 | order 1.0 | order 1.0 (commutative) / 0.5 (non-commutative, documented) |
| SRI | order 1.5 | order 1.5 | **not implemented** — requires Levy area (future work, documented) |

Milstein for nD full correlated uses only diagonal Levy area terms
(`I_{jk} = (dW_j^2 - dt) * delta_{jk}`); cross terms dropped. This is strong order 1.0
when the commutativity condition `g * dg/dx = dg/dx * g` holds and degrades to 0.5
otherwise. The condition is documented and a `commutative: bool` field on the `Milstein`
struct carries an assertion.

---

## Section 2: pathwise-core Changes

### New error variants

`PathwiseError` gains:
- `FellerViolation(String)` — CIR Feller condition `sigma^2 / (2 * kappa * theta) < 1` not met; simulation continues with zero-clipping but emits this error as a warning-level concern if the caller checks.
- `DimensionMismatch(String)` — diffusion matrix shape incompatible with noise or state dimensions.

### New processes (process/markov.rs)

**CIR** (Cox-Ingersoll-Ross):
```
dX = kappa * (theta - X) dt + sigma * sqrt(X) dW
```
- Scalar; constructor `cir(kappa, theta, sigma) -> SDE<f64, ...>`.
- `simulate` clips `x` to `0.0` (not NaN) when discretization produces negative values.
- Feller condition `sigma^2 < 2 * kappa * theta` checked at construction; returns `Err(FellerViolation)` if violated (caller may ignore and proceed with clipping).

**Heston**:
```
d(log S) = (mu - V/2) dt + sqrt(V) dW1
dV       = kappa * (theta - V) dt + xi * sqrt(V) * (rho * dW1 + sqrt(1-rho^2) * dW2)
```
- 2D, `S = SVector<f64, 2>`, `B = SVector<f64, 2>`.
- Constructor `heston(mu, kappa, theta, xi, rho) -> SDE<SVector<2>, ...>`.
- Correlated increments generated via Cholesky in `simulate`: `dW1 = z1*sqrt_dt`, `dW2 = (rho*z1 + sqrt(1-rho^2)*z2)*sqrt_dt`.
- Variance component clipped to `0.0` after each step (full truncation scheme).

**CorrOU** (correlated Ornstein-Uhlenbeck):
```
dX = Theta * (mu - X) dt + L dW
```
- N-dimensional; `L = chol(Sigma)` computed at construction.
- Constructor `corr_ou<const N: usize>(theta_diag, mu, sigma_matrix) -> SDE<SVector<N>, ...>`.

### Updated simulate

```rust
pub fn simulate<S, SC>(
    drift: &dyn Drift<S>,
    diffusion: &dyn Diffusion<S, SC::Noise>,
    scheme: &SC,
    x0: S,
    t0: f64,
    t1: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Result<PathOutput<S>, PathwiseError>
where
    S: State,
    SC: Scheme<S>,
```

`PathOutput<S>` is an enum:
- `PathOutput::Scalar(Array2<f64>)` — shape `(n_paths, n_steps+1)` when `S = f64`
- `PathOutput::Vector(Array3<f64>)` — shape `(n_paths, n_steps+1, N)` when `S = SVector<N>`

Convenience wrappers `simulate_scalar` and `simulate_nd` return the inner arrays directly.

Increment generation per step:
```rust
let z1: f64 = normal.sample(&mut rng);
let z2: f64 = normal.sample(&mut rng);
let dw = z1 * sqrt_dt;
let dz = (dt / 2.0) * dw - (dt.powi(3) / 12.0).sqrt() * z2;
let inc = Increment { dw, dz };
```

For nD, each noise component gets its own independent `(z1, z2)` pair; Cholesky
correlation (for Heston / CorrOU) is applied to the `dw` vector after generation.

---

## Section 3: pathwise-geo (new crate)

### Workspace addition

```toml
# Cargo.toml (workspace members)
members = ["pathwise-core", "pathwise-geo", "pathwise-py"]

# pathwise-geo/Cargo.toml
[dependencies]
pathwise-core = { path = "../pathwise-core" }
cartan-core   = { path = "../../cartan/cartan-core", version = "0.1" }
nalgebra      = { version = "0.33", default-features = false }
rayon         = "1"
```

### ManifoldSDE struct

```rust
pub struct ManifoldSDE<M: Manifold, D, G> {
    pub manifold: M,
    pub drift: D,      // Fn(&M::Point, f64) -> M::Tangent
    pub diffusion: G,  // Fn(&M::Point, f64) -> M::Tangent (scalar noise) or SMatrix (nD)
}
```

`M::Point` and `M::Tangent` are the associated types from cartan's `Manifold` trait.

### Schemes

**GeodesicEuler** (strong order 0.5, Stratonovich):
```
x_{n+1} = exp_{x_n}( f(x_n, t)*dt + g(x_n, t)*dW )
```
Uses `cartan_core::Manifold::exp`. Keeps state exactly on the manifold at every step.

**GeodesicMilstein** (strong order 1.0, scalar noise):
Adds Stratonovich-to-Ito correction via cartan's `Connection` trait:
```
correction = -0.5 * Gamma(x_n)(g, g) * dt
```
where `Gamma` is computed via `cartan_core::Connection::riemannian_hessian_vector_product`.
Full step:
```
x_{n+1} = exp_{x_n}( (f - 0.5*Gamma(g,g))*dt + g*dW + 0.5*g*Dg*g*(dW^2 - dt) )
```

**GeodesicSRI** (strong order 1.5, scalar noise):
Extends GeodesicMilstein with the `dz` term. Requires `Manifold + Connection + Curvature`.

### Process constructors

```rust
// Pure diffusion, zero drift
pub fn brownian_motion_on<M: Manifold>(manifold: M) -> ManifoldSDE<M, ...>;

// Mean-reverting toward mu_point via log map as drift direction
pub fn ou_on<M: Manifold>(manifold: M, kappa: f64, mu_point: M::Point) -> ManifoldSDE<M, ...>;
```

### manifold_simulate

```rust
pub fn manifold_simulate<M, SC>(
    sde: &ManifoldSDE<M, ...>,
    scheme: &SC,
    x0: M::Point,
    t0: f64,
    t1: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Vec<Vec<M::Point>>
```

Returns `Vec<Vec<M::Point>>` (outer: paths, inner: time steps). A helper
`paths_to_array(paths, manifold, ref_point)` flattens by projecting each point onto
the tangent space at `ref_point` via `log`, yielding `Array3<f64>` of shape
`(n_paths, n_steps+1, dim)` for downstream numerical use.

---

## Section 4: pathwise-py Changes

### Internal dispatch

`py_simulate` replaces `is_instance_of` boolean checks with internal enums:

```rust
enum SchemeKind { Euler, Milstein, Sri }
enum ProcessKind {
    Bm,
    Gbm { mu: f64, sigma: f64 },
    Ou  { theta: f64, mu: f64, sigma: f64 },
    Cir { kappa: f64, theta: f64, sigma: f64 },
    Heston { mu: f64, kappa: f64, theta: f64, xi: f64, rho: f64 },
}
```

Invalid combinations raise `ValueError` with an explicit message:
```
"SRI requires scalar or diagonal noise; use Milstein for Heston"
```

### New Python API

```python
# Schemes
pathwise.sri()

# Processes
pathwise.cir(kappa, theta, sigma)
pathwise.heston(mu, kappa, theta, xi, rho)

# Output shapes
# simulate(gbm(...),    scheme, ...) -> np.ndarray (n_paths, n_steps+1)
# simulate(heston(...), scheme, ...) -> np.ndarray (n_paths, n_steps+1, 2)

# Manifold simulation
import pathwise.geo as geo

geo.brownian_motion_on(manifold)   # manifold is a cartan manifold object via pathwise-geo FFI
geo.ou_on(manifold, kappa, mu_arr) # mu_arr: np.ndarray matching manifold point shape
geo.simulate(sde, scheme, n_paths, n_steps, t1, x0_arr)
# returns list[np.ndarray] of shape (n_steps+1, dim) per path
```

Custom nD callables (serial path): `SDEKind::CustomNd` accepts Python callables
`drift(x: np.ndarray, t: float) -> np.ndarray` and
`diffusion(x: np.ndarray, t: float) -> np.ndarray` (diagonal noise only).

---

## Section 5: Testing

### pathwise-core (tests/convergence.rs)

- `sri_strong_order_on_gbm`: log-log regression, expected order in `[1.2, 1.8]`, 8000 paths, step counts `[25, 50, 100, 200, 400]`.
- `sri_stronger_than_milstein_strong`: SRI error < Milstein error at N=50 steps.
- `cir_stays_nonnegative`: 1000 paths, Feller condition marginally violated, assert all values `>= 0`.
- `cir_mean_exact`: `E[X_T] = theta + (x0 - theta)*exp(-kappa*T)`, within 2% at 20k paths.
- `heston_variance_nonnegative`: all variance column values `>= 0`.
- `heston_output_shape`: `Array3` shape `(n_paths, n_steps+1, 2)`.
- `corr_ou_covariance`: simulate 2D CorrOU to stationarity, sample covariance within 5% of `Sigma / (2*theta)`.
- `euler_nd_diagonal_matches_scalar`: diagonal 2D Euler matches two independent scalar Euler runs path-by-path with same seed.

### pathwise-geo (pathwise-geo/tests/manifold_sde.rs)

- `geodesic_euler_stays_on_sphere`: 500 paths on S², all points satisfy `|x| approx 1` to 1e-6.
- `geodesic_euler_stays_on_so3`: all path points satisfy `R^T R approx I` and `det(R) approx 1` to 1e-6.
- `ou_on_spd_stays_positive_definite`: all SPD path points pass eigenvalue positivity check.
- `ou_on_manifold_mean_reverts`: OU on S² converges toward `mu_point` over long time, verified by geodesic distance decreasing on average.
- `paths_to_array_shape`: shape matches `(n_paths, n_steps+1, dim)`.

### pathwise-py (tests/test_schemes.py)

- `test_sri_convergence_python`: two step counts, error decreases.
- `test_heston_output_shape`: `simulate(heston(...), milstein(), ...).shape == (n_paths, n_steps+1, 2)`.
- `test_sri_on_heston_raises`: `ValueError` raised.
- `test_cir_nonnegative`: all output values `>= 0`.

---

## Out of Scope (Future Work)

- **Levy area approximation for SRI on non-commutative nD systems** (Wiktorsson 2001 truncated series). Required to upgrade correlated Milstein to strong order 1.5.
- **Adaptive step size control** (SOSRI error estimator from Rossler 2010).
- **Jump-diffusion / compound Poisson processes** (Merton model).
- **GPU ensemble parallelism** (DiffEqGPU-style wgpu backend).
- **Full-matrix custom nD Python callables** (only diagonal diffusion supported in custom path).
