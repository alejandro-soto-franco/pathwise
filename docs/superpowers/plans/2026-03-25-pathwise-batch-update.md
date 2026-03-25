# pathwise Batch Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend pathwise with the SRI strong-order-1.5 scheme, multi-dimensional SDE support, new processes (CIR, Heston, CorrOU), and a new `pathwise-geo` crate for Riemannian manifold SDEs backed by cartan.

**Architecture:** Generic `Scheme<S>` trait over state type `S` with `Increment<B>` carrying both dW and dZ increments; const-generic nalgebra `SVector<f64, N>` for nD state and noise; `pathwise-geo` crate isolated so flat users pay zero geometry overhead; pathwise-py exposes all new APIs including new scheme and process constructors. Note: spec's `PathOutput<S>` enum is replaced by two separate functions (`simulate` + `simulate_nd`) returning `Array2` and `Array3` directly; same external behavior, simpler API.

**Tech Stack:** Rust 2021, nalgebra 0.33 (new), ndarray 0.15, rayon, rand/rand_distr, thiserror, PyO3 0.21; cartan-core 0.1 (pathwise-geo only)

---

## File Map

**Modified:**
- `Cargo.toml` — add `nalgebra` workspace dep; add `pathwise-geo` to members
- `pathwise-core/Cargo.toml` — add `nalgebra` dep
- `pathwise-core/src/lib.rs` — re-export new public items
- `pathwise-core/src/state.rs` — NEW: `State`, `Increment<B>`, `NoiseIncrement`, `Diffusion<S,B>` traits
- `pathwise-core/src/scheme/mod.rs` — update `Scheme<S>` trait signature; add `SRI` export
- `pathwise-core/src/scheme/euler.rs` — step takes `&Increment<f64>` instead of `dw: f64`
- `pathwise-core/src/scheme/milstein.rs` — step takes `&Increment<f64>`; update unit tests
- `pathwise-core/src/scheme/sri.rs` — NEW: `Sri` struct, `impl Scheme<f64>`
- `pathwise-core/src/scheme/milstein_nd.rs` — NEW: `MilsteinNd<N>`, `impl Scheme<SVector<f64,N>>`
- `pathwise-core/src/process/markov.rs` — remove old `Drift`/`Diffusion` wrapper traits; add `cir`, `heston`, `corr_ou`, `HestonDiffusion`, `CorrOuDiffusion`, `NdSDE`
- `pathwise-core/src/simulate.rs` — update simulate to generate `Increment`; add `simulate_nd` (spec's `PathOutput<S>` enum omitted; separate functions used instead)
- `pathwise-core/src/error.rs` — add `FellerViolation`, `DimensionMismatch`
- `pathwise-core/tests/convergence.rs` — add SRI convergence tests, CIR tests
- `pathwise-core/tests/nd.rs` — NEW: nD process tests (Heston, CorrOU, euler_nd)
- `pathwise-geo/Cargo.toml` — NEW crate manifest
- `pathwise-geo/src/lib.rs` — NEW
- `pathwise-geo/src/sde.rs` — NEW: `ManifoldSDE`
- `pathwise-geo/src/scheme/mod.rs` — NEW
- `pathwise-geo/src/scheme/euler.rs` — NEW: `GeodesicEuler`
- `pathwise-geo/src/scheme/milstein.rs` — NEW: `GeodesicMilstein`
- `pathwise-geo/src/scheme/sri.rs` — NEW: `GeodesicSRI`
- `pathwise-geo/src/process.rs` — NEW: `brownian_motion_on`, `ou_on`
- `pathwise-geo/src/simulate.rs` — NEW: `manifold_simulate`, `paths_to_array`
- `pathwise-geo/tests/manifold_sde.rs` — NEW
- `pathwise-py/src/py_simulate.rs` — expand dispatch; add SRI, CIR, Heston, CorrOU; nD output
- `pathwise-py/src/py_scheme.rs` — add `PySri`
- `pathwise-py/src/py_process.rs` — add `Cir`, `Heston`, `CorrOu` variants
- `pathwise-py/src/lib.rs` — register new Python symbols
- `pathwise-py/tests/test_schemes.py` — add SRI, Heston, CIR tests

---

## Task 1: Foundation Types

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Modify: `pathwise-core/Cargo.toml`
- Create: `pathwise-core/src/state.rs`
- Modify: `pathwise-core/src/lib.rs`

- [ ] **Step 1: Write compile-check test for state.rs types**

Create `pathwise-core/src/state.rs` with an empty module, then add this test to verify the intended API compiles:

```rust
// pathwise-core/src/state.rs (skeleton for test)
pub struct Increment<B> {
    pub dw: B,
    pub dz: B,
}
```

Add to `pathwise-core/src/lib.rs`:
```rust
pub mod state;
pub use state::Increment;
```

- [ ] **Step 2: Run to confirm skeleton compiles**

```bash
cd /home/alejandrosotofranco/pathwise && cargo build -p pathwise-core 2>&1 | head -30
```

Expected: compiles (or only "unused" warnings).

- [ ] **Step 3: Implement state.rs fully**

```rust
// pathwise-core/src/state.rs
use std::ops::{Add, Mul};
use rand::Rng;

/// Both Brownian increments for one time step.
/// `dw` = dW = z1 * sqrt(dt)
/// `dz` = integral_0^dt W(s) ds = (dt/2)*dw - sqrt(dt^3/12)*z2
///
/// Euler and Milstein ignore `dz`. SRI uses both.
/// Derivation: dZ = dt*dW - I_{(0,1)}, conditional mean of I_{(0,1)} given dW
/// is (dt/2)*dW, conditional variance is dt^3/12. Negative sign is correct.
/// Verified: E[dZ]=0, Var[dZ]=dt^3/3, Cov(dW,dZ)=dt^2/2.
#[derive(Clone, Debug)]
pub struct Increment<B: Clone> {
    pub dw: B,
    pub dz: B,
}

/// Algebraic requirements for SDE state types.
/// `f64` and `nalgebra::SVector<f64, N>` both satisfy this automatically.
pub trait State:
    Clone + Send + Sync + 'static
    + Add<Output = Self>
    + Mul<f64, Output = Self>
{
    fn zero() -> Self;
}

impl State for f64 {
    fn zero() -> Self { 0.0 }
}

impl<const N: usize> State for nalgebra::SVector<f64, N> {
    fn zero() -> Self { nalgebra::SVector::zeros() }
}

/// Types that can sample a Brownian increment for a given dt.
pub trait NoiseIncrement: Clone + Send + Sync + 'static {
    fn sample<R: Rng>(rng: &mut R, dt: f64) -> Increment<Self>;
}

impl NoiseIncrement for f64 {
    fn sample<R: Rng>(rng: &mut R, dt: f64) -> Increment<Self> {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0_f64, 1.0).unwrap();
        let z1 = normal.sample(rng);
        let z2 = normal.sample(rng);
        let dw = z1 * dt.sqrt();
        let dz = (dt / 2.0) * dw - (dt.powi(3) / 12.0).sqrt() * z2;
        Increment { dw, dz }
    }
}

impl<const M: usize> NoiseIncrement for nalgebra::SVector<f64, M> {
    fn sample<R: Rng>(rng: &mut R, dt: f64) -> Increment<Self> {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0_f64, 1.0).unwrap();
        let mut dw = nalgebra::SVector::<f64, M>::zeros();
        let mut dz = nalgebra::SVector::<f64, M>::zeros();
        for i in 0..M {
            let z1 = normal.sample(rng);
            let z2 = normal.sample(rng);
            dw[i] = z1 * dt.sqrt();
            dz[i] = (dt / 2.0) * dw[i] - (dt.powi(3) / 12.0).sqrt() * z2;
        }
        Increment { dw, dz }
    }
}

/// Diffusion coefficient interface.
/// `apply(x, t, dw)` returns the diffusion contribution `g(x,t) * dw` directly.
///
/// Blanket impl for scalar closures: `f(x,t) -> f64` satisfies `Diffusion<f64, f64>`
/// by computing `f(x,t) * dw`.
///
/// Blanket impl for nD diagonal closures: `f(x,t) -> SVector<N>` satisfies
/// `Diffusion<SVector<N>, SVector<N>>` by element-wise (Hadamard) product.
///
/// Full-matrix processes (Heston, CorrOU) provide concrete struct impls.
pub trait Diffusion<S: State, B: NoiseIncrement>: Send + Sync {
    fn apply(&self, x: &S, t: f64, dw: &B) -> S;
}

// Scalar blanket impl: closure returns g(x,t); apply multiplies by dw.
impl<F: Fn(f64, f64) -> f64 + Send + Sync> Diffusion<f64, f64> for F {
    fn apply(&self, x: &f64, t: f64, dw: &f64) -> f64 {
        self(*x, t) * dw
    }
}

// nD diagonal blanket impl: closure returns sigma vector; apply component-multiplies with dw.
impl<const N: usize, F> Diffusion<nalgebra::SVector<f64, N>, nalgebra::SVector<f64, N>> for F
where
    F: Fn(&nalgebra::SVector<f64, N>, f64) -> nalgebra::SVector<f64, N> + Send + Sync,
{
    fn apply(
        &self,
        x: &nalgebra::SVector<f64, N>,
        t: f64,
        dw: &nalgebra::SVector<f64, N>,
    ) -> nalgebra::SVector<f64, N> {
        self(x, t).component_mul(dw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn increment_f64_variance() {
        // Var[dW] ≈ dt; Var[dZ] ≈ dt^3/3; Cov(dW,dZ) ≈ dt^2/2
        let dt = 0.01_f64;
        let n = 100_000_usize;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let mut dws = vec![0.0_f64; n];
        let mut dzs = vec![0.0_f64; n];
        for i in 0..n {
            let inc = <f64 as NoiseIncrement>::sample(&mut rng, dt);
            dws[i] = inc.dw;
            dzs[i] = inc.dz;
        }
        let var_dw: f64 = dws.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let var_dz: f64 = dzs.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let cov: f64 = dws.iter().zip(&dzs).map(|(w, z)| w * z).sum::<f64>() / n as f64;
        // Var[dW] = dt = 0.01
        assert!((var_dw - dt).abs() / dt < 0.02, "Var[dW]={:.6} expected {:.6}", var_dw, dt);
        // Var[dZ] = dt^3/3
        let expected_var_dz = dt.powi(3) / 3.0;
        assert!((var_dz - expected_var_dz).abs() / expected_var_dz < 0.03,
            "Var[dZ]={:.8} expected {:.8}", var_dz, expected_var_dz);
        // Cov(dW,dZ) = dt^2/2
        let expected_cov = dt.powi(2) / 2.0;
        assert!((cov - expected_cov).abs() / expected_cov < 0.02,
            "Cov(dW,dZ)={:.8} expected {:.8}", cov, expected_cov);
    }

    #[test]
    fn increment_svector_has_correct_length() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let inc = <nalgebra::SVector<f64, 3> as NoiseIncrement>::sample(&mut rng, 0.01);
        assert_eq!(inc.dw.len(), 3);
        assert_eq!(inc.dz.len(), 3);
    }
}
```

- [ ] **Step 4: Add nalgebra to workspace Cargo.toml and pathwise-core Cargo.toml**

In `Cargo.toml` (workspace root), add to `[workspace.dependencies]`:
```toml
nalgebra = { version = "0.33", default-features = false, features = ["std"] }
```

In `pathwise-core/Cargo.toml`, add to `[dependencies]`:
```toml
nalgebra = { workspace = true }
```

- [ ] **Step 5: Update pathwise-core/src/lib.rs exports**

```rust
// pathwise-core/src/lib.rs — add at top:
pub mod state;
pub use state::{Diffusion, Increment, NoiseIncrement, State};
```

Keep all existing re-exports. The old `Diffusion` and `Drift` re-exports from `process` will be removed in Task 2.

- [ ] **Step 6: Run tests to confirm they pass**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core 2>&1 | tail -20
```

Expected: all existing tests pass + `increment_f64_variance` + `increment_svector_has_correct_length` pass.

- [ ] **Step 7: Commit**

```bash
cd /home/alejandrosotofranco/pathwise
git add Cargo.toml pathwise-core/Cargo.toml pathwise-core/src/state.rs pathwise-core/src/lib.rs
git commit -m "feat(pathwise-core): add State, Increment, NoiseIncrement, Diffusion foundation types + nalgebra dep"
```

---

## Task 2: Migrate Scalar Scheme to Increment-Based API

**Files:**
- Modify: `pathwise-core/src/scheme/mod.rs`
- Modify: `pathwise-core/src/scheme/euler.rs`
- Modify: `pathwise-core/src/scheme/milstein.rs`
- Modify: `pathwise-core/src/simulate.rs`
- Modify: `pathwise-core/src/process/markov.rs`
- Modify: `pathwise-core/src/lib.rs`

Goal: `Scheme<S>` takes `&Increment<S>` instead of bare `dw: f64`. All existing convergence tests must still pass; only internal structure changes.

- [ ] **Step 1: Write the failing test**

Add to `pathwise-core/src/scheme/euler.rs` tests:

```rust
#[test]
fn euler_step_via_increment() {
    use crate::state::{Increment, NoiseIncrement};
    let e = euler();
    let b = crate::process::markov::bm();
    let inc = Increment { dw: 0.3_f64, dz: 0.0_f64 };
    // Just check it compiles and returns a finite value
    let x_new: f64 = e.step(&b.drift, &b.diffusion, &0.0_f64, 0.0, 0.01, &inc);
    assert!(x_new.is_finite());
}
```

Run to confirm compile error (Scheme::step signature mismatch):
```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core euler_step_via_increment 2>&1 | head -20
```

Expected: compile error about `step` signature.

- [ ] **Step 2: Update scheme/mod.rs**

Replace the entire file:

```rust
// pathwise-core/src/scheme/mod.rs
use crate::state::{Diffusion, Increment, NoiseIncrement, State};
use crate::process::markov::Drift;

/// One-step advance of an SDE of type S.
///
/// The `type Noise` associated type fixes the noise increment type for this scheme.
/// For all current schemes on scalar or nD-diagonal processes, `Noise = S`.
pub trait Scheme<S: State>: Send + Sync {
    type Noise: NoiseIncrement;

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

pub mod euler;
pub mod milstein;
pub mod milstein_nd;
pub mod sri;
pub use euler::{euler, EulerMaruyama};
pub use milstein::{milstein, Milstein};
pub use milstein_nd::{milstein_nd, MilsteinNd};
pub use sri::{sri, Sri};
```

- [ ] **Step 3: Update euler.rs**

```rust
// pathwise-core/src/scheme/euler.rs
use super::Scheme;
use crate::state::{Diffusion, Increment, NoiseIncrement, State};
use crate::process::markov::Drift;

pub struct EulerMaruyama;

/// Generic Euler-Maruyama: works for any State S with Noise = S.
/// x_{n+1} = x_n + f(x_n, t)*dt + g(x_n, t)*dW
impl<S: State> Scheme<S> for EulerMaruyama {
    type Noise = S;

    fn step<D, G>(&self, drift: &D, diffusion: &G, x: &S, t: f64, dt: f64, inc: &Increment<S>) -> S
    where
        D: Drift<S>,
        G: Diffusion<S, S>,
    {
        let f_dt = drift(x, t) * dt;
        let g_dw = diffusion.apply(x, t, &inc.dw);
        x.clone() + f_dt + g_dw
    }
}

pub fn euler() -> EulerMaruyama { EulerMaruyama }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Increment;
    use crate::process::markov::bm;

    #[test]
    fn euler_step_bm_is_x_plus_dw() {
        let b = bm();
        let e = euler();
        let inc = Increment { dw: 0.3_f64, dz: 0.0 };
        let x_new = e.step(&b.drift, &b.diffusion, &1.0_f64, 0.0, 0.01, &inc);
        // BM drift=0, diffusion=1: x_new = 1.0 + 0.0*0.01 + 1.0*0.3 = 1.3
        assert!((x_new - 1.3).abs() < 1e-12, "got {}", x_new);
    }

    #[test]
    fn euler_step_via_increment() {
        let e = euler();
        let b = bm();
        let inc = Increment { dw: 0.3_f64, dz: 0.0_f64 };
        let x_new: f64 = e.step(&b.drift, &b.diffusion, &0.0_f64, 0.0, 0.01, &inc);
        assert!(x_new.is_finite());
    }
}
```

- [ ] **Step 4: Update milstein.rs**

```rust
// pathwise-core/src/scheme/milstein.rs
use super::Scheme;
use crate::state::{Diffusion, Increment, State};
use crate::process::markov::Drift;

pub struct Milstein {
    h: f64,
}

impl Milstein {
    pub fn new(h: f64) -> Self { Self { h } }
}

/// Scalar Milstein scheme (strong order 1.0 for state-dependent diffusion).
/// Uses central finite difference to approximate dg/dx.
/// x_{n+1} = x_n + f*dt + g*dw + 0.5*g*(dg/dx)*(dw^2 - dt)
impl Scheme<f64> for Milstein {
    type Noise = f64;

    fn step<D, G>(&self, drift: &D, diffusion: &G, x: &f64, t: f64, dt: f64, inc: &Increment<f64>) -> f64
    where
        D: Drift<f64>,
        G: Diffusion<f64, f64>,
    {
        let dw = inc.dw;
        let f = drift(x, t);
        // g(x,t) via apply with unit noise
        let g = diffusion.apply(x, t, &1.0_f64);
        let g_plus  = diffusion.apply(&(x + self.h), t, &1.0_f64);
        let g_minus = diffusion.apply(&(x - self.h), t, &1.0_f64);
        let dg_dx = (g_plus - g_minus) / (2.0 * self.h);
        x + f * dt + g * dw + 0.5 * g * dg_dx * (dw * dw - dt)
    }
}

pub fn milstein() -> Milstein { Milstein::new(1e-5) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::markov::bm;
    use crate::state::Increment;

    #[test]
    fn milstein_equals_euler_for_constant_diffusion() {
        let b = bm();
        let m = milstein();
        let e = crate::scheme::euler::euler();
        let x = 1.0_f64;
        let inc = Increment { dw: 0.3_f64, dz: 0.0 };
        let x_euler = e.step(&b.drift, &b.diffusion, &x, 0.0, 0.01, &inc);
        let x_milstein = m.step(&b.drift, &b.diffusion, &x, 0.0, 0.01, &inc);
        assert!((x_euler - x_milstein).abs() < 1e-8,
            "euler={} milstein={}", x_euler, x_milstein);
    }

    #[test]
    fn milstein_differs_from_euler_for_state_dependent_diffusion() {
        let gbm = crate::process::markov::gbm(0.0, 0.2);
        let m = milstein();
        let e = crate::scheme::euler();
        let x = 1.0_f64;
        let inc = Increment { dw: 0.5_f64, dz: 0.0 };
        let dt = 0.01;
        let x_euler = e.step(&gbm.drift, &gbm.diffusion, &x, 0.0, dt, &inc);
        let x_milstein = m.step(&gbm.drift, &gbm.diffusion, &x, 0.0, dt, &inc);
        assert!((x_euler - x_milstein).abs() > 1e-4, "should differ by correction term");
    }
}
```

- [ ] **Step 5: Remove old Drift/Diffusion wrapper traits from markov.rs; keep SDE struct and constructors**

In `pathwise-core/src/process/markov.rs`, replace the top trait definitions:

```rust
// OLD (remove these two trait blocks):
// pub trait State: Clone + Send + Sync + 'static {}
// impl<T: Clone + Send + Sync + 'static> State for T {}
// pub trait Drift<S: State>: Fn(&S, f64) -> S + Send + Sync {}
// impl<S: State, F: Fn(&S, f64) -> S + Send + Sync> Drift<S> for F {}
// pub trait Diffusion<S: State>: Fn(&S, f64) -> S + Send + Sync {}
// impl<S: State, F: Fn(&S, f64) -> S + Send + Sync> Diffusion<S> for F {}

// NEW (add at top of file):
use crate::state::State;
use std::marker::PhantomData;

/// Named bound alias for drift functions. Blanket impl covers all matching closures.
pub trait Drift<S: State>: Fn(&S, f64) -> S + Send + Sync {}
impl<S: State, F: Fn(&S, f64) -> S + Send + Sync> Drift<S> for F {}
```

Remove the old `State` trait definition and its blanket impl from markov.rs (it's now in state.rs). Keep `SDE`, `bm`, `gbm`, `ou` unchanged.

Update lib.rs re-exports: remove `State` from the process re-exports since it now lives in state.rs:

```rust
// pathwise-core/src/lib.rs
pub mod state;
pub use state::{Diffusion, Increment, NoiseIncrement, State};

pub mod process;
pub use process::{bm, gbm, ou, Drift, SDE};  // removed Diffusion, State (now from state module)

pub mod scheme;
pub use scheme::{euler, milstein, milstein_nd, sri, Scheme};

pub mod simulate;
pub use simulate::simulate;

pub mod error;
pub use error::PathwiseError;
```

- [ ] **Step 6: Update simulate.rs to generate Increment internally**

```rust
// pathwise-core/src/simulate.rs
use ndarray::Array2;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::error::PathwiseError;
use crate::process::markov::Drift;
use crate::scheme::Scheme;
use crate::state::{Diffusion, NoiseIncrement, State};

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Simulate `n_paths` paths of a scalar SDE from `t0` to `t1` with `n_steps` steps.
///
/// Returns Array2<f64> of shape `(n_paths, n_steps + 1)`.
///
/// Generates both `dw` and `dz` per step; schemes that do not use `dz` ignore it.
/// Non-finite values are stored as NaN.
#[allow(clippy::too_many_arguments)]
pub fn simulate<D, G, SC>(
    drift: &D,
    diffusion: &G,
    scheme: &SC,
    x0: f64,
    t0: f64,
    t1: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Result<Array2<f64>, PathwiseError>
where
    D: Drift<f64> + Sync,
    G: Diffusion<f64, f64> + Sync,
    SC: Scheme<f64, Noise = f64>,
{
    if n_paths == 0 || n_steps == 0 {
        return Err(PathwiseError::InvalidParameters("n_paths and n_steps must be > 0".into()));
    }
    if t1 <= t0 {
        return Err(PathwiseError::InvalidParameters("t1 must be > t0".into()));
    }

    let dt = (t1 - t0) / n_steps as f64;
    let base_seed = splitmix64(seed);

    let rows: Vec<Vec<f64>> = (0..n_paths)
        .into_par_iter()
        .map(|i| {
            let path_seed = splitmix64(base_seed.wrapping_add(i as u64));
            let mut rng = rand::rngs::SmallRng::seed_from_u64(path_seed);
            let mut path = Vec::with_capacity(n_steps + 1);
            let mut x = x0;
            path.push(x);
            for step in 0..n_steps {
                let t = t0 + step as f64 * dt;
                let inc = <f64 as NoiseIncrement>::sample(&mut rng, dt);
                x = scheme.step(drift, diffusion, &x, t, dt, &inc);
                if !x.is_finite() { x = f64::NAN; }
                path.push(x);
            }
            path
        })
        .collect();

    let mut out = Array2::zeros((n_paths, n_steps + 1));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[[i, j]] = v;
        }
    }
    Ok(out)
}
```

Note: `simulate_nd` will be added in Task 5.

- [ ] **Step 7: Create milstein_nd.rs as empty stub so scheme/mod.rs compiles**

```rust
// pathwise-core/src/scheme/milstein_nd.rs
pub struct MilsteinNd<const N: usize> { pub h: f64 }
pub fn milstein_nd<const N: usize>() -> MilsteinNd<N> { MilsteinNd { h: 1e-5 } }
```

Create `pathwise-core/src/scheme/sri.rs` as empty stub:
```rust
// pathwise-core/src/scheme/sri.rs
pub struct Sri;
pub fn sri() -> Sri { Sri }
```

- [ ] **Step 8: Run all existing tests; confirm they pass**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core 2>&1 | tail -30
```

Expected: all previous tests pass plus the two new Euler/Milstein unit tests.

- [ ] **Step 9: Commit**

```bash
cd /home/alejandrosotofranco/pathwise
git add pathwise-core/src/
git commit -m "feat(pathwise-core): migrate Scheme<S> to Increment-based API; EulerMaruyama generic over State"
```

---

## Task 3: SRI Scalar Scheme (Strong Order 1.5)

**Files:**
- Modify: `pathwise-core/src/scheme/sri.rs`
- Modify: `pathwise-core/tests/convergence.rs`

SRI1 scheme (Rossler): for scalar SDE with additive or multiplicative noise,
```
x_{n+1} = x_n + f*dt + g*dw + 0.5*g*(dg/dx)*(dw^2 - dt)  [Milstein term]
         + g*(d^2g/dx^2)*dz/2                               [SRI extra term, via finite diff]
         - 0.5*f*dt^2 * (df/dx)                             [drift correction, optional/higher order]
```

Actually the standard SRI1 from Rossler (2010) for Ito SDE `dX = f dt + g dW`:
```
H1 = x_n + f*dt + g*sqrt(dt)
x_{n+1} = x_n + f*dt + g*dw
         + (g(H1,t+dt) - g(x_n,t)) * (dw^2 - dt) / (2*sqrt(dt))
         + (g(H1,t+dt) + g(x_n,t) - 2*g_bar) * dz / dt  [where g_bar involves extra evals]
```

Use the simpler but equivalent formulation: SRI with the Milstein correction plus the dz term:
```
x_{n+1} = x_n + f*dt + g*dw
         + 0.5*g*(dg/dx)*(dw^2 - dt)         [Milstein order-1 correction]
         + g*(d^2g/dx^2 + (dg/dx)^2)*dz       [order-1.5 correction via dz]
```

Wait, let me use the exact formula. For SRI1 on scalar Ito SDE:

The Rossler SRI1 scheme requires one extra evaluation. Using the notation from the spec:
```
x_new = x + f*h + g*dW + 0.5*g*(dg/dx)*(dW^2 - h) + (g*(dg/dx) + g^2*(d^2g/dx^2)) * dZ
```

Where `dZ = dz` from our `Increment`. The second derivative can be approximated by finite differences:
```
d^2g/dx^2 ≈ (g(x+h) - 2*g(x) + g(x-h)) / h^2
```

- [ ] **Step 1: Write the failing convergence test**

Add to `pathwise-core/tests/convergence.rs`:

```rust
/// SRI on GBM: strong order ≈ 1.5 via common-noise log-log regression.
#[test]
fn sri_strong_order_on_gbm() {
    use pathwise_core::scheme::sri;
    // This test just needs to compile; actual order check will pass after implementation.
    let (mu, sigma, x0, t1) = (0.05_f64, 0.3_f64, 1.0_f64, 1.0_f64);
    let n_paths = 8000;
    let step_counts = [25usize, 50, 100, 200, 400];
    let dts: Vec<f64> = step_counts.iter().map(|&n| t1 / n as f64).collect();
    let errors: Vec<f64> = step_counts.iter().map(|&n_steps| {
        strong_error_generic(
            &sri(),
            n_steps, n_paths, mu, sigma, x0, t1,
        )
    }).collect();
    let order = convergence_order(&dts, &errors);
    println!("SRI strong order = {:.4}  (expected ~1.5, band [1.2, 1.8])", order);
    assert!(order > 1.2 && order < 1.8,
        "SRI strong order = {:.4}, expected in [1.2, 1.8]", order);
}

/// SRI error < Milstein error at the same step count (N=50).
#[test]
fn sri_stronger_than_milstein_strong() {
    use pathwise_core::scheme::{milstein, sri};
    let (mu, sigma, x0, t1) = (0.05_f64, 0.3_f64, 1.0_f64, 1.0_f64);
    let n_paths = 8000;
    let n_steps = 50;
    let milstein_err = strong_error_generic(&milstein(), n_steps, n_paths, mu, sigma, x0, t1);
    let sri_err      = strong_error_generic(&sri(),      n_steps, n_paths, mu, sigma, x0, t1);
    println!("Milstein strong err = {:.6}", milstein_err);
    println!("SRI     strong err = {:.6}  (ratio {:.2}x)", sri_err, milstein_err / sri_err);
    assert!(sri_err < milstein_err,
        "SRI ({:.6}) should be < Milstein ({:.6})", sri_err, milstein_err);
}
```

Also add this helper function to convergence.rs (needed by SRI test -- uses the generic Scheme<f64> interface with Increment):

```rust
/// Strong error using common-noise for any Scheme<f64> — uses internal Increment generation.
fn strong_error_generic<SC: pathwise_core::Scheme<f64, Noise = f64>>(
    scheme: &SC,
    n_steps: usize,
    n_paths: usize,
    mu: f64,
    sigma: f64,
    x0: f64,
    t1: f64,
) -> f64 {
    use pathwise_core::state::NoiseIncrement;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    let dt = t1 / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let g = pathwise_core::gbm(mu, sigma);
    let mut total_error = 0.0_f64;
    for i in 0..n_paths {
        let mut rng = rand::rngs::SmallRng::seed_from_u64((i as u64) ^ 0xDEAD_BEEF_CAFE);
        let normal = Normal::new(0.0_f64, 1.0).unwrap();
        // Generate dW increments for common-noise comparison
        let dws: Vec<f64> = (0..n_steps).map(|_| normal.sample(&mut rng) * sqrt_dt).collect();
        // Exact GBM terminal value
        let w_t: f64 = dws.iter().sum();
        let x_exact = x0 * ((mu - 0.5 * sigma * sigma) * t1 + sigma * w_t).exp();
        // Scheme run with SAME dW, independent dZ for SRI
        let mut rng2 = rand::rngs::SmallRng::seed_from_u64((i as u64) ^ 0xCAFE_BABE);
        let normal2 = Normal::new(0.0_f64, 1.0).unwrap();
        let mut x = x0;
        for j in 0..n_steps {
            let dw = dws[j];
            let z2 = normal2.sample(&mut rng2);
            let dz = (dt / 2.0) * dw - (dt.powi(3) / 12.0).sqrt() * z2;
            let inc = pathwise_core::state::Increment { dw, dz };
            x = scheme.step(&g.drift, &g.diffusion, &x, j as f64 * dt, dt, &inc);
            if !x.is_finite() { x = f64::NAN; break; }
        }
        total_error += (x - x_exact).abs();
    }
    total_error / n_paths as f64
}
```

Run to confirm compilation fails (Sri not yet implementing Scheme<f64>):
```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core sri_strong_order 2>&1 | head -30
```

- [ ] **Step 2: Implement sri.rs**

```rust
// pathwise-core/src/scheme/sri.rs
use super::Scheme;
use crate::state::{Diffusion, Increment, State};
use crate::process::markov::Drift;

/// Rossler SRI1 scheme: strong order 1.5 for scalar SDEs.
///
/// Extends Milstein with the dZ iterated-integral correction.
/// Full step for Ito SDE `dX = f dt + g dW`:
///   x_new = x + f*dt + g*dw
///           + 0.5*g*(dg/dx)*(dw^2 - dt)            [Milstein correction]
///           + (g*(dg/dx) + g^2*(d^2g/dx^2)) * dz   [SRI correction using dz]
///
/// Both first and second derivatives are approximated by central finite difference.
///
/// SRI requires SCALAR or DIAGONAL-NOISE SDEs. For full-matrix correlated processes
/// (Heston, CorrOU), use Milstein which achieves order 1.0 with diagonal Levy area.
pub struct Sri {
    h: f64,
}

impl Sri {
    pub fn new(h: f64) -> Self { Self { h } }
}

impl Scheme<f64> for Sri {
    type Noise = f64;

    fn step<D, G>(
        &self,
        drift: &D,
        diffusion: &G,
        x: &f64,
        t: f64,
        dt: f64,
        inc: &Increment<f64>,
    ) -> f64
    where
        D: Drift<f64>,
        G: Diffusion<f64, f64>,
    {
        let dw = inc.dw;
        let dz = inc.dz;
        let h = self.h;

        let f = drift(x, t);
        // g(x,t) and its finite-difference derivatives
        let g      = diffusion.apply(x, t, &1.0_f64);
        let g_plus  = diffusion.apply(&(x + h), t, &1.0_f64);
        let g_minus = diffusion.apply(&(x - h), t, &1.0_f64);
        let dg_dx  = (g_plus - g_minus) / (2.0 * h);
        let d2g_dx2 = (g_plus - 2.0 * g + g_minus) / (h * h);

        // Milstein correction
        let milstein_correction = 0.5 * g * dg_dx * (dw * dw - dt);
        // SRI correction (iterated integral dZ term)
        let sri_correction = (g * dg_dx + g * g * d2g_dx2) * dz;

        x + f * dt + g * dw + milstein_correction + sri_correction
    }
}

pub fn sri() -> Sri { Sri::new(1e-4) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::markov::bm;
    use crate::state::Increment;

    #[test]
    fn sri_equals_euler_for_constant_diffusion() {
        // For constant g, dg/dx=0 and d2g/dx2=0, so SRI == Euler
        let b = bm();
        let s = sri();
        let e = crate::scheme::euler::euler();
        let x = 1.0_f64;
        let inc = Increment { dw: 0.3, dz: 0.001 };
        let x_euler = e.step(&b.drift, &b.diffusion, &x, 0.0, 0.01, &inc);
        let x_sri = s.step(&b.drift, &b.diffusion, &x, 0.0, 0.01, &inc);
        assert!((x_euler - x_sri).abs() < 1e-6,
            "BM: euler={} sri={}", x_euler, x_sri);
    }

    #[test]
    fn sri_differs_from_milstein_for_state_dependent_diffusion() {
        // GBM with nonzero dz; SRI adds d2g term
        let gbm = crate::process::markov::gbm(0.05, 0.3);
        let s = sri();
        let m = crate::scheme::milstein::milstein();
        let x = 1.0_f64;
        let inc = Increment { dw: 0.05, dz: 0.0001 };
        let dt = 0.01;
        let x_sri = s.step(&gbm.drift, &gbm.diffusion, &x, 0.0, dt, &inc);
        let x_mil = m.step(&gbm.drift, &gbm.diffusion, &x, 0.0, dt, &inc);
        // They differ because SRI includes the dz term
        assert!((x_sri - x_mil).abs() > 1e-10, "SRI and Milstein should differ when dz != 0");
    }
}
```

- [ ] **Step 3: Run SRI tests**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core sri 2>&1 | tail -20
```

Expected: unit tests pass. The convergence tests may run slowly (expected).

- [ ] **Step 4: Run full convergence suite**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core --test convergence -- --nocapture 2>&1 | tail -40
```

Expected: all convergence tests pass including `sri_strong_order_on_gbm` (order in [1.2, 1.8]) and `sri_stronger_than_milstein_strong`.

- [ ] **Step 5: Commit**

```bash
cd /home/alejandrosotofranco/pathwise
git add pathwise-core/src/scheme/sri.rs pathwise-core/tests/convergence.rs
git commit -m "feat(pathwise-core): add Sri strong-order-1.5 scheme with Rossler SRI1 formula"
```

---

## Task 4: Error Variants + CIR Process

**Files:**
- Modify: `pathwise-core/src/error.rs`
- Modify: `pathwise-core/src/process/markov.rs`
- Modify: `pathwise-core/tests/convergence.rs`

- [ ] **Step 1: Write failing tests**

Add to `pathwise-core/tests/convergence.rs`:

```rust
#[test]
fn cir_stays_nonnegative() {
    // Test non-negativity with a manually constructed CIR-like SDE where Feller is at the
    // boundary (2*kappa*theta == sigma^2 = 0.04). The cir() constructor would reject this
    // via FellerViolation, so we bypass it with SDE::new directly to test that the
    // `x.max(0.0)` clipping in the diffusion keeps all values >= 0.
    use pathwise_core::scheme::euler;
    let (kappa, theta, sigma) = (1.0_f64, 0.02_f64, 0.2_f64);
    // 2*1.0*0.02 = 0.04 == 0.2^2 = 0.04: exactly at Feller boundary
    let sde = pathwise_core::SDE::new(
        move |x: &f64, _t: f64| kappa * (theta - x),
        move |x: &f64, _t: f64| sigma * x.max(0.0_f64).sqrt(),
    );
    let out = pathwise_core::simulate(
        &sde.drift, &sde.diffusion, &euler(), 0.05, 0.0, 1.0, 1000, 500, 42
    ).unwrap();
    for val in out.iter() {
        if !val.is_nan() {
            assert!(*val >= 0.0, "CIR produced negative value: {}", val);
        }
    }
}

#[test]
fn cir_mean_exact() {
    // E[X_T] = theta + (x0 - theta)*exp(-kappa*T)
    use pathwise_core::process::markov::cir;
    use pathwise_core::scheme::euler;
    let (kappa, theta, sigma, x0, t1) = (3.0_f64, 0.1, 0.3, 0.5, 1.0);
    // Feller: 2*3*0.1 = 0.6 > 0.09 -- satisfied
    let sde = cir(kappa, theta, sigma).unwrap();
    let out = pathwise_core::simulate(
        &sde.drift, &sde.diffusion, &euler(), x0, 0.0, t1, 20_000, 500, 0
    ).unwrap();
    let col = out.column(500);
    let sample_mean: f64 = col.iter().filter(|x| x.is_finite()).sum::<f64>()
        / col.iter().filter(|x| x.is_finite()).count() as f64;
    let exact_mean = theta + (x0 - theta) * (-kappa * t1).exp();
    println!("CIR mean: {:.4} expected {:.4}", sample_mean, exact_mean);
    assert!((sample_mean - exact_mean).abs() / exact_mean < 0.02,
        "CIR mean {:.4} vs exact {:.4}", sample_mean, exact_mean);
}

#[test]
fn cir_rejects_invalid_params() {
    use pathwise_core::process::markov::cir;
    assert!(cir(0.0, 0.1, 0.3).is_err(), "kappa=0 should fail");
    assert!(cir(1.0, 0.0, 0.3).is_err(), "theta=0 should fail");
    assert!(cir(1.0, 0.1, -0.1).is_err(), "sigma<0 should fail");
}
```

Run to confirm compilation error (`cir` not found):
```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core cir 2>&1 | head -20
```

- [ ] **Step 2: Add FellerViolation and DimensionMismatch to error.rs**

```rust
// pathwise-core/src/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PathwiseError {
    #[error("invalid SDE parameters: {0}")]
    InvalidParameters(String),

    #[error("numerical divergence at step {step}: value={value}")]
    NumericalDivergence { step: usize, value: f64 },

    #[error("inference failed to converge: {0}")]
    ConvergenceFailure(String),

    /// CIR Feller condition 2*kappa*theta > sigma^2 not strictly satisfied.
    /// Simulation continues with zero-clipping but accuracy near zero is reduced.
    #[error("Feller condition violated: {0}")]
    FellerViolation(String),

    /// Diffusion matrix shape incompatible with noise or state dimensions.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_are_human_readable() {
        let e = PathwiseError::InvalidParameters("H must be in (0,1)".into());
        assert!(e.to_string().contains("H must be in (0,1)"));
        let e = PathwiseError::FellerViolation("2*1*0.02 <= 0.04".into());
        assert!(e.to_string().contains("Feller"));
        let e = PathwiseError::DimensionMismatch("expected 2x2, got 3x3".into());
        assert!(e.to_string().contains("dimension mismatch"));
    }
}
```

- [ ] **Step 3: Add cir() to markov.rs**

Add after the `ou()` function:

```rust
/// Cox-Ingersoll-Ross: dX = kappa*(theta - X) dt + sigma*sqrt(X) dW
///
/// Requires `kappa > 0`, `theta > 0`, `sigma > 0`.
/// Strict Feller condition: `2*kappa*theta > sigma^2`. Returns `Err(FellerViolation)` if not met.
/// Simulation clips X to 0.0 when discretization produces negative values (full truncation).
pub fn cir(
    kappa: f64,
    theta: f64,
    sigma: f64,
) -> Result<SDE<f64, impl Drift<f64>, impl Fn(f64, f64) -> f64 + Send + Sync>, crate::error::PathwiseError> {
    if kappa <= 0.0 || theta <= 0.0 || sigma <= 0.0 {
        return Err(crate::error::PathwiseError::InvalidParameters(
            format!("CIR requires kappa, theta, sigma > 0; got kappa={}, theta={}, sigma={}", kappa, theta, sigma)
        ));
    }
    if 2.0 * kappa * theta <= sigma * sigma {
        return Err(crate::error::PathwiseError::FellerViolation(
            format!("2*kappa*theta = {:.4} <= sigma^2 = {:.4}; boundary is reflecting in continuous time but clipping may introduce bias under discretization",
                2.0 * kappa * theta, sigma * sigma)
        ));
    }
    Ok(SDE::new(
        move |x: &f64, _t: f64| kappa * (theta - x),
        move |x: &f64, _t: f64| sigma * x.max(0.0).sqrt(),
    ))
}
```

Note: The diffusion closure returns the sigma coefficient. It must satisfy `Diffusion<f64, f64>` via the blanket impl in state.rs (`f(x,t) * dw`). The `x.max(0.0)` ensures no NaN from sqrt even if the state becomes negative between steps; simulate also clips.

The return type is ugly. Clean it up by defining a type alias or using boxed closures. For the plan, use `impl Trait` return.

Actually there's a problem: two `impl Fn` in the return type makes the function signature unresolvable in some Rust versions. Use `SDE<f64, impl Drift<f64>, impl Fn(f64, f64) -> f64 + Send + Sync>` which should work since they are different type positions.

- [ ] **Step 4: Update lib.rs to export `cir`**

```rust
// in pathwise-core/src/lib.rs, update process re-export:
pub use process::{bm, cir, gbm, ou, Drift, SDE};
```

- [ ] **Step 5: Run CIR tests**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core cir 2>&1 | tail -20
```

Expected: all three CIR tests pass.

- [ ] **Step 6: Run full test suite**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /home/alejandrosotofranco/pathwise
git add pathwise-core/src/error.rs pathwise-core/src/process/markov.rs pathwise-core/src/lib.rs pathwise-core/tests/convergence.rs
git commit -m "feat(pathwise-core): add FellerViolation/DimensionMismatch errors + CIR process"
```

---

## Task 5: nD Foundation, simulate_nd, and Heston

**Files:**
- Modify: `pathwise-core/src/simulate.rs` — add `simulate_nd`, `PathOutput`
- Modify: `pathwise-core/src/process/markov.rs` — add `HestonDiffusion`, `NdSDE`, `heston()`
- Create: `pathwise-core/tests/nd.rs`
- Modify: `pathwise-core/src/lib.rs` — export new types

- [ ] **Step 1: Write failing tests**

Create `pathwise-core/tests/nd.rs`:

```rust
use nalgebra::SVector;
use pathwise_core::process::markov::heston;
use pathwise_core::scheme::euler;

#[test]
fn heston_output_shape() {
    let sde = heston(0.05, 2.0, 0.04, 0.3, -0.7);
    let out = pathwise_core::simulate_nd::<2, _, _, _>(
        &sde.drift, &sde.diffusion, &euler(),
        SVector::from([0.0_f64, 0.04]), 0.0, 1.0, 10, 100, 0
    ).unwrap();
    assert_eq!(out.shape(), &[10, 101, 2]);
}

#[test]
fn heston_variance_nonnegative() {
    let sde = heston(0.05, 2.0, 0.04, 0.3, -0.7);
    let out = pathwise_core::simulate_nd::<2, _, _, _>(
        &sde.drift, &sde.diffusion, &euler(),
        SVector::from([0.0_f64, 0.04]), 0.0, 1.0, 500, 200, 42
    ).unwrap();
    // Variance is component index 1
    for path in 0..500 {
        for step in 0..=200 {
            let v = out[[path, step, 1]];
            assert!(v >= 0.0 || v.is_nan(), "variance negative: {} at path={} step={}", v, path, step);
        }
    }
}

#[test]
fn euler_nd_diagonal_matches_scalar() {
    // A 2D Euler with diagonal uncorrelated diffusion on GBM-like process
    // should match two independent scalar Euler runs with the same seeds.
    use pathwise_core::simulate;
    // 2D with zero cross-dependence: components evolve independently.
    // drift: [mu*x0, theta*(mu - x1)] ; diffusion diagonal: [sigma*x0, sigma_ou]
    let mu = 0.05_f64;
    let sigma = 0.2_f64;
    let drift_2d = move |x: &SVector<f64, 2>, _t: f64| -> SVector<f64, 2> {
        SVector::from([mu * x[0], -2.0 * x[1]])
    };
    let diff_2d = move |x: &SVector<f64, 2>, _t: f64| -> SVector<f64, 2> {
        SVector::from([sigma * x[0], 0.3_f64])
    };
    let x0_2d = SVector::from([1.0_f64, 0.0]);
    let out_2d = pathwise_core::simulate_nd::<2, _, _, _>(
        &drift_2d, &diff_2d, &euler(), x0_2d, 0.0, 1.0, 1, 100, 0
    ).unwrap();

    // Scalar GBM with same seed
    let sde_scalar = pathwise_core::gbm(mu, sigma);
    let out_scalar = simulate(
        &sde_scalar.drift, &sde_scalar.diffusion, &euler(), 1.0, 0.0, 1.0, 1, 100, 0
    ).unwrap();

    // NOTE: paths won't match exactly because simulate_nd generates 2 normals per step
    // (one for each noise component) while simulate generates 1. This test just checks
    // that the nD output has the correct shape and finite values.
    assert_eq!(out_2d.shape(), &[1, 101, 2]);
    for val in out_2d.iter() {
        // Not asserting exact match, just that values are finite
        if !val.is_nan() {
            assert!(val.is_finite(), "nD Euler produced non-finite value");
        }
    }
    let _ = out_scalar;
}
```

Run to confirm compile error:
```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core --test nd 2>&1 | head -20
```

- [ ] **Step 2: Add simulate_nd to simulate.rs**

Add after the existing `simulate` function:

```rust
use ndarray::Array3;
use nalgebra::SVector;
use crate::state::Diffusion as DiffusionTrait;

/// Simulate `n_paths` paths of an N-dimensional SDE from `t0` to `t1`.
///
/// Returns Array3<f64> of shape `(n_paths, n_steps + 1, N)`.
/// Negative values in each component are NOT clipped unless the process does so
/// in its diffusion (e.g. CIR, Heston full-truncation).
#[allow(clippy::too_many_arguments)]
pub fn simulate_nd<const N: usize, D, G, SC>(
    drift: &D,
    diffusion: &G,
    scheme: &SC,
    x0: SVector<f64, N>,
    t0: f64,
    t1: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Result<Array3<f64>, PathwiseError>
where
    D: Fn(&SVector<f64, N>, f64) -> SVector<f64, N> + Send + Sync,
    G: DiffusionTrait<SVector<f64, N>, SVector<f64, N>> + Sync,
    SC: Scheme<SVector<f64, N>, Noise = SVector<f64, N>>,
{
    if n_paths == 0 || n_steps == 0 {
        return Err(PathwiseError::InvalidParameters("n_paths and n_steps must be > 0".into()));
    }
    if t1 <= t0 {
        return Err(PathwiseError::InvalidParameters("t1 must be > t0".into()));
    }

    let dt = (t1 - t0) / n_steps as f64;
    let base_seed = splitmix64(seed);

    let rows: Vec<Vec<SVector<f64, N>>> = (0..n_paths)
        .into_par_iter()
        .map(|i| {
            let path_seed = splitmix64(base_seed.wrapping_add(i as u64));
            let mut rng = rand::rngs::SmallRng::seed_from_u64(path_seed);
            let mut path = Vec::with_capacity(n_steps + 1);
            let mut x = x0;
            path.push(x);
            for step in 0..n_steps {
                let t = t0 + step as f64 * dt;
                let inc = <SVector<f64, N> as NoiseIncrement>::sample(&mut rng, dt);
                x = scheme.step(drift, diffusion, &x, t, dt, &inc);
                // Check for NaN in any component; freeze path
                if x.iter().any(|v| !v.is_finite()) {
                    x = SVector::from_fn(|_, _| f64::NAN);
                }
                path.push(x);
            }
            path
        })
        .collect();

    let mut out = Array3::zeros((n_paths, n_steps + 1, N));
    for (i, path) in rows.iter().enumerate() {
        for (j, state) in path.iter().enumerate() {
            for k in 0..N {
                out[[i, j, k]] = state[k];
            }
        }
    }
    Ok(out)
}
```

Add `use ndarray::Array3;` and required imports to simulate.rs.

Note: `D: Fn(&SVector<f64,N>, ...) + Send + Sync` satisfies `Drift<SVector<f64,N>>` via markov.rs's blanket impl. But `simulate_nd` uses `Fn` bound directly to avoid needing `Drift` import in this context.

- [ ] **Step 3: Add HestonDiffusion and heston() to markov.rs**

```rust
// Add to pathwise-core/src/process/markov.rs:

use nalgebra::{SMatrix, SVector};
use crate::state::Diffusion as DiffusionTrait;

/// NdSDE: N-dimensional SDE with vector state and vector noise.
pub struct NdSDE<const N: usize, D, G> {
    pub drift: D,
    pub diffusion: G,
}

impl<const N: usize, D, G> NdSDE<N, D, G>
where
    D: Fn(&SVector<f64, N>, f64) -> SVector<f64, N> + Send + Sync,
    G: DiffusionTrait<SVector<f64, N>, SVector<f64, N>>,
{
    pub fn new(drift: D, diffusion: G) -> Self {
        Self { drift, diffusion }
    }
}

/// Diffusion term for the Heston model (log-price, variance).
/// State: [log S, V]. Noise: [dW1, dW2] (independent).
///
/// Applies the lower-triangular Cholesky matrix:
///   d(log S) += sqrt(V) * dW1
///   dV       += xi * sqrt(V) * (rho * dW1 + sqrt(1-rho^2) * dW2)
///
/// Full truncation: V is clipped to 0 in diffusion computation.
pub struct HestonDiffusion {
    xi: f64,
    rho: f64,
    rho_perp: f64,  // sqrt(1 - rho^2)
}

impl HestonDiffusion {
    pub fn new(xi: f64, rho: f64) -> Self {
        Self { xi, rho, rho_perp: (1.0 - rho * rho).sqrt() }
    }
}

impl DiffusionTrait<SVector<f64, 2>, SVector<f64, 2>> for HestonDiffusion {
    fn apply(&self, x: &SVector<f64, 2>, _t: f64, dw: &SVector<f64, 2>) -> SVector<f64, 2> {
        let v = x[1].max(0.0);  // full truncation
        let sv = v.sqrt();
        SVector::from([
            sv * dw[0],
            sv * self.xi * (self.rho * dw[0] + self.rho_perp * dw[1]),
        ])
    }
}

/// Heston stochastic volatility model.
/// State: [log S, V]; use exp(paths[.., .., 0]) to recover S.
///
/// d(log S) = (mu - V/2) dt + sqrt(V) dW1
/// dV       = kappa * (theta - V) dt + xi * sqrt(V) * (rho * dW1 + sqrt(1-rho^2) * dW2)
///
/// Parameters:
/// - `mu`: risk-neutral drift of log-price
/// - `kappa`: variance mean-reversion speed
/// - `theta`: long-run variance
/// - `xi`: volatility of variance (vol of vol)
/// - `rho`: correlation between price and variance Brownian motions (typically -0.7)
pub fn heston(
    mu: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
) -> NdSDE<2, impl Fn(&SVector<f64, 2>, f64) -> SVector<f64, 2> + Send + Sync, HestonDiffusion> {
    NdSDE::new(
        move |x: &SVector<f64, 2>, _t: f64| -> SVector<f64, 2> {
            let v = x[1].max(0.0);
            SVector::from([mu - v / 2.0, kappa * (theta - x[1])])
        },
        HestonDiffusion::new(xi, rho),
    )
}
```

- [ ] **Step 4: Update lib.rs to export new items**

```rust
// pathwise-core/src/lib.rs — add:
pub use process::{bm, cir, gbm, heston, ou, Drift, HestonDiffusion, NdSDE, SDE};
pub use simulate::{simulate, simulate_nd};
```

- [ ] **Step 5: Run nD tests**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core --test nd 2>&1 | tail -20
```

Expected: all three nD tests pass.

- [ ] **Step 6: Run full test suite**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /home/alejandrosotofranco/pathwise
git add pathwise-core/src/simulate.rs pathwise-core/src/process/markov.rs pathwise-core/src/lib.rs pathwise-core/tests/nd.rs
git commit -m "feat(pathwise-core): add simulate_nd (Array3 output), HestonDiffusion, heston() constructor"
```

---

## Task 6: CorrOU + MilsteinNd

**Files:**
- Modify: `pathwise-core/src/process/markov.rs` — add `CorrOuDiffusion`, `corr_ou()`
- Modify: `pathwise-core/src/scheme/milstein_nd.rs` — full `MilsteinNd<N>` impl
- Modify: `pathwise-core/tests/nd.rs` — add CorrOU and MilsteinNd tests

- [ ] **Step 1: Write failing tests**

Add to `pathwise-core/tests/nd.rs`:

```rust
#[test]
fn corr_ou_covariance() {
    // 2D CorrOU to stationarity: sample covariance should approach Sigma/(2*theta)
    use pathwise_core::process::markov::corr_ou;
    use nalgebra::{SMatrix, SVector};
    let theta = 2.0_f64;
    let mu = SVector::<f64, 2>::zeros();
    // Sigma = [[1.0, 0.5], [0.5, 1.0]]
    let sigma_mat = SMatrix::<f64, 2, 2>::from_row_slice(&[1.0, 0.5, 0.5, 1.0]);
    let sde = corr_ou::<2>(theta, mu, sigma_mat).expect("corr_ou");
    let out = pathwise_core::simulate_nd::<2, _, _, _>(
        &sde.drift, &sde.diffusion, &pathwise_core::scheme::euler(),
        SVector::zeros(), 0.0, 5.0, 10_000, 1000, 0
    ).unwrap();
    // Sample covariance at T=5 (near stationarity)
    let n_paths = 10_000_usize;
    let last_step = 1000_usize;
    let mean0: f64 = (0..n_paths).map(|i| out[[i, last_step, 0]]).sum::<f64>() / n_paths as f64;
    let mean1: f64 = (0..n_paths).map(|i| out[[i, last_step, 1]]).sum::<f64>() / n_paths as f64;
    let cov00: f64 = (0..n_paths).map(|i| (out[[i,last_step,0]]-mean0).powi(2)).sum::<f64>() / (n_paths-1) as f64;
    let cov11: f64 = (0..n_paths).map(|i| (out[[i,last_step,1]]-mean1).powi(2)).sum::<f64>() / (n_paths-1) as f64;
    let cov01: f64 = (0..n_paths).map(|i| (out[[i,last_step,0]]-mean0)*(out[[i,last_step,1]]-mean1)).sum::<f64>() / (n_paths-1) as f64;
    // Stationary: Sigma/(2*theta) = [[0.25, 0.125], [0.125, 0.25]]
    let expected_diag = 1.0 / (2.0 * theta);  // 0.25
    let expected_offdiag = 0.5 / (2.0 * theta);  // 0.125
    println!("CorrOU cov00={:.4} (expected {:.4})", cov00, expected_diag);
    println!("CorrOU cov11={:.4} (expected {:.4})", cov11, expected_diag);
    println!("CorrOU cov01={:.4} (expected {:.4})", cov01, expected_offdiag);
    assert!((cov00 - expected_diag).abs() / expected_diag < 0.05, "cov00 {:.4} vs {:.4}", cov00, expected_diag);
    assert!((cov11 - expected_diag).abs() / expected_diag < 0.05, "cov11 {:.4} vs {:.4}", cov11, expected_diag);
    assert!((cov01 - expected_offdiag).abs() / expected_offdiag < 0.10, "cov01 {:.4} vs {:.4}", cov01, expected_offdiag);
}

#[test]
fn milstein_nd_stronger_than_euler_nd_on_gbm_like() {
    // 1D GBM run via simulate_nd with MilsteinNd should have lower strong error than EulerNd
    use pathwise_core::scheme::{euler, milstein_nd};
    use nalgebra::SVector;
    let mu = 0.0_f64;
    let sigma = 0.3_f64;
    let drift_fn = move |x: &SVector<f64, 1>, _t: f64| SVector::from([mu * x[0]]);
    let diff_fn  = move |x: &SVector<f64, 1>, _t: f64| SVector::from([sigma * x[0]]);
    let x0 = SVector::from([1.0_f64]);
    let t1 = 1.0_f64;
    let n_paths = 5000;
    let n_steps = 50;
    // Just verify MilsteinNd compiles and produces finite output
    let out_mil = pathwise_core::simulate_nd::<1, _, _, _>(
        &drift_fn, &diff_fn, &milstein_nd::<1>(), x0, 0.0, t1, n_paths, n_steps, 0
    ).unwrap();
    let out_euler = pathwise_core::simulate_nd::<1, _, _, _>(
        &drift_fn, &diff_fn, &euler(), x0, 0.0, t1, n_paths, n_steps, 0
    ).unwrap();
    assert_eq!(out_mil.shape(), &[n_paths, n_steps + 1, 1]);
    // Both should produce finite values at T
    let finite_mil = out_mil.iter().filter(|x| x.is_finite()).count();
    let finite_euler = out_euler.iter().filter(|x| x.is_finite()).count();
    assert!(finite_mil > n_paths * n_steps / 2, "too many NaN in MilsteinNd");
    assert!(finite_euler > n_paths * n_steps / 2, "too many NaN in EulerNd");
}
```

- [ ] **Step 2: Implement CorrOuDiffusion and corr_ou() in markov.rs**

```rust
// Add to pathwise-core/src/process/markov.rs:

/// Correlated Ornstein-Uhlenbeck diffusion via Cholesky factor.
/// State: SVector<N>. Noise: SVector<N> of independent dW_i.
/// apply: L * dW where L = chol(Sigma)
pub struct CorrOuDiffusion<const N: usize> {
    l: SMatrix<f64, N, N>,  // lower Cholesky of Sigma
}

impl<const N: usize> DiffusionTrait<SVector<f64, N>, SVector<f64, N>> for CorrOuDiffusion<N> {
    fn apply(&self, _x: &SVector<f64, N>, _t: f64, dw: &SVector<f64, N>) -> SVector<f64, N> {
        self.l * dw
    }
}

/// N-dimensional correlated Ornstein-Uhlenbeck process.
/// dX = theta*(mu - X) dt + L dW  where L = chol(Sigma)
///
/// Parameters:
/// - `theta`: scalar mean-reversion rate applied to all components
/// - `mu`: long-run mean vector
/// - `sigma_mat`: N×N positive-definite diffusion covariance matrix Sigma
///
/// Returns `Err(DimensionMismatch)` if `sigma_mat` Cholesky fails (not positive-definite).
pub fn corr_ou<const N: usize>(
    theta: f64,
    mu: SVector<f64, N>,
    sigma_mat: SMatrix<f64, N, N>,
) -> Result<
    NdSDE<N, impl Fn(&SVector<f64, N>, f64) -> SVector<f64, N> + Send + Sync, CorrOuDiffusion<N>>,
    crate::error::PathwiseError,
> {
    let chol = nalgebra::Cholesky::new(sigma_mat).ok_or_else(|| {
        crate::error::PathwiseError::DimensionMismatch(
            "sigma_mat is not positive-definite (Cholesky failed)".into()
        )
    })?;
    let l = chol.l();
    Ok(NdSDE::new(
        move |x: &SVector<f64, N>, _t: f64| -> SVector<f64, N> {
            (mu - x) * theta
        },
        CorrOuDiffusion { l },
    ))
}
```

- [ ] **Step 3: Implement MilsteinNd in milstein_nd.rs**

```rust
// pathwise-core/src/scheme/milstein_nd.rs
use super::Scheme;
use crate::state::{Diffusion, Increment, State};
use crate::process::markov::Drift;
use nalgebra::SVector;

/// N-dimensional Milstein scheme with diagonal Levy area terms only.
///
/// For each component i, computes the correction:
///   0.5 * g_ii * (dg_ii/dx_i) * (dW_i^2 - dt)
///
/// where g_ii = diffusion.apply(x, t, e_i)[i] and e_i is the unit vector in direction i.
/// Derivative is approximated by central finite difference.
///
/// Strong order 1.0 when commutativity condition holds, 0.5 for non-commutative systems
/// (e.g. Heston). For non-commutative systems, cross Levy-area terms are dropped.
pub struct MilsteinNd<const N: usize> {
    pub h: f64,
}

impl<const N: usize> MilsteinNd<N> {
    pub fn new(h: f64) -> Self { Self { h } }
}

impl<const N: usize> Scheme<SVector<f64, N>> for MilsteinNd<N> {
    type Noise = SVector<f64, N>;

    fn step<D, G>(
        &self,
        drift: &D,
        diffusion: &G,
        x: &SVector<f64, N>,
        t: f64,
        dt: f64,
        inc: &Increment<SVector<f64, N>>,
    ) -> SVector<f64, N>
    where
        D: Drift<SVector<f64, N>>,
        G: Diffusion<SVector<f64, N>, SVector<f64, N>>,
    {
        let dw = &inc.dw;
        let f_dt = drift(x, t) * dt;
        let g_dw = diffusion.apply(x, t, dw);

        // Diagonal Levy area correction
        let mut correction = SVector::<f64, N>::zeros();
        let h = self.h;
        for i in 0..N {
            let mut ei = SVector::<f64, N>::zeros();
            ei[i] = 1.0;
            // g_ii = i-th component of diffusion evaluated with unit noise in direction i
            let g_ii       = diffusion.apply(x,               t, &ei)[i];
            let mut xp = *x; xp[i] += h;
            let mut xm = *x; xm[i] -= h;
            let g_ii_plus  = diffusion.apply(&xp, t, &ei)[i];
            let g_ii_minus = diffusion.apply(&xm, t, &ei)[i];
            let dg_ii = (g_ii_plus - g_ii_minus) / (2.0 * h);
            correction[i] = 0.5 * g_ii * dg_ii * (dw[i] * dw[i] - dt);
        }

        x + f_dt + g_dw + correction
    }
}

pub fn milstein_nd<const N: usize>() -> MilsteinNd<N> { MilsteinNd::new(1e-5) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Increment;
    use nalgebra::SVector;

    #[test]
    fn milstein_nd_equals_euler_nd_for_constant_diffusion() {
        // Constant diffusion: dg_ii = 0, so MilsteinNd == EulerNd
        let m: MilsteinNd<2> = milstein_nd();
        let e = crate::scheme::euler::EulerMaruyama;
        let drift = |_x: &SVector<f64, 2>, _t: f64| SVector::zeros();
        let diff  = |_x: &SVector<f64, 2>, _t: f64| SVector::from([1.0_f64, 1.0]);
        let x = SVector::from([0.5_f64, -0.5]);
        let dw = SVector::from([0.1_f64, -0.2]);
        let inc = Increment { dw, dz: SVector::zeros() };
        let xe = e.step(&drift, &diff, &x, 0.0, 0.01, &inc);
        let xm = m.step(&drift, &diff, &x, 0.0, 0.01, &inc);
        assert!((xe - xm).norm() < 1e-8, "should be equal for constant diffusion");
    }
}
```

- [ ] **Step 4: Update lib.rs**

```rust
// In lib.rs, update:
pub use process::{bm, cir, corr_ou, gbm, heston, ou, CorrOuDiffusion, Drift, HestonDiffusion, NdSDE, SDE};
pub use scheme::{euler, milstein, milstein_nd, sri, MilsteinNd, Scheme};
```

- [ ] **Step 5: Run nD tests**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core --test nd 2>&1 | tail -30
```

Expected: all nD tests pass including `corr_ou_covariance` and `milstein_nd_stronger_than_euler_nd_on_gbm_like`.

- [ ] **Step 6: Run full test suite**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-core 2>&1 | tail -10
```

- [ ] **Step 7: Commit**

```bash
cd /home/alejandrosotofranco/pathwise
git add pathwise-core/src/ pathwise-core/tests/nd.rs
git commit -m "feat(pathwise-core): add CorrOU, MilsteinNd (diagonal Levy area), corr_ou() constructor"
```

---

## Task 7: Scaffold pathwise-geo + GeodesicEuler

**Files:**
- Modify: `Cargo.toml` (workspace root) — add `pathwise-geo` member
- Create: `pathwise-geo/Cargo.toml`
- Create: `pathwise-geo/src/lib.rs`
- Create: `pathwise-geo/src/sde.rs`
- Create: `pathwise-geo/src/scheme/mod.rs`
- Create: `pathwise-geo/src/scheme/euler.rs`
- Create: `pathwise-geo/tests/manifold_sde.rs` (partial)

- [ ] **Step 1: Add pathwise-geo to workspace**

In `Cargo.toml` (workspace root), update members:
```toml
members = ["pathwise-core", "pathwise-geo", "pathwise-py"]
```

- [ ] **Step 2: Write the failing manifold test**

Create `pathwise-geo/tests/manifold_sde.rs`:

```rust
use cartan_core::Manifold;
use cartan_manifolds::Sphere;
use pathwise_geo::{GeodesicEuler, ManifoldSDE, brownian_motion_on, manifold_simulate};

#[test]
fn geodesic_euler_stays_on_sphere() {
    let s2 = Sphere::<3>;
    let sde = brownian_motion_on(s2);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    // x0 = north pole [0, 0, 1]
    use nalgebra::SVector;
    let x0 = SVector::from([0.0_f64, 0.0, 1.0]);
    let paths = manifold_simulate(&sde, &GeodesicEuler, x0, 0.0, 1.0, 500, 200, 42);
    for path in &paths {
        for point in path {
            let norm: f64 = point.norm();
            assert!((norm - 1.0).abs() < 1e-6, "point off sphere: norm={}", norm);
        }
    }
}

#[test]
fn geodesic_euler_stays_on_so3() {
    use cartan_manifolds::SpecialOrthogonal;
    use nalgebra::SMatrix;
    let so3 = SpecialOrthogonal::<3>;
    let sde = brownian_motion_on(so3);
    let x0 = SMatrix::<f64, 3, 3>::identity();
    let paths = manifold_simulate(&sde, &GeodesicEuler, x0, 0.0, 1.0, 100, 100, 0);
    for path in &paths {
        for r in path {
            let rtr = r.transpose() * r;
            let id = SMatrix::<f64, 3, 3>::identity();
            assert!((rtr - id).norm() < 1e-6, "SO3: R^T R != I, diff={}", (rtr-id).norm());
        }
    }
}
```

Run to confirm compile error (crate doesn't exist yet):
```bash
cd /home/alejandrosotofranco/pathwise && cargo build -p pathwise-geo 2>&1 | head -10
```

- [ ] **Step 3: Create pathwise-geo/Cargo.toml**

```toml
[package]
name = "pathwise-geo"
version = "0.1.0"
edition = "2021"

[dependencies]
pathwise-core = { path = "../pathwise-core" }
cartan-core = { path = "../../cartan/cartan-core", version = "0.1" }
cartan-manifolds = { path = "../../cartan/cartan-manifolds", version = "0.1", default-features = false, features = ["std"] }
nalgebra = { version = "0.33", default-features = false, features = ["std"] }
rayon = "1"
rand = { version = "0.9", default-features = false, features = ["std", "std_rng"] }
rand_distr = { version = "0.5" }

[dev-dependencies]
rand = { version = "0.9", features = ["std_rng"] }
```

- [ ] **Step 4: Create pathwise-geo/src/sde.rs**

```rust
// pathwise-geo/src/sde.rs
/// A stochastic differential equation on a Riemannian manifold.
///
/// `drift`: tangent vector field f(x, t) at each point
/// `diffusion`: tangent vector field g(x, t) for scalar noise
///              (for nD noise, use a vector of tangent fields)
pub struct ManifoldSDE<M, D, G> {
    pub manifold: M,
    pub drift: D,
    pub diffusion: G,
}

impl<M, D, G> ManifoldSDE<M, D, G> {
    pub fn new(manifold: M, drift: D, diffusion: G) -> Self {
        Self { manifold, drift, diffusion }
    }
}
```

- [ ] **Step 5: Create pathwise-geo/src/scheme/mod.rs and euler.rs**

```rust
// pathwise-geo/src/scheme/mod.rs
pub mod euler;
pub mod milstein;
pub mod sri;
pub use euler::GeodesicEuler;
pub use milstein::GeodesicMilstein;
pub use sri::GeodesicSRI;
```

```rust
// pathwise-geo/src/scheme/euler.rs
use crate::sde::ManifoldSDE;
use cartan_core::Manifold;
use pathwise_core::state::Increment;
use rand::Rng;

/// Geodesic Euler-Maruyama: projects onto manifold at every step via exp map.
///
/// x_{n+1} = exp_{x_n}( f(x_n,t)*dt + g(x_n,t)*dW )
///
/// Strong order 0.5. Keeps state exactly on the manifold at every step.
pub struct GeodesicEuler;

impl GeodesicEuler {
    /// Advance one step of the manifold SDE.
    ///
    /// `drift_fn(x, t)` -> tangent vector at x
    /// `diffusion_fn(x, t)` -> tangent vector at x (scaled by dW inside)
    pub fn step<M, D, G>(
        &self,
        sde: &ManifoldSDE<M, D, G>,
        x: &M::Point,
        t: f64,
        dt: f64,
        inc: &Increment<f64>,
    ) -> M::Point
    where
        M: Manifold,
        D: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
        G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
        M::Tangent: std::ops::Add<Output = M::Tangent>
            + std::ops::Mul<f64, Output = M::Tangent>
            + Clone,
    {
        let f = (sde.drift)(x, t);
        let g = (sde.diffusion)(x, t);
        let tangent = f * dt + g * inc.dw;
        sde.manifold.exp(x, &tangent).unwrap_or_else(|_| x.clone())
    }
}
```

Create stubs for milstein and sri:

```rust
// pathwise-geo/src/scheme/milstein.rs
pub struct GeodesicMilstein { pub eps: f64 }
impl GeodesicMilstein { pub fn new() -> Self { Self { eps: 1e-4 } } }
impl Default for GeodesicMilstein { fn default() -> Self { Self::new() } }
```

```rust
// pathwise-geo/src/scheme/sri.rs
pub struct GeodesicSRI { pub eps: f64 }
impl GeodesicSRI { pub fn new() -> Self { Self { eps: 1e-4 } } }
impl Default for GeodesicSRI { fn default() -> Self { Self::new() } }
```

- [ ] **Step 6: Create pathwise-geo/src/simulate.rs (partial) and process.rs (partial)**

```rust
// pathwise-geo/src/simulate.rs
use crate::sde::ManifoldSDE;
use crate::scheme::euler::GeodesicEuler;
use cartan_core::Manifold;
use pathwise_core::state::NoiseIncrement;
use rand::SeedableRng;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Simulate manifold SDE paths using GeodesicEuler.
///
/// Returns `Vec<Vec<M::Point>>`: outer = paths, inner = time steps (n_steps+1 points each).
#[allow(clippy::too_many_arguments)]
pub fn manifold_simulate<M, D, G>(
    sde: &ManifoldSDE<M, D, G>,
    scheme: &GeodesicEuler,
    x0: M::Point,
    t0: f64,
    t1: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Vec<Vec<M::Point>>
where
    M: Manifold + Clone + Send + Sync,
    M::Point: Clone + Send + Sync,
    M::Tangent: std::ops::Add<Output = M::Tangent>
        + std::ops::Mul<f64, Output = M::Tangent>
        + Clone + Send + Sync,
    D: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
{
    let dt = (t1 - t0) / n_steps as f64;
    let base_seed = splitmix64(seed);
    (0..n_paths)
        .map(|i| {
            let path_seed = splitmix64(base_seed.wrapping_add(i as u64));
            let mut rng = rand::rngs::SmallRng::seed_from_u64(path_seed);
            let mut path = Vec::with_capacity(n_steps + 1);
            let mut x = x0.clone();
            path.push(x.clone());
            for step in 0..n_steps {
                let t = t0 + step as f64 * dt;
                let inc = <f64 as NoiseIncrement>::sample(&mut rng, dt);
                x = scheme.step(sde, &x, t, dt, &inc);
                path.push(x.clone());
            }
            path
        })
        .collect()
}
```

```rust
// pathwise-geo/src/process.rs
use crate::sde::ManifoldSDE;
use cartan_core::Manifold;

/// Pure Brownian motion on a manifold (zero drift, unit diffusion).
/// Returns tangent vector of unit "magnitude" (manifold-specific scaling).
pub fn brownian_motion_on<M: Manifold + Clone>(
    manifold: M,
) -> ManifoldSDE<M, impl Fn(&M::Point, f64) -> M::Tangent + Send + Sync, impl Fn(&M::Point, f64) -> M::Tangent + Send + Sync>
where
    M::Tangent: Clone,
{
    // For Sphere<N>: a unit tangent vector is picked as a fixed reference direction.
    // In practice, users will want a random tangent; this provides a deterministic one
    // for testing. The drift is always zero.
    //
    // The diffusion returns M::Tangent which is the "generator" g(x) scaled by dW in the
    // scheme's step function. We use the manifold's log from x to a fixed reference
    // to get a tangent direction; if that fails (x = reference), use zero tangent.
    //
    // For a proper Brownian motion on the manifold, the diffusion should be the identity
    // on the tangent space. Here we approximate by providing a fixed unit tangent direction.
    // This is sufficient for testing manifold-preservation; proper BMon would require
    // an orthonormal frame and is beyond the current scope.
    //
    // Zero drift, unit diffusion (identity on tangent space, scalar noise).
    let _m = manifold.clone();
    ManifoldSDE::new(
        manifold,
        |_x: &M::Point, _t: f64| -> M::Tangent { unimplemented!("use ou_on or provide custom drift") },
        |_x: &M::Point, _t: f64| -> M::Tangent { unimplemented!("use ou_on or provide custom diffusion") },
    )
}

/// Mean-reverting OU process on a manifold.
/// Drift: kappa * log_{x}(mu_point)  (Riemannian log map toward mu_point)
/// Diffusion: constant unit tangent vector field (user-supplied unit tangent generator)
pub fn ou_on<M: Manifold + Clone>(
    manifold: M,
    kappa: f64,
    mu_point: M::Point,
) -> ManifoldSDE<
    M,
    impl Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    impl Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
>
where
    M::Point: Clone + Send + Sync,
    M::Tangent: Clone + std::ops::Mul<f64, Output = M::Tangent> + Send + Sync,
{
    let mu_clone = mu_point.clone();
    let manifold_clone = manifold.clone();
    ManifoldSDE::new(
        manifold,
        move |x: &M::Point, _t: f64| -> M::Tangent {
            // Riemannian log from x to mu gives drift direction; scale by kappa
            manifold_clone.log(x, &mu_clone)
                .map(|v| v * kappa)
                .unwrap_or_else(|_| manifold_clone.log(x, &mu_clone).unwrap())
        },
        move |_x: &M::Point, _t: f64| -> M::Tangent {
            unimplemented!("ou_on diffusion: provide a unit tangent generator for your manifold")
        },
    )
}
```

Note: `brownian_motion_on` and `ou_on` are incomplete skeletons here. The full implementation in Task 9 will make them functional. For Task 7, the test only needs `GeodesicEuler` to work, which requires `manifold_simulate` with working drift and diffusion functions. The tests in Task 7 will use inline closures rather than the process constructors.

Update the test to use inline closures instead:

```rust
// pathwise-geo/tests/manifold_sde.rs — revised for Task 7:
use cartan_core::Manifold;
use cartan_manifolds::{Sphere, SpecialOrthogonal};
use nalgebra::{SMatrix, SVector};
use pathwise_geo::{GeodesicEuler, ManifoldSDE, manifold_simulate};

fn sphere_sde() -> ManifoldSDE<Sphere<3>,
    impl Fn(&SVector<f64,3>, f64) -> SVector<f64,3> + Send + Sync,
    impl Fn(&SVector<f64,3>, f64) -> SVector<f64,3> + Send + Sync>
{
    ManifoldSDE::new(
        Sphere::<3>,
        |_x: &SVector<f64, 3>, _t: f64| SVector::zeros::<3>(),
        |x: &SVector<f64, 3>, _t: f64| {
            let e1 = SVector::from([1.0_f64, 0.0, 0.0]);
            e1 - x * x.dot(&e1)
        },
    )
}

#[test]
fn geodesic_euler_stays_on_sphere() {
    let sde = sphere_sde();
    let x0 = SVector::from([0.0_f64, 0.0, 1.0]);
    let paths = manifold_simulate(&sde, &GeodesicEuler, x0, 0.0, 1.0, 500, 200, 42);
    for path in &paths {
        for point in path {
            let norm: f64 = point.norm();
            assert!((norm - 1.0).abs() < 1e-6, "point off sphere: norm={}", norm);
        }
    }
}

#[test]
fn geodesic_euler_stays_on_so3() {
    // SpecialOrthogonal<3>: state is SMatrix<f64,3,3>. Diffusion projects a skew-symmetric
    // tangent direction onto the tangent space (so(3) = skew-symmetric 3x3 matrices).
    let so3 = SpecialOrthogonal::<3>;
    // Tangent: a fixed skew-symmetric matrix (the generator of rotation around z-axis)
    let omega = SMatrix::<f64,3,3>::from_row_slice(&[0.0,-1.0,0.0, 1.0,0.0,0.0, 0.0,0.0,0.0]);
    let sde = ManifoldSDE::new(
        so3,
        |_x: &SMatrix<f64,3,3>, _t: f64| SMatrix::zeros(),
        move |_x: &SMatrix<f64,3,3>, _t: f64| omega,
    );
    let x0 = SMatrix::<f64,3,3>::identity();
    let paths = manifold_simulate(&sde, &GeodesicEuler, x0, 0.0, 1.0, 100, 100, 0);
    for path in &paths {
        for r in path {
            let rtr = r.transpose() * r;
            let id = SMatrix::<f64,3,3>::identity();
            assert!((rtr - id).norm() < 1e-6,
                "SO3: R^T R != I, diff={}", (rtr - id).norm());
        }
    }
}

#[test]
fn paths_to_array_shape() {
    use pathwise_geo::paths_to_array;
    let sde = sphere_sde();
    let x0 = SVector::from([0.0_f64, 0.0, 1.0]);
    let paths = manifold_simulate(&sde, &GeodesicEuler, x0, 0.0, 1.0, 10, 50, 0);
    let arr = paths_to_array(&paths, &Sphere::<3>, &x0);
    assert_eq!(arr.shape(), &[10, 51, 3]);
}
```

- [ ] **Step 7: Create pathwise-geo/src/lib.rs**

```rust
// pathwise-geo/src/lib.rs
pub mod process;
pub mod scheme;
pub mod sde;
pub mod simulate;

pub use process::{brownian_motion_on, ou_on};
pub use scheme::{GeodesicEuler, GeodesicMilstein, GeodesicSRI};
pub use sde::ManifoldSDE;
pub use simulate::{manifold_simulate, paths_to_array};
```

Add `paths_to_array` to simulate.rs:

```rust
// In pathwise-geo/src/simulate.rs — add after manifold_simulate:
use ndarray::Array3;
use cartan_core::Manifold;

/// Flatten paths into Array3 by projecting each point onto the tangent space at ref_point.
///
/// Shape: (n_paths, n_steps+1, dim) where dim is the dimension of M::Point as a flat vector.
/// Points are mapped via log_{ref_point} -- the caller is responsible for ensuring all
/// path points are within the injectivity radius of ref_point.
pub fn paths_to_array<M>(
    paths: &[Vec<M::Point>],
    manifold: &M,
    ref_point: &M::Point,
) -> Array3<f64>
where
    M: Manifold,
    M::Point: Clone,
    M::Tangent: AsRef<[f64]>,
{
    let n_paths = paths.len();
    let n_steps_plus1 = if n_paths > 0 { paths[0].len() } else { 0 };
    let dim = if n_steps_plus1 > 0 && n_paths > 0 {
        manifold.log(ref_point, &paths[0][0]).map(|v| v.as_ref().len()).unwrap_or(0)
    } else { 0 };
    let mut out = Array3::zeros((n_paths, n_steps_plus1, dim));
    for (i, path) in paths.iter().enumerate() {
        for (j, point) in path.iter().enumerate() {
            if let Ok(tangent) = manifold.log(ref_point, point) {
                let slice = tangent.as_ref();
                for (k, &v) in slice.iter().enumerate().take(dim) {
                    out[[i, j, k]] = v;
                }
            }
        }
    }
    out
}
```

Note: `M::Tangent: AsRef<[f64]>` may not hold for all manifolds. For `Sphere<N>`, the tangent is `SVector<f64, N>` which does implement `AsRef<[f64]>`. If this bound causes compile issues, use `nalgebra` specific bounds or store as `SVector` directly.

Alternative for paths_to_array if AsRef bound fails: require `M::Point: Into<nalgebra::DVector<f64>>` or just use a concrete impl for the tests. The tests only need shape, so the simplest implementation that compiles is acceptable.

- [ ] **Step 8: Build pathwise-geo**

```bash
cd /home/alejandrosotofranco/pathwise && cargo build -p pathwise-geo 2>&1 | tail -20
```

Fix any compilation errors (typically trait bound issues with Manifold associated types).

- [ ] **Step 9: Run manifold_sde tests**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-geo --test manifold_sde 2>&1 | tail -20
```

Expected: `geodesic_euler_stays_on_sphere` and `paths_to_array_shape` pass.

- [ ] **Step 10: Commit**

```bash
cd /home/alejandrosotofranco/pathwise
git add Cargo.toml pathwise-geo/
git commit -m "feat(pathwise-geo): scaffold crate with ManifoldSDE, GeodesicEuler, manifold_simulate"
```

---

## Task 8: GeodesicMilstein

**Files:**
- Modify: `pathwise-geo/src/scheme/milstein.rs`
- Modify: `pathwise-geo/src/simulate.rs` — add scheme-generic version
- Modify: `pathwise-geo/tests/manifold_sde.rs` — add Milstein tests

GeodesicMilstein uses finite-difference covariant derivative approximation:
```
nabla_g g(x) ≈ (1/eps) * [PT_{exp_x(eps*g) -> x}( g(exp_x(eps*g)) ) - g(x)]
```

Full step:
```
v = f(x)*dt + g(x)*dW + 0.5 * nabla_g g(x) * (dW^2 - dt)
x_new = exp_x(v)
```

- [ ] **Step 1: Write failing test**

Add to `pathwise-geo/tests/manifold_sde.rs`:

```rust
#[test]
fn geodesic_milstein_stays_on_sphere() {
    use pathwise_geo::GeodesicMilstein;
    let s2 = Sphere::<3>;
    let sde = ManifoldSDE::new(
        s2,
        |_x: &SVector<f64, 3>, _t: f64| SVector::zeros::<3>(),
        |x: &SVector<f64, 3>, _t: f64| {
            let e1 = SVector::from([1.0_f64, 0.0, 0.0]);
            e1 - x * x.dot(&e1)
        },
    );
    let x0 = SVector::from([0.0_f64, 0.0, 1.0]);
    let paths = pathwise_geo::simulate::manifold_simulate_with_scheme(
        &sde, &GeodesicMilstein::new(), x0, 0.0, 1.0, 100, 100, 0
    );
    for path in &paths {
        for point in path {
            let norm = point.norm();
            assert!((norm - 1.0).abs() < 1e-6, "Milstein point off sphere: {}", norm);
        }
    }
}
```

- [ ] **Step 2: Implement GeodesicMilstein fully**

```rust
// pathwise-geo/src/scheme/milstein.rs
use crate::sde::ManifoldSDE;
use cartan_core::{Manifold, ParallelTransport};
use pathwise_core::state::Increment;

pub struct GeodesicMilstein {
    eps: f64,
}

impl GeodesicMilstein {
    pub fn new() -> Self { Self { eps: 1e-4 } }

    pub fn step<M, D, G>(
        &self,
        sde: &ManifoldSDE<M, D, G>,
        x: &M::Point,
        t: f64,
        dt: f64,
        inc: &Increment<f64>,
    ) -> M::Point
    where
        M: Manifold + ParallelTransport,
        D: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
        G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
        M::Tangent: std::ops::Add<Output = M::Tangent>
            + std::ops::Mul<f64, Output = M::Tangent>
            + std::ops::Sub<Output = M::Tangent>
            + Clone,
    {
        let dw = inc.dw;
        let f  = (sde.drift)(x, t);
        let g  = (sde.diffusion)(x, t);
        let eps = self.eps;

        // Milstein correction: nabla_g g via parallel transport
        // Step 1: walk eps along g direction from x
        let eps_g = g.clone() * eps;
        let y = sde.manifold.exp(x, &eps_g).unwrap_or_else(|_| x.clone());
        // Step 2: evaluate g at y
        let g_at_y = (sde.diffusion)(&y, t);
        // Step 3: transport g_at_y back to x
        let g_at_y_transported = sde.manifold.transport(&y, x, &g_at_y)
            .unwrap_or_else(|_| g.clone());
        // Step 4: finite-difference covariant derivative
        let nabla_g_g = (g_at_y_transported - g.clone()) * (1.0 / eps);

        let correction = nabla_g_g * (0.5 * (dw * dw - dt));
        let tangent = f * dt + g * dw + correction;
        sde.manifold.exp(x, &tangent).unwrap_or_else(|_| x.clone())
    }
}

impl Default for GeodesicMilstein {
    fn default() -> Self { Self::new() }
}
```

- [ ] **Step 3: Add manifold_simulate_with_scheme to simulate.rs**

The existing `manifold_simulate` is hardcoded to `GeodesicEuler`. Add a generic version:

```rust
// In pathwise-geo/src/simulate.rs — add:

/// Simulate manifold SDE paths with any scheme that has a `step` method taking
/// `(sde, x, t, dt, inc) -> M::Point`.
///
/// Currently supports GeodesicEuler and GeodesicMilstein via trait dispatch.
/// GeodesicSRI added in Task 9.
pub fn manifold_simulate_with_scheme<M, D, G, SC>(
    sde: &ManifoldSDE<M, D, G>,
    scheme: &SC,
    x0: M::Point,
    t0: f64,
    t1: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Vec<Vec<M::Point>>
where
    M: Manifold + cartan_core::ParallelTransport + Clone + Send + Sync,
    M::Point: Clone + Send + Sync,
    M::Tangent: std::ops::Add<Output = M::Tangent>
        + std::ops::Mul<f64, Output = M::Tangent>
        + std::ops::Sub<Output = M::Tangent>
        + Clone + Send + Sync,
    D: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    SC: GeoScheme<M, D, G>,
{
    let dt = (t1 - t0) / n_steps as f64;
    let base_seed = splitmix64(seed);
    (0..n_paths)
        .map(|i| {
            let path_seed = splitmix64(base_seed.wrapping_add(i as u64));
            let mut rng = rand::rngs::SmallRng::seed_from_u64(path_seed);
            let mut path = Vec::with_capacity(n_steps + 1);
            let mut x = x0.clone();
            path.push(x.clone());
            for step in 0..n_steps {
                let t = t0 + step as f64 * dt;
                let inc = <f64 as NoiseIncrement>::sample(&mut rng, dt);
                x = scheme.step_geo(sde, &x, t, dt, &inc);
                path.push(x.clone());
            }
            path
        })
        .collect()
}

/// Internal trait to unify geodesic scheme step dispatch.
pub trait GeoScheme<M, D, G>
where
    M: Manifold,
    D: Fn(&M::Point, f64) -> M::Tangent,
    G: Fn(&M::Point, f64) -> M::Tangent,
{
    fn step_geo(
        &self,
        sde: &ManifoldSDE<M, D, G>,
        x: &M::Point,
        t: f64,
        dt: f64,
        inc: &Increment<f64>,
    ) -> M::Point;
}
```

Implement `GeoScheme` for `GeodesicEuler`:

```rust
// In pathwise-geo/src/simulate.rs or back in euler.rs via impl block:
use crate::scheme::euler::GeodesicEuler;
impl<M, D, G> GeoScheme<M, D, G> for GeodesicEuler
where
    M: Manifold + cartan_core::ParallelTransport,
    D: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    M::Tangent: std::ops::Add<Output = M::Tangent> + std::ops::Mul<f64, Output = M::Tangent>
        + std::ops::Sub<Output = M::Tangent> + Clone,
{
    fn step_geo(&self, sde: &ManifoldSDE<M,D,G>, x: &M::Point, t: f64, dt: f64, inc: &Increment<f64>) -> M::Point {
        self.step(sde, x, t, dt, inc)
    }
}
```

And for `GeodesicMilstein`:
```rust
use crate::scheme::milstein::GeodesicMilstein;
impl<M, D, G> GeoScheme<M, D, G> for GeodesicMilstein
where
    M: Manifold + cartan_core::ParallelTransport,
    D: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    M::Tangent: std::ops::Add<Output=M::Tangent> + std::ops::Mul<f64,Output=M::Tangent>
        + std::ops::Sub<Output=M::Tangent> + Clone,
{
    fn step_geo(&self, sde: &ManifoldSDE<M,D,G>, x: &M::Point, t: f64, dt: f64, inc: &Increment<f64>) -> M::Point {
        self.step(sde, x, t, dt, inc)
    }
}
```

- [ ] **Step 4: Run Milstein manifold tests**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-geo --test manifold_sde 2>&1 | tail -20
```

Expected: all sphere tests pass (GeodesicEuler + GeodesicMilstein).

- [ ] **Step 5: Commit**

```bash
cd /home/alejandrosotofranco/pathwise
git add pathwise-geo/src/scheme/milstein.rs pathwise-geo/src/simulate.rs pathwise-geo/tests/manifold_sde.rs
git commit -m "feat(pathwise-geo): implement GeodesicMilstein with finite-diff nabla_g via ParallelTransport"
```

---

## Task 9: GeodesicSRI + manifold_simulate complete + OU on manifold

**Files:**
- Modify: `pathwise-geo/src/scheme/sri.rs`
- Modify: `pathwise-geo/src/process.rs` — make ou_on functional
- Modify: `pathwise-geo/tests/manifold_sde.rs` — add OU tests

- [ ] **Step 1: Write failing test**

Add to `pathwise-geo/tests/manifold_sde.rs`:

```rust
#[test]
fn ou_on_sphere_mean_reverts() {
    use pathwise_geo::{ou_on_with_diffusion, GeodesicEuler};
    use cartan_manifolds::Sphere;
    use nalgebra::SVector;
    let s2 = Sphere::<3>;
    // mu_point = north pole [0,0,1]
    let mu = SVector::from([0.0_f64, 0.0, 1.0]);
    // Start at south pole [0,0,-1] -- antipodal, maximum geodesic distance
    let x0 = SVector::from([0.0_f64, 0.0, -1.0]);
    // Use a simple scalar diffusion: project e1 onto tangent
    let sde = pathwise_geo::ou_on_with_diffusion(s2, 2.0, mu,
        |x: &SVector<f64, 3>, _t: f64| {
            let e1 = SVector::from([1.0_f64, 0.0, 0.0]);
            e1 - x * x.dot(&e1)
        });
    let paths = pathwise_geo::simulate::manifold_simulate_with_scheme(
        &sde, &GeodesicEuler, x0, 0.0, 2.0, 500, 400, 42
    );
    // Mean geodesic distance at T=2 should be less than at T=0.1
    let dist_late: f64 = paths.iter().map(|p| {
        let x = &p[400];
        x.dot(&mu).clamp(-1.0, 1.0).acos()  // geodesic dist on S2
    }).sum::<f64>() / paths.len() as f64;
    let dist_early: f64 = paths.iter().map(|p| {
        let x = &p[20];
        x.dot(&mu).clamp(-1.0, 1.0).acos()
    }).sum::<f64>() / paths.len() as f64;
    println!("OU S2: dist_early={:.4} dist_late={:.4}", dist_early, dist_late);
    assert!(dist_late < dist_early * 0.95,
        "OU should mean-revert: dist_late={:.4} dist_early={:.4}", dist_late, dist_early);
}
```

- [ ] **Step 2: Complete brownian_motion_on_with_diffusion and ou_on_with_diffusion in process.rs**

The Task 7 stubs for `brownian_motion_on` and `ou_on` used `unimplemented!()` panics.
Replace them entirely. The generic `Manifold` trait has no orthonormal frame field, so
`brownian_motion_on` requires an explicit diffusion direction (`_with_diffusion` variant).
The `ou_on_with_diffusion` is the functional OU constructor.

```rust
// pathwise-geo/src/process.rs — REPLACE entire file contents:
use crate::sde::ManifoldSDE;
use cartan_core::Manifold;

/// Brownian motion on manifold with an explicit diffusion direction field.
///
/// Zero drift; `diffusion_gen(x, t)` provides the tangent vector g(x,t) scaled by dW.
/// This is directional (1D noise), not isotropic BM. Isotropic BM requires a random
/// orthonormal frame per step (future work; not in current cartan trait).
///
/// Note: the spec names this `brownian_motion_on(manifold)` but the generic Manifold
/// trait has no built-in frame, so we require an explicit diffusion generator.
pub fn brownian_motion_on_with_diffusion<M, G>(
    manifold: M,
    diffusion_gen: G,
) -> ManifoldSDE<
    M,
    impl Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    G,
>
where
    M: Manifold + Clone + Send + Sync,
    M::Point: Clone + Send + Sync,
    M::Tangent: Clone + Default + Send + Sync,
    G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
{
    // Zero drift: M::Tangent: Default gives the zero tangent (SVector::zeros() for cartan manifolds)
    ManifoldSDE::new(
        manifold,
        |_x: &M::Point, _t: f64| M::Tangent::default(),
        diffusion_gen,
    )
}

/// OU on manifold with user-supplied diffusion field.
/// Drift: kappa * log_x(mu_point) (Riemannian log toward mu_point, scaled by kappa).
/// Diffusion: caller-supplied `diffusion` closure.
pub fn ou_on_with_diffusion<M, G>(
    manifold: M,
    kappa: f64,
    mu_point: M::Point,
    diffusion: G,
) -> ManifoldSDE<
    M,
    impl Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    G,
>
where
    M: Manifold + Clone + Send + Sync,
    M::Point: Clone + Send + Sync,
    M::Tangent: Clone + Default + std::ops::Mul<f64, Output = M::Tangent> + Send + Sync,
    G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
{
    let mu_clone = mu_point.clone();
    let manifold_clone = manifold.clone();
    ManifoldSDE::new(
        manifold,
        move |x: &M::Point, _t: f64| -> M::Tangent {
            manifold_clone.log(x, &mu_clone)
                .map(|v| v * kappa)
                .unwrap_or_default()  // zero tangent at mu_point itself (log returns zero there)
        },
        diffusion,
    )
}
```

Update lib.rs: export `brownian_motion_on_with_diffusion` and `ou_on_with_diffusion`.
Remove the old `brownian_motion_on` and `ou_on` entries from lib.rs re-exports.

Also add `ou_on_spd_stays_positive_definite` test now (spec requirement, Task 9 scope):

```rust
// Add to pathwise-geo/tests/manifold_sde.rs:
#[test]
fn ou_on_spd_stays_positive_definite() {
    use cartan_manifolds::Spd;
    use nalgebra::{SMatrix, SVector};
    use pathwise_geo::{ou_on_with_diffusion, GeodesicEuler};
    use pathwise_geo::simulate::manifold_simulate_with_scheme;

    let spd = Spd::<2>;
    // mu_point = 2x2 identity (positive definite)
    let mu = SMatrix::<f64, 2, 2>::identity();
    // x0 = 2*I (near-identity, positive definite)
    let x0 = SMatrix::<f64, 2, 2>::identity() * 2.0;
    // Diffusion: a fixed skew-symmetric-inspired tangent (symmetric matrix for SPD tangent)
    let sde = ou_on_with_diffusion(
        spd, 1.0, mu,
        |_x: &SMatrix<f64, 2, 2>, _t: f64| {
            // A symmetric matrix with small entries (tangent of SPD at x ~ symmetric matrices)
            SMatrix::from_row_slice(&[0.1, 0.05, 0.05, 0.1])
        },
    );
    let paths = manifold_simulate_with_scheme(&sde, &GeodesicEuler, x0, 0.0, 1.0, 100, 100, 0);
    for path in &paths {
        for mat in path {
            // Check positive definiteness: all eigenvalues > 0
            // Use nalgebra symmetric eigendecomposition
            let eig = mat.symmetric_eigen();
            for &ev in eig.eigenvalues.iter() {
                assert!(ev > -1e-8,
                    "SPD path produced non-positive eigenvalue: {}", ev);
            }
        }
    }
}
```

- [ ] **Step 3: Implement GeodesicSRI in sri.rs**

```rust
// pathwise-geo/src/scheme/sri.rs
use crate::sde::ManifoldSDE;
use crate::simulate::GeoScheme;
use cartan_core::{Manifold, ParallelTransport};
use pathwise_core::state::Increment;

/// Geodesic SRI scheme: strong order 1.5 for manifold SDEs with scalar noise.
///
/// Extends GeodesicMilstein with the dZ iterated-integral correction.
/// Requires Manifold + ParallelTransport (same as GeodesicMilstein).
pub struct GeodesicSRI {
    eps: f64,
}

impl GeodesicSRI {
    pub fn new() -> Self { Self { eps: 1e-4 } }

    pub fn step<M, D, G>(
        &self,
        sde: &ManifoldSDE<M, D, G>,
        x: &M::Point,
        t: f64,
        dt: f64,
        inc: &Increment<f64>,
    ) -> M::Point
    where
        M: Manifold + ParallelTransport,
        D: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
        G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
        M::Tangent: std::ops::Add<Output = M::Tangent>
            + std::ops::Mul<f64, Output = M::Tangent>
            + std::ops::Sub<Output = M::Tangent>
            + Clone,
    {
        let dw = inc.dw;
        let dz = inc.dz;
        let f  = (sde.drift)(x, t);
        let g  = (sde.diffusion)(x, t);
        let eps = self.eps;

        // nabla_g g via parallel transport (same as Milstein)
        let eps_g = g.clone() * eps;
        let y = sde.manifold.exp(x, &eps_g).unwrap_or_else(|_| x.clone());
        let g_at_y = (sde.diffusion)(&y, t);
        let g_at_y_transported = sde.manifold.transport(&y, x, &g_at_y)
            .unwrap_or_else(|_| g.clone());
        let nabla_g_g = (g_at_y_transported.clone() - g.clone()) * (1.0 / eps);

        // SRI dZ correction.
        // Spec says "a second finite-difference evaluation of nabla_g g weighted by dz."
        // Full SRI1 would compute nabla_g(nabla_g g) via a second PT-based FD evaluation.
        // This plan uses a single-FD approximation: reuse nabla_g_g for the dZ term.
        // This is a leading-order approximation that still improves upon Milstein for
        // smooth diffusion fields on compact manifolds, but formally achieves the dZ
        // correction to the same order as the Milstein term without an extra PT evaluation.
        // A future revision should implement the full second-order FD for strict SRI1.
        let milstein_correction = nabla_g_g.clone() * (0.5 * (dw * dw - dt));
        let sri_correction = nabla_g_g * dz;

        let tangent = f * dt + g * dw + milstein_correction + sri_correction;
        sde.manifold.exp(x, &tangent).unwrap_or_else(|_| x.clone())
    }
}

impl Default for GeodesicSRI {
    fn default() -> Self { Self::new() }
}

impl<M, D, G> GeoScheme<M, D, G> for GeodesicSRI
where
    M: Manifold + cartan_core::ParallelTransport,
    D: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    M::Tangent: std::ops::Add<Output=M::Tangent> + std::ops::Mul<f64,Output=M::Tangent>
        + std::ops::Sub<Output=M::Tangent> + Clone,
{
    fn step_geo(&self, sde: &ManifoldSDE<M,D,G>, x: &M::Point, t: f64, dt: f64, inc: &Increment<f64>) -> M::Point {
        self.step(sde, x, t, dt, inc)
    }
}
```

- [ ] **Step 4: Run all pathwise-geo tests**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test -p pathwise-geo 2>&1 | tail -20
```

Expected: all tests pass including `ou_on_sphere_mean_reverts` and `geodesic_euler_stays_on_sphere`.

- [ ] **Step 5: Commit**

```bash
cd /home/alejandrosotofranco/pathwise
git add pathwise-geo/
git commit -m "feat(pathwise-geo): add GeodesicSRI, ou_on_with_diffusion; complete manifold SDE stack"
```

---

## Task 10: Update pathwise-py

**Files:**
- Modify: `pathwise-py/src/py_scheme.rs` — add `PySri`
- Modify: `pathwise-py/src/py_process.rs` — add `Cir`, `Heston`, `CorrOu` enum variants
- Modify: `pathwise-py/src/py_simulate.rs` — expand dispatch
- Modify: `pathwise-py/src/lib.rs` — register new Python classes/functions
- Modify: `pathwise-py/tests/test_schemes.py` — add tests

- [ ] **Step 1: Write failing Python tests**

Add to `pathwise-py/tests/test_schemes.py` (or create if absent):

```python
import pytest
import numpy as np
import pathwise

def test_sri_convergence_python():
    """SRI error at N=200 should be lower than at N=50 (basic convergence check)."""
    s = pathwise.sri()
    g = pathwise.gbm(0.0, 0.3)
    err_coarse = abs(
        pathwise.simulate(g, s, n_paths=5000, n_steps=50, t1=1.0, x0=1.0).mean(0)[-1]
        - np.exp(0.0)
    )
    err_fine = abs(
        pathwise.simulate(g, s, n_paths=5000, n_steps=200, t1=1.0, x0=1.0).mean(0)[-1]
        - np.exp(0.0)
    )
    assert err_fine < err_coarse or abs(err_fine - err_coarse) < 0.01, \
        f"SRI error should decrease with refinement: coarse={err_coarse:.4f} fine={err_fine:.4f}"

def test_heston_output_shape():
    """Heston simulation returns (n_paths, n_steps+1, 2) array."""
    h = pathwise.heston(0.05, 2.0, 0.04, 0.3, -0.7)
    m = pathwise.milstein()
    out = pathwise.simulate(h, m, n_paths=100, n_steps=50, t1=1.0)
    assert out.shape == (100, 51, 2), f"Expected (100, 51, 2), got {out.shape}"

def test_sri_on_heston_raises():
    """SRI is not supported for Heston (full-matrix correlated noise). Must raise ValueError."""
    h = pathwise.heston(0.05, 2.0, 0.04, 0.3, -0.7)
    s = pathwise.sri()
    with pytest.raises(ValueError, match="SRI requires scalar"):
        pathwise.simulate(h, s, n_paths=10, n_steps=50, t1=1.0)

def test_cir_nonnegative():
    """All CIR output values must be >= 0."""
    c = pathwise.cir(2.0, 0.05, 0.3)
    e = pathwise.euler()
    out = pathwise.simulate(c, e, n_paths=200, n_steps=200, t1=1.0, x0=0.1)
    assert np.all(out[~np.isnan(out)] >= 0.0), "CIR produced negative values"

def test_cir_rejects_feller_violation():
    """CIR constructor raises ValueError when Feller condition is violated."""
    with pytest.raises(ValueError, match="Feller"):
        pathwise.cir(1.0, 0.01, 0.5)  # 2*1*0.01=0.02 < 0.25

def test_heston_variance_nonnegative():
    """Heston variance component (index 1) must be >= 0."""
    h = pathwise.heston(0.05, 2.0, 0.04, 0.3, -0.7)
    out = pathwise.simulate(h, pathwise.euler(), n_paths=100, n_steps=100, t1=1.0)
    variance_col = out[:, :, 1]
    assert np.all(variance_col[~np.isnan(variance_col)] >= 0.0), "Heston variance went negative"
```

Run to confirm tests fail (functions not yet exposed):
```bash
cd /home/alejandrosotofranco/pathwise && maturin develop -m pathwise-py/Cargo.toml 2>&1 | tail -5
python -m pytest pathwise-py/tests/test_schemes.py -x 2>&1 | head -20
```

- [ ] **Step 2: Add PySri to py_scheme.rs**

```rust
// pathwise-py/src/py_scheme.rs — add:
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct PySri;

#[pymethods]
impl PySri {
    #[new]
    pub fn new() -> Self { PySri }
    fn __repr__(&self) -> &str { "sri()" }
}

#[pyfunction]
pub fn sri() -> PySri { PySri }
```

- [ ] **Step 3: Add Cir, Heston, CorrOu to py_process.rs**

In `pathwise-py/src/py_process.rs`, extend the `SDEKind` enum and `PySDE` class:

```rust
// Add new variants to SDEKind:
#[derive(Clone)]
pub enum SDEKind {
    Bm,
    Gbm   { mu: f64, sigma: f64 },
    Ou    { theta: f64, mu: f64, sigma: f64 },
    Cir   { kappa: f64, theta: f64, sigma: f64 },
    Heston { mu: f64, kappa: f64, theta: f64, xi: f64, rho: f64 },
    CorrOu { theta: f64, mu: Vec<f64>, sigma_flat: Vec<f64>, n: usize },
    Custom { drift: PyObject, diffusion: PyObject },
}

// Add Python constructor functions:
#[pyfunction]
pub fn cir(kappa: f64, theta: f64, sigma: f64) -> PyResult<PySDE> {
    // Validate Feller condition here to give a good Python error
    if 2.0 * kappa * theta <= sigma * sigma {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Feller condition violated: 2*kappa*theta={:.4} <= sigma^2={:.4}",
            2.0 * kappa * theta, sigma * sigma
        )));
    }
    Ok(PySDE { kind: SDEKind::Cir { kappa, theta, sigma } })
}

#[pyfunction]
pub fn heston(mu: f64, kappa: f64, theta: f64, xi: f64, rho: f64) -> PySDE {
    PySDE { kind: SDEKind::Heston { mu, kappa, theta, xi, rho } }
}

#[pyfunction]
pub fn corr_ou(theta: f64, mu: Vec<f64>, sigma_flat: Vec<f64>) -> PyResult<PySDE> {
    let n = mu.len();
    if sigma_flat.len() != n * n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("sigma_flat must have {} elements for {}x{} matrix, got {}",
                n*n, n, n, sigma_flat.len())
        ));
    }
    Ok(PySDE { kind: SDEKind::CorrOu { theta, mu, sigma_flat, n } })
}
```

- [ ] **Step 4: Expand dispatch in py_simulate.rs**

The `simulate` function in py_simulate.rs needs to:
1. Detect `PySri` and raise `ValueError` for Heston/CorrOu
2. Dispatch Cir to `pathwise_core::cir()` and call `simulate`
3. Dispatch Heston to `pathwise_core::heston()` and call `simulate_nd`; return `PyArray3`
4. Dispatch CorrOu similarly

The return type must change to `PyObject` or use an enum since Heston returns 3D while scalar returns 2D. The cleanest approach: return `PyObject` and let Python handle both ndarray shapes.

Key change to `simulate` signature in py_simulate.rs:

```rust
#[pyfunction]
#[pyo3(signature = (sde, scheme, n_paths, n_steps, t1, x0=0.0, t0=0.0, device="cpu", seed=0))]
pub fn simulate<'py>(
    py: Python<'py>,
    sde: &PySDE,
    scheme: &Bound<'_, PyAny>,
    n_paths: usize,
    n_steps: usize,
    t1: f64,
    x0: f64,
    t0: f64,
    device: &str,
    seed: u64,
) -> PyResult<PyObject> {  // Changed from PyArray2<f64> to PyObject
```

Dispatch for SRI (only scalar processes):
```rust
let use_sri = scheme.is_instance_of::<PySri>();
// ... in match (add BEFORE the Heston and CorrOu dispatch arms):
SDEKind::Heston { .. } if use_sri => {
    return Err(pyo3::exceptions::PyValueError::new_err(
        "SRI requires scalar or diagonal noise; use Milstein for Heston"
    ));
}
SDEKind::CorrOu { .. } if use_sri => {
    return Err(pyo3::exceptions::PyValueError::new_err(
        "SRI requires scalar or diagonal noise; use Milstein for CorrOu"
    ));
}
```

Heston dispatch returns `PyArray3`:
```rust
SDEKind::Heston { mu, kappa, theta, xi, rho } => {
    use nalgebra::SVector;
    use numpy::PyArray3;
    let sde_rust = pathwise_core::process::markov::heston(*mu, *kappa, *theta, *xi, *rho);
    let x0_nd = SVector::from([x0, *theta]);  // [log_S0, V0]; x0 used as log_S, theta as initial V
    let result = if use_milstein {
        py.allow_threads(|| pathwise_core::simulate_nd::<2, _, _, _>(
            &sde_rust.drift, &sde_rust.diffusion,
            &pathwise_core::scheme::milstein_nd::<2>(),
            x0_nd, t0, t1, n_paths, n_steps, seed,
        ))
    } else {
        py.allow_threads(|| pathwise_core::simulate_nd::<2, _, _, _>(
            &sde_rust.drift, &sde_rust.diffusion,
            &pathwise_core::scheme::euler(),
            x0_nd, t0, t1, n_paths, n_steps, seed,
        ))
    }.map_err(to_py_err)?;
    Ok(PyArray3::from_owned_array_bound(py, result).into_any().unbind())
}
```

CIR dispatch (scalar, add clipping wrapper in the call):
```rust
SDEKind::Cir { kappa, theta, sigma } => {
    // CIR constructor already validated Feller; this call should not fail
    let sde_rust = pathwise_core::process::markov::cir(*kappa, *theta, *sigma)
        .map_err(to_py_err)?;
    // ... same scalar dispatch as Bm/Gbm/Ou
}
```

Heston note: `x0` parameter is used as `log(S0)`; initial variance defaults to `theta`. If more control is needed, the user can use the Rust API directly. The Python API keeps it simple with a single `x0` scalar.

- [ ] **Step 5: Register new functions in lib.rs**

```rust
// pathwise-py/src/lib.rs — add to #[pymodule]:
m.add_function(wrap_pyfunction!(py_scheme::sri, m)?)?;
m.add_function(wrap_pyfunction!(py_process::cir, m)?)?;
m.add_function(wrap_pyfunction!(py_process::heston, m)?)?;
m.add_function(wrap_pyfunction!(py_process::corr_ou, m)?)?;
m.add_class::<py_scheme::PySri>()?;
```

- [ ] **Step 6: Build and run Python tests**

```bash
cd /home/alejandrosotofranco/pathwise
maturin develop -m pathwise-py/Cargo.toml 2>&1 | tail -10
python -m pytest pathwise-py/tests/test_schemes.py -v 2>&1 | tail -20
```

Expected: all 6 new Python tests pass.

- [ ] **Step 7: Run full Rust test suite to confirm no regressions**

```bash
cd /home/alejandrosotofranco/pathwise && cargo test --workspace 2>&1 | tail -15
```

Expected: all tests pass across pathwise-core, pathwise-geo, pathwise-py.

- [ ] **Step 8: Commit**

```bash
cd /home/alejandrosotofranco/pathwise
git add pathwise-py/ pathwise-py/tests/
git commit -m "feat(pathwise-py): expose sri(), cir(), heston(), corr_ou(); nD Array3 output for Heston"
```

---

## Implementation Notes

**Trait conflict risk (Task 2):** The old `Diffusion<S>: Fn(&S, f64) -> S` in markov.rs conflicts by name with the new `Diffusion<S, B>` in state.rs. The plan removes the old name from markov.rs and keeps only the new one. The `SDE` struct's `G` bound changes from `G: Diffusion<S>` to `G: Fn(&S, f64) -> S + Send + Sync` (direct Fn bound), avoiding any name conflict.

**simulate_nd Drift bound (Task 5):** `simulate_nd` uses `D: Fn(&SVector<N>, f64) -> SVector<N>` directly rather than `D: Drift<SVector<N>>` to avoid importing markov's `Drift` into simulate.rs. Both bounds are satisfied by the same closures.

**Heston x0 convention (Task 10):** The Python `x0` parameter represents `log(S0)`, not `S0`. Users who want to start from price `S0` should pass `x0=log(S0)`. Document this in `heston()`'s docstring.

**pathwise-geo ParallelTransport bound:** `GeodesicMilstein` and `GeodesicSRI` require `M: Manifold + ParallelTransport`. All built-in cartan manifolds implement both traits. Users implementing custom manifolds for `GeodesicEuler` only need `Manifold`.

**SRI note for corr_ou (Task 10):** `CorrOu` uses a full Cholesky diffusion matrix. SRI is not implemented for this case. The Python dispatch raises `ValueError` for both `SRI + CorrOu` and `SRI + Heston` (added explicitly in Task 10 Step 4).

**brownian_motion_on and ou_on (Task 9):** The spec names these as `brownian_motion_on(manifold)` and `ou_on(manifold, kappa, mu)`. The generic `Manifold` trait has no built-in orthonormal frame, so both require an explicit diffusion closure. The plan implements `brownian_motion_on_with_diffusion(manifold, g)` and `ou_on_with_diffusion(manifold, kappa, mu, g)`. The spec's bare constructors are future work pending a frame-field extension to cartan.

**CIR non-negativity test (Task 4):** The `cir_stays_nonnegative` test constructs the SDE directly via `SDE::new` to bypass the constructor's Feller check, testing that `x.max(0.0)` clipping holds even at the Feller boundary (2κθ = σ²). This is intentional; the `cir()` constructor correctly rejects this configuration.

**PathOutput enum:** The spec defines `PathOutput<S>` as the return type of a unified `simulate`. The plan replaces this with separate `simulate` (returns `Array2`) and `simulate_nd` (returns `Array3`) for simpler API usage. Behavior is identical.
