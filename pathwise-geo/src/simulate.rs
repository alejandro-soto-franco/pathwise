use crate::sde::ManifoldSDE;
use crate::scheme::euler::GeodesicEuler;
use crate::scheme::milstein::GeodesicMilstein;
use cartan_core::{Manifold, ParallelTransport};
use pathwise_core::state::{Increment, NoiseIncrement};
use rand::SeedableRng;

// Same derivation as pathwise_core::rng::splitmix64 — must stay in sync.
// pathwise-geo cannot take a dependency on pathwise-core::rng (private module),
// so this is an intentional local copy.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Simulate manifold SDE paths using GeodesicEuler.
///
/// Returns `Vec<Vec<M::Point>>` with outer index = path, inner index = time step.
/// Each inner vec has `n_steps + 1` points (including the initial condition x0).
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
    M::Tangent: Clone + Send + Sync,
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

/// Internal trait to unify geodesic scheme step dispatch.
///
/// Implementors provide a single `step_geo` method that advances x by one step.
/// The M: ParallelTransport bound is required for the GeodesicMilstein impl;
/// GeodesicEuler satisfies it but does not use the transport operation.
pub trait GeoScheme<M, D, G>
where
    M: Manifold + ParallelTransport,
    D: Fn(&M::Point, f64) -> M::Tangent,
    G: Fn(&M::Point, f64) -> M::Tangent,
{
    /// Advance x by one scheme step.
    fn step_geo(
        &self,
        sde: &ManifoldSDE<M, D, G>,
        x: &M::Point,
        t: f64,
        dt: f64,
        inc: &Increment<f64>,
    ) -> M::Point;
}

impl<M, D, G> GeoScheme<M, D, G> for GeodesicEuler
where
    M: Manifold + ParallelTransport,
    D: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
{
    fn step_geo(
        &self,
        sde: &ManifoldSDE<M, D, G>,
        x: &M::Point,
        t: f64,
        dt: f64,
        inc: &Increment<f64>,
    ) -> M::Point {
        self.step(sde, x, t, dt, inc)
    }
}

impl<M, D, G> GeoScheme<M, D, G> for GeodesicMilstein
where
    M: Manifold + ParallelTransport,
    D: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
{
    fn step_geo(
        &self,
        sde: &ManifoldSDE<M, D, G>,
        x: &M::Point,
        t: f64,
        dt: f64,
        inc: &Increment<f64>,
    ) -> M::Point {
        self.step(sde, x, t, dt, inc)
    }
}

/// Simulate manifold SDE paths using any `GeoScheme` implementor.
///
/// Generic version of `manifold_simulate` that works with GeodesicEuler,
/// GeodesicMilstein, or any future scheme implementing `GeoScheme`.
/// Returns `Vec<Vec<M::Point>>` with outer index = path, inner index = time step.
/// Each inner vec has `n_steps + 1` points (including the initial condition x0).
#[allow(clippy::too_many_arguments)]
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
    M: Manifold + ParallelTransport + Clone + Send + Sync,
    M::Point: Clone + Send + Sync,
    M::Tangent: Clone + Send + Sync,
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

/// Flatten manifold SDE paths into a 3-D array by applying `log` from a reference point.
///
/// Shape: `(n_paths, n_steps+1, tangent_dim)`.
/// Requires `M::Tangent: AsRef<[f64]>`, which holds for `nalgebra::SVector<f64, N>`.
/// If `log` fails for a point (cut locus), that entry is filled with NaN.
pub fn paths_to_array<M>(
    paths: &[Vec<M::Point>],
    manifold: &M,
    ref_point: &M::Point,
) -> ndarray::Array3<f64>
where
    M: Manifold,
    M::Point: Clone,
    M::Tangent: AsRef<[f64]>,
{
    let n_paths = paths.len();
    if n_paths == 0 {
        return ndarray::Array3::zeros((0, 0, 0));
    }
    let n_steps_plus1 = paths[0].len();
    // Determine tangent dimension from the first valid log.
    let dim = manifold
        .log(ref_point, &paths[0][0])
        .map(|v| v.as_ref().len())
        .unwrap_or(0);
    if dim == 0 {
        return ndarray::Array3::zeros((n_paths, n_steps_plus1, 0));
    }
    let mut out = ndarray::Array3::from_elem((n_paths, n_steps_plus1, dim), f64::NAN);
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
