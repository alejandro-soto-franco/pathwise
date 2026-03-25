use ndarray::{Array2, Array3};
use rand::SeedableRng;
use rayon::prelude::*;

use crate::error::PathwiseError;
use crate::process::markov::Drift;
use crate::rng::splitmix64;
use crate::scheme::Scheme;
use crate::state::{Diffusion, NoiseIncrement};

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
        return Err(PathwiseError::InvalidParameters(
            "n_paths and n_steps must be > 0".into(),
        ));
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
                if !x.is_finite() {
                    x = f64::NAN;
                }
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
    x0: nalgebra::SVector<f64, N>,
    t0: f64,
    t1: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> Result<Array3<f64>, PathwiseError>
where
    D: Fn(&nalgebra::SVector<f64, N>, f64) -> nalgebra::SVector<f64, N> + Send + Sync,
    G: Diffusion<nalgebra::SVector<f64, N>, nalgebra::SVector<f64, N>> + Sync,
    SC: Scheme<nalgebra::SVector<f64, N>, Noise = nalgebra::SVector<f64, N>>,
{
    if n_paths == 0 || n_steps == 0 {
        return Err(PathwiseError::InvalidParameters(
            "n_paths and n_steps must be > 0".into(),
        ));
    }
    if t1 <= t0 {
        return Err(PathwiseError::InvalidParameters("t1 must be > t0".into()));
    }

    let dt = (t1 - t0) / n_steps as f64;
    let base_seed = splitmix64(seed);

    let rows: Vec<Vec<nalgebra::SVector<f64, N>>> = (0..n_paths)
        .into_par_iter()
        .map(|i| {
            let path_seed = splitmix64(base_seed.wrapping_add(i as u64));
            let mut rng = rand::rngs::SmallRng::seed_from_u64(path_seed);
            let mut path = Vec::with_capacity(n_steps + 1);
            let mut x = x0;
            path.push(x);
            for step in 0..n_steps {
                let t = t0 + step as f64 * dt;
                let inc = <nalgebra::SVector<f64, N> as NoiseIncrement>::sample(&mut rng, dt);
                x = scheme.step(drift, diffusion, &x, t, dt, &inc);
                // Check for non-finite in any component; freeze path
                if x.iter().any(|v| !v.is_finite()) {
                    x = nalgebra::SVector::from_fn(|_, _| f64::NAN);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::markov::{bm, ou};
    use crate::scheme::euler;

    #[test]
    fn simulate_returns_correct_shape() {
        let b = bm();
        let out = simulate(&b.drift, &b.diffusion, &euler(), 0.0, 0.0, 1.0, 10, 100, 0).unwrap();
        assert_eq!(out.shape(), &[10, 101]);
    }

    #[test]
    fn simulate_first_column_is_x0() {
        let b = bm();
        let out = simulate(&b.drift, &b.diffusion, &euler(), 2.5, 0.0, 1.0, 50, 200, 0).unwrap();
        for i in 0..50 {
            assert!((out[[i, 0]] - 2.5).abs() < 1e-12, "path {} x0 wrong", i);
        }
    }

    #[test]
    fn simulate_errors_on_zero_paths() {
        let b = bm();
        let result = simulate(&b.drift, &b.diffusion, &euler(), 0.0, 0.0, 1.0, 0, 100, 0);
        assert!(result.is_err());
    }

    #[test]
    fn simulate_ou_mean_reverts_on_average() {
        let o = ou(5.0, 3.0, 0.1);
        let out = simulate(
            &o.drift,
            &o.diffusion,
            &euler(),
            0.0,
            0.0,
            1.0,
            2000,
            500,
            0,
        )
        .unwrap();
        let last_col = out.column(500);
        let mean: f64 = last_col.iter().sum::<f64>() / 2000.0;
        assert!((mean - 3.0).abs() < 0.1, "OU mean={} expected ~3.0", mean);
    }

    #[test]
    fn simulate_is_reproducible_with_same_seed() {
        let b = bm();
        let r1 = simulate(&b.drift, &b.diffusion, &euler(), 0.0, 0.0, 1.0, 10, 50, 42).unwrap();
        let r2 = simulate(&b.drift, &b.diffusion, &euler(), 0.0, 0.0, 1.0, 10, 50, 42).unwrap();
        assert_eq!(r1, r2, "same seed must produce identical output");
    }

    #[test]
    fn simulate_different_seeds_produce_different_paths() {
        let b = bm();
        let r1 = simulate(&b.drift, &b.diffusion, &euler(), 0.0, 0.0, 1.0, 10, 50, 0).unwrap();
        let r2 = simulate(&b.drift, &b.diffusion, &euler(), 0.0, 0.0, 1.0, 10, 50, 1).unwrap();
        assert_ne!(r1, r2, "different seeds must produce different paths");
    }
}
