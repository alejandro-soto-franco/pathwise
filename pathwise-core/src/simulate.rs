use ndarray::Array2;
use rayon::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::error::PathwiseError;
use crate::process::markov::{Drift, Diffusion};
use crate::scheme::Scheme;

/// Simulate `n_paths` paths of a scalar SDE from `t0` to `t1` with `n_steps` steps.
///
/// Returns Array2<f64> of shape `(n_paths, n_steps + 1)`.
/// Row `i` is path `i`; column `j` is the state at time `t0 + j*dt`.
pub fn simulate<D, G, SC>(
    drift: &D,
    diffusion: &G,
    scheme: &SC,
    x0: f64,
    t0: f64,
    t1: f64,
    n_paths: usize,
    n_steps: usize,
) -> Result<Array2<f64>, PathwiseError>
where
    D: Drift<f64>,
    G: Diffusion<f64>,
    SC: Scheme,
{
    if n_paths == 0 || n_steps == 0 {
        return Err(PathwiseError::InvalidParameters(
            "n_paths and n_steps must be > 0".into(),
        ));
    }
    if t1 <= t0 {
        return Err(PathwiseError::InvalidParameters(
            "t1 must be > t0".into(),
        ));
    }

    let dt = (t1 - t0) / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Parallel: each path gets its own seeded RNG derived from its index
    let rows: Vec<Vec<f64>> = (0..n_paths)
        .into_par_iter()
        .map(|i| {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(i as u64);
            let mut path = Vec::with_capacity(n_steps + 1);
            let mut x = x0;
            path.push(x);
            for step in 0..n_steps {
                let t = t0 + step as f64 * dt;
                let dw = normal.sample(&mut rng) * sqrt_dt;
                x = scheme.step(drift, diffusion, x, t, dt, dw);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::markov::{bm, ou};
    use crate::scheme::euler;

    #[test]
    fn simulate_returns_correct_shape() {
        let b = bm();
        let out = simulate(&b.drift, &b.diffusion, &euler(), 0.0, 0.0, 1.0, 10, 100).unwrap();
        assert_eq!(out.shape(), &[10, 101]);
    }

    #[test]
    fn simulate_first_column_is_x0() {
        let b = bm();
        let out = simulate(&b.drift, &b.diffusion, &euler(), 2.5, 0.0, 1.0, 50, 200).unwrap();
        for i in 0..50 {
            assert!((out[[i, 0]] - 2.5).abs() < 1e-12, "path {} x0 wrong", i);
        }
    }

    #[test]
    fn simulate_errors_on_zero_paths() {
        let b = bm();
        let result = simulate(&b.drift, &b.diffusion, &euler(), 0.0, 0.0, 1.0, 0, 100);
        assert!(result.is_err());
    }

    #[test]
    fn simulate_ou_mean_reverts_on_average() {
        let o = ou(5.0, 3.0, 0.1);
        let out = simulate(&o.drift, &o.diffusion, &euler(), 0.0, 0.0, 1.0, 2000, 500).unwrap();
        let last_col = out.column(500);
        let mean: f64 = last_col.iter().sum::<f64>() / 2000.0;
        assert!((mean - 3.0).abs() < 0.1, "OU mean={} expected ~3.0", mean);
    }
}
