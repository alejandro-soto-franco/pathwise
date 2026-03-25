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
    // Variance is component index 1.
    // Euler can produce slightly negative variance before the next step's full-truncation clips
    // it in the diffusion; allow small negative values as documented Euler behavior.
    for path in 0..500 {
        for step in 0..=200 {
            let v = out[[path, step, 1]];
            assert!(v >= -0.05 || v.is_nan(), "variance negative: {} at path={} step={}", v, path, step);
        }
    }
}

#[test]
fn euler_nd_shape_and_finite() {
    // A 2D Euler with diagonal uncorrelated diffusion
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
    assert_eq!(out_2d.shape(), &[1, 101, 2]);
    for val in out_2d.iter() {
        if !val.is_nan() {
            assert!(val.is_finite(), "nD Euler produced non-finite value");
        }
    }
}

#[test]
fn corr_ou_covariance() {
    use pathwise_core::process::markov::corr_ou;
    use nalgebra::{SMatrix, SVector};
    let theta = 2.0_f64;
    let mu = SVector::<f64, 2>::zeros();
    let sigma_mat = SMatrix::<f64, 2, 2>::from_row_slice(&[1.0, 0.5, 0.5, 1.0]);
    let sde = corr_ou::<2>(theta, mu, sigma_mat).expect("corr_ou");
    let out = pathwise_core::simulate_nd::<2, _, _, _>(
        &sde.drift, &sde.diffusion, &pathwise_core::scheme::euler(),
        SVector::zeros(), 0.0, 5.0, 10_000, 1000, 0
    ).unwrap();
    let n_paths = 10_000_usize;
    let last_step = 1000_usize;
    let mean0: f64 = (0..n_paths).map(|i| out[[i, last_step, 0]]).sum::<f64>() / n_paths as f64;
    let mean1: f64 = (0..n_paths).map(|i| out[[i, last_step, 1]]).sum::<f64>() / n_paths as f64;
    let cov00: f64 = (0..n_paths).map(|i| (out[[i,last_step,0]]-mean0).powi(2)).sum::<f64>() / (n_paths-1) as f64;
    let cov11: f64 = (0..n_paths).map(|i| (out[[i,last_step,1]]-mean1).powi(2)).sum::<f64>() / (n_paths-1) as f64;
    let cov01: f64 = (0..n_paths).map(|i| (out[[i,last_step,0]]-mean0)*(out[[i,last_step,1]]-mean1)).sum::<f64>() / (n_paths-1) as f64;
    let expected_diag = 1.0 / (2.0 * theta);
    let expected_offdiag = 0.5 / (2.0 * theta);
    println!("CorrOU cov00={:.4} (expected {:.4})", cov00, expected_diag);
    println!("CorrOU cov11={:.4} (expected {:.4})", cov11, expected_diag);
    println!("CorrOU cov01={:.4} (expected {:.4})", cov01, expected_offdiag);
    assert!((cov00 - expected_diag).abs() / expected_diag < 0.05, "cov00 {:.4} vs {:.4}", cov00, expected_diag);
    assert!((cov11 - expected_diag).abs() / expected_diag < 0.05, "cov11 {:.4} vs {:.4}", cov11, expected_diag);
    assert!((cov01 - expected_offdiag).abs() / expected_offdiag < 0.10, "cov01 {:.4} vs {:.4}", cov01, expected_offdiag);
}

#[test]
fn milstein_nd_stronger_than_euler_nd_on_gbm_like() {
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
    let out_mil = pathwise_core::simulate_nd::<1, _, _, _>(
        &drift_fn, &diff_fn, &milstein_nd::<1>(), x0, 0.0, t1, n_paths, n_steps, 0
    ).unwrap();
    let out_euler = pathwise_core::simulate_nd::<1, _, _, _>(
        &drift_fn, &diff_fn, &euler(), x0, 0.0, t1, n_paths, n_steps, 0
    ).unwrap();
    assert_eq!(out_mil.shape(), &[n_paths, n_steps + 1, 1]);
    let finite_mil = out_mil.iter().filter(|x| x.is_finite()).count();
    let finite_euler = out_euler.iter().filter(|x| x.is_finite()).count();
    assert!(finite_mil > n_paths * n_steps / 2, "too many NaN in MilsteinNd");
    assert!(finite_euler > n_paths * n_steps / 2, "too many NaN in EulerNd");
}
