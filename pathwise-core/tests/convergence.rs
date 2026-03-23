use pathwise_core::process::markov::gbm;
use pathwise_core::scheme::{euler, milstein};
use pathwise_core::simulate::simulate;

/// Weak convergence check for Euler-Maruyama on GBM.
/// E[X_T] = X_0 * exp(mu*T) — error should decrease as n_steps increases.
#[test]
fn euler_weak_error_decreases_with_dt() {
    let mu = 0.05;
    let sigma = 0.2;
    let x0 = 1.0;
    let t1 = 1.0;
    let n_paths = 5000;
    let exact_mean = x0 * (mu * t1 as f64).exp();

    let step_counts = [50usize, 100, 200, 400];
    let errors: Vec<f64> = step_counts.iter().map(|&n_steps| {
        let g = gbm(mu, sigma);
        let out = simulate(&g.drift, &g.diffusion, &euler(), x0, 0.0, t1, n_paths, n_steps).unwrap();
        let last_col = out.column(n_steps);
        let sample_mean: f64 = last_col.iter().sum::<f64>() / n_paths as f64;
        (sample_mean - exact_mean).abs()
    }).collect();

    assert!(errors[1] < errors[0] * 1.5, "error should decrease with finer grid");
    assert!(errors[3] < errors[0], "finest grid should have smaller error than coarsest");
}

#[test]
fn milstein_weak_error_decreases_with_dt() {
    let mu = 0.05;
    let sigma = 0.2;
    let x0 = 1.0;
    let t1 = 1.0;
    let n_paths = 5000;
    let exact_mean = x0 * (mu * t1 as f64).exp();

    let step_counts = [50usize, 100, 200, 400];
    let errors: Vec<f64> = step_counts.iter().map(|&n_steps| {
        let g = gbm(mu, sigma);
        let out = simulate(&g.drift, &g.diffusion, &milstein(), x0, 0.0, t1, n_paths, n_steps).unwrap();
        let last_col = out.column(n_steps);
        let sample_mean: f64 = last_col.iter().sum::<f64>() / n_paths as f64;
        (sample_mean - exact_mean).abs()
    }).collect();

    assert!(errors[3] < errors[0], "milstein finest grid should have smaller error than coarsest");
}
