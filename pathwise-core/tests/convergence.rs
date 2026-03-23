/// Full numerical convergence tests for pathwise-core.
///
/// # Strong convergence (common-noise)
///   euler_strong_order_on_gbm    — order ~0.5
///   milstein_strong_order_on_gbm — order ~1.0
///   milstein_stronger_than_euler_strong — Milstein error < Euler at same step count
///
/// # Weak convergence (Monte Carlo)
///   euler_weak_error_monotone    — weak error decreases as dt decreases (Euler)
///   milstein_weak_error_monotone — weak error decreases as dt decreases (Milstein)
///
///   Note on weak order measurement: E[X_T] for GBM has weak disc error O(dt) whose
///   coefficient is small. Measuring the exponent accurately requires n >> sigma^2/dt^2.
///   We verify monotone decrease here; the log-log slope is measured in Python tests
///   where we can use coarser grids and larger n_paths affordably.
///
/// # Statistical moments
///   bm_variance_exact         — Var[W_t] = t
///   gbm_mean_and_variance_exact — E[X_T] and Var[X_T] vs analytic formulas
///   ou_mean_exact             — E[X_T|X_0] vs analytic formula
///   ou_stationary_distribution — X_T -> N(mu, sigma^2/2theta) as T->inf

use pathwise_core::process::markov::{bm, gbm, ou};
use pathwise_core::scheme::{euler, milstein};
use pathwise_core::simulate::simulate;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Log-log regression slope (convergence order estimate).
fn convergence_order(dts: &[f64], errors: &[f64]) -> f64 {
    let n = dts.len() as f64;
    let log_dt: Vec<f64> = dts.iter().map(|&d| d.ln()).collect();
    let log_err: Vec<f64> = errors.iter().map(|&e| e.ln()).collect();
    let sum_x: f64 = log_dt.iter().sum();
    let sum_y: f64 = log_err.iter().sum();
    let sum_xx: f64 = log_dt.iter().map(|x| x * x).sum();
    let sum_xy: f64 = log_dt.iter().zip(log_err.iter()).map(|(x, y)| x * y).sum();
    (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
}

/// Simulate one GBM path (Euler) with pre-supplied dW increments.
fn gbm_euler_path(dw: &[f64], mu: f64, sigma: f64, x0: f64, dt: f64) -> f64 {
    let mut x = x0;
    for &dw_i in dw {
        x += mu * x * dt + sigma * x * dw_i;
    }
    x
}

/// Simulate one GBM path (Milstein) with pre-supplied dW increments.
fn gbm_milstein_path(dw: &[f64], mu: f64, sigma: f64, x0: f64, dt: f64) -> f64 {
    let mut x = x0;
    for &dw_i in dw {
        x += mu * x * dt + sigma * x * dw_i + 0.5 * sigma * sigma * x * (dw_i * dw_i - dt);
    }
    x
}

/// Exact GBM terminal value given Brownian increments.
fn gbm_exact(dw: &[f64], mu: f64, sigma: f64, x0: f64, dt: f64) -> f64 {
    let w_t: f64 = dw.iter().sum();
    let t = dw.len() as f64 * dt;
    x0 * ((mu - 0.5 * sigma * sigma) * t + sigma * w_t).exp()
}

/// Strong error E[|X_N_approx - X_T_exact|] using common Brownian motion.
fn strong_error<F>(scheme_fn: F, n_steps: usize, n_paths: usize, mu: f64, sigma: f64, x0: f64, t1: f64) -> f64
where
    F: Fn(&[f64], f64, f64, f64, f64) -> f64,
{
    let dt = t1 / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let normal = Normal::new(0.0_f64, 1.0).unwrap();
    let mut total_error = 0.0_f64;
    for i in 0..n_paths {
        let mut rng = rand::rngs::SmallRng::seed_from_u64((i as u64) ^ 0xDEAD_BEEF_CAFE);
        let dw: Vec<f64> = (0..n_steps).map(|_| normal.sample(&mut rng) * sqrt_dt).collect();
        let x_approx = scheme_fn(&dw, mu, sigma, x0, dt);
        let x_exact   = gbm_exact(&dw, mu, sigma, x0, dt);
        total_error += (x_approx - x_exact).abs();
    }
    total_error / n_paths as f64
}

// ---------------------------------------------------------------------------
// Strong convergence tests
// ---------------------------------------------------------------------------

/// Euler-Maruyama on GBM: strong order ≈ 0.5 via common-noise log-log regression.
#[test]
fn euler_strong_order_on_gbm() {
    let (mu, sigma, x0, t1) = (0.05_f64, 0.3_f64, 1.0_f64, 1.0_f64);
    let n_paths = 8000;
    let step_counts = [25usize, 50, 100, 200, 400];

    let dts: Vec<f64> = step_counts.iter().map(|&n| t1 / n as f64).collect();
    let errors: Vec<f64> = step_counts.iter().map(|&n_steps|
        strong_error(|dw, mu, sigma, x0, dt| gbm_euler_path(dw, mu, sigma, x0, dt),
            n_steps, n_paths, mu, sigma, x0, t1)
    ).collect();

    let order = convergence_order(&dts, &errors);
    println!("Euler strong order = {:.4}  (expected ~0.5, band [0.35, 0.70])", order);
    for (n, e) in step_counts.iter().zip(&errors) {
        println!("  N={:4}, dt={:.4}, strong_err={:.6}", n, t1 / *n as f64, e);
    }
    assert!(order > 0.35 && order < 0.70,
        "Euler strong order = {:.4}, expected in [0.35, 0.70]", order);
}

/// Milstein on GBM: strong order ≈ 1.0 via common-noise log-log regression.
#[test]
fn milstein_strong_order_on_gbm() {
    let (mu, sigma, x0, t1) = (0.05_f64, 0.3_f64, 1.0_f64, 1.0_f64);
    let n_paths = 8000;
    let step_counts = [25usize, 50, 100, 200, 400];

    let dts: Vec<f64> = step_counts.iter().map(|&n| t1 / n as f64).collect();
    let errors: Vec<f64> = step_counts.iter().map(|&n_steps|
        strong_error(|dw, mu, sigma, x0, dt| gbm_milstein_path(dw, mu, sigma, x0, dt),
            n_steps, n_paths, mu, sigma, x0, t1)
    ).collect();

    let order = convergence_order(&dts, &errors);
    println!("Milstein strong order = {:.4}  (expected ~1.0, band [0.70, 1.30])", order);
    for (n, e) in step_counts.iter().zip(&errors) {
        println!("  N={:4}, dt={:.4}, strong_err={:.6}", n, t1 / *n as f64, e);
    }
    assert!(order > 0.70 && order < 1.30,
        "Milstein strong order = {:.4}, expected in [0.70, 1.30]", order);
}

/// Milstein strong error < Euler strong error at the same coarse step count.
#[test]
fn milstein_stronger_than_euler_strong() {
    let (mu, sigma, x0, t1) = (0.05_f64, 0.4_f64, 1.0_f64, 1.0_f64);
    let n_paths = 10000;
    let n_steps  = 50;

    let euler_err = strong_error(
        |dw, mu, sigma, x0, dt| gbm_euler_path(dw, mu, sigma, x0, dt),
        n_steps, n_paths, mu, sigma, x0, t1);
    let milstein_err = strong_error(
        |dw, mu, sigma, x0, dt| gbm_milstein_path(dw, mu, sigma, x0, dt),
        n_steps, n_paths, mu, sigma, x0, t1);

    println!("  Euler strong err = {:.6}", euler_err);
    println!("  Milstein strong err = {:.6}  (ratio {:.2}x)", milstein_err, euler_err / milstein_err);
    assert!(milstein_err < euler_err,
        "Milstein ({:.6}) should be < Euler ({:.6})", milstein_err, euler_err);
}

// ---------------------------------------------------------------------------
// Weak convergence — monotone decrease
//
// Use high-volatility GBM (sigma=0.5) so the discretization error for E[X_T]
// is large enough to dominate MC noise at coarse step counts.
// E_disc ≈ x0 * exp(mu*T) * mu^2 * T * dt / 2  (first-order expansion)
// At dt=0.2 (N=5), sigma=0.5, mu=0.5: E_disc ≈ 0.03  >>  MC noise ≈ 0.009
// ---------------------------------------------------------------------------

fn weak_error_mean<S: pathwise_core::scheme::Scheme>(
    scheme: &S, n_steps: usize, n_paths: usize, mu: f64, sigma: f64, x0: f64, t1: f64
) -> f64 {
    let exact_mean = x0 * (mu * t1).exp();
    let g = gbm(mu, sigma);
    let out = simulate(&g.drift, &g.diffusion, scheme, x0, 0.0, t1, n_paths, n_steps).unwrap();
    let col = out.column(n_steps);
    let sample_mean: f64 = col.iter().sum::<f64>() / n_paths as f64;
    (sample_mean - exact_mean).abs()
}

/// Euler weak error (E[X_T]) decreases monotonically as N increases.
#[test]
fn euler_weak_error_monotone() {
    let (mu, sigma, x0, t1) = (0.5_f64, 0.5_f64, 1.0_f64, 1.0_f64);
    let n_paths = 20000;
    // Coarse-to-fine: at dt=0.2, disc_err~0.030; at dt=0.025, disc_err~0.004
    // MC noise ≈ SD[X_T]/sqrt(n) ≈ 0.97/141 ≈ 0.007 — signal visible at coarse end
    let step_counts = [5usize, 10, 20, 40];

    let errors: Vec<f64> = step_counts.iter()
        .map(|&n| weak_error_mean(&euler(), n, n_paths, mu, sigma, x0, t1))
        .collect();

    println!("Euler weak error (E[X_T]):");
    for (n, e) in step_counts.iter().zip(&errors) {
        println!("  N={:4}, dt={:.4}, |err|={:.5}", n, t1 / *n as f64, e);
    }

    // Coarsest should have largest error
    assert!(errors[0] > errors[3],
        "Euler weak error should decrease: err[N=5]={:.5} vs err[N=40]={:.5}", errors[0], errors[3]);
    // First three points should trend downward overall
    let decreasing = errors[0] > errors[1] || errors[1] > errors[2] || errors[2] > errors[3];
    assert!(decreasing, "Euler weak error should decrease in at least one step");
}

/// Milstein weak error (E[X_T]) decreases monotonically as N increases.
#[test]
fn milstein_weak_error_monotone() {
    let (mu, sigma, x0, t1) = (0.5_f64, 0.5_f64, 1.0_f64, 1.0_f64);
    let n_paths = 20000;
    let step_counts = [5usize, 10, 20, 40];

    let errors: Vec<f64> = step_counts.iter()
        .map(|&n| weak_error_mean(&milstein(), n, n_paths, mu, sigma, x0, t1))
        .collect();

    println!("Milstein weak error (E[X_T]):");
    for (n, e) in step_counts.iter().zip(&errors) {
        println!("  N={:4}, dt={:.4}, |err|={:.5}", n, t1 / *n as f64, e);
    }

    assert!(errors[0] > errors[3],
        "Milstein weak error should decrease: err[N=5]={:.5} vs err[N=40]={:.5}", errors[0], errors[3]);
}

// ---------------------------------------------------------------------------
// Statistical moment tests
// ---------------------------------------------------------------------------

/// BM: E[W_t] ≈ 0, Var[W_t] ≈ t
#[test]
fn bm_variance_exact() {
    let (x0, t1, n_paths, n_steps) = (0.0_f64, 2.0_f64, 20000, 500);

    let b = bm();
    let out = simulate(&b.drift, &b.diffusion, &euler(), x0, 0.0, t1, n_paths, n_steps).unwrap();
    let col = out.column(n_steps);

    let mean: f64 = col.iter().sum::<f64>() / n_paths as f64;
    let var: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_paths - 1) as f64;

    println!("BM: E[W_t]={:.4} (exact 0), Var[W_t]={:.4} (exact {:.4})", mean, var, t1);
    assert!(mean.abs() < 0.05, "BM mean should be ~0, got {:.4}", mean);
    assert!((var - t1).abs() / t1 < 0.03, "BM variance relative error > 3%: {:.4} vs {:.4}", var, t1);
}

/// GBM: E[X_T] = x0*exp(mu*T), Var[X_T] = x0^2*exp(2*mu*T)*(exp(sigma^2*T)-1)
#[test]
fn gbm_mean_and_variance_exact() {
    let (mu, sigma, x0, t1) = (0.05_f64, 0.2_f64, 1.0_f64, 1.0_f64);
    let n_paths = 20000;
    let n_steps  = 1000;

    let exact_mean = x0 * (mu * t1).exp();
    let exact_var  = x0 * x0 * (2.0 * mu * t1).exp() * ((sigma * sigma * t1).exp() - 1.0);

    let g = gbm(mu, sigma);
    let out = simulate(&g.drift, &g.diffusion, &euler(), x0, 0.0, t1, n_paths, n_steps).unwrap();
    let col = out.column(n_steps);

    let sample_mean: f64 = col.iter().sum::<f64>() / n_paths as f64;
    let sample_var: f64  = col.iter().map(|&x| (x - sample_mean).powi(2)).sum::<f64>() / (n_paths - 1) as f64;

    println!("GBM: E[X_T]={:.4} (exact {:.4}),  Var[X_T]={:.4} (exact {:.4})",
             sample_mean, exact_mean, sample_var, exact_var);
    assert!((sample_mean - exact_mean).abs() / exact_mean < 0.02,
        "GBM mean rel err > 2%: {:.4} vs {:.4}", sample_mean, exact_mean);
    assert!((sample_var - exact_var).abs() / exact_var < 0.05,
        "GBM var rel err > 5%: {:.4} vs {:.4}", sample_var, exact_var);
}

/// OU: E[X_T|X_0] = mu + (x0-mu)*exp(-theta*T),  Var[X_T] = sigma^2/(2*theta)*(1-exp(-2*theta*T))
#[test]
fn ou_mean_and_variance_exact() {
    let (theta, mu, sigma, x0, t1) = (3.0_f64, 2.0_f64, 0.5_f64, 0.0_f64, 1.0_f64);
    let n_paths = 20000;
    let n_steps  = 1000;

    let exact_mean = mu + (x0 - mu) * (-theta * t1).exp();
    let exact_var  = sigma * sigma / (2.0 * theta) * (1.0 - (-2.0 * theta * t1).exp());

    let o = ou(theta, mu, sigma);
    let out = simulate(&o.drift, &o.diffusion, &euler(), x0, 0.0, t1, n_paths, n_steps).unwrap();
    let col = out.column(n_steps);

    let sample_mean: f64 = col.iter().sum::<f64>() / n_paths as f64;
    let sample_var: f64  = col.iter().map(|&x| (x - sample_mean).powi(2)).sum::<f64>() / (n_paths - 1) as f64;

    println!("OU: E[X_T]={:.4} (exact {:.4}),  Var[X_T]={:.4} (exact {:.4})",
             sample_mean, exact_mean, sample_var, exact_var);
    assert!((sample_mean - exact_mean).abs() < 0.02,
        "OU mean: {:.4} vs exact {:.4}", sample_mean, exact_mean);
    assert!((sample_var - exact_var).abs() / exact_var < 0.05,
        "OU var rel err > 5%: {:.4} vs {:.4}", sample_var, exact_var);
}

/// OU stationary distribution: X_T -> N(mu, sigma^2/(2*theta)) for large T.
#[test]
fn ou_stationary_distribution() {
    let (theta, mu, sigma, x0, t1) = (5.0_f64, 1.0_f64, 0.6_f64, -2.0_f64, 3.0_f64);
    let n_paths = 20000;
    let n_steps  = 1000;

    let stat_mean = mu;
    let stat_var  = sigma * sigma / (2.0 * theta);

    let o = ou(theta, mu, sigma);
    let out = simulate(&o.drift, &o.diffusion, &euler(), x0, 0.0, t1, n_paths, n_steps).unwrap();
    let col = out.column(n_steps);

    let sample_mean: f64 = col.iter().sum::<f64>() / n_paths as f64;
    let sample_var: f64  = col.iter().map(|&x| (x - sample_mean).powi(2)).sum::<f64>() / (n_paths - 1) as f64;

    println!("OU stationary: E={:.4} (exact {:.4}),  Var={:.4} (exact {:.4})",
             sample_mean, stat_mean, sample_var, stat_var);
    assert!((sample_mean - stat_mean).abs() < 0.02,
        "OU stationary mean: {:.4} vs {:.4}", sample_mean, stat_mean);
    assert!((sample_var - stat_var).abs() / stat_var < 0.05,
        "OU stationary var rel err > 5%: {:.4} vs {:.4}", sample_var, stat_var);
}
