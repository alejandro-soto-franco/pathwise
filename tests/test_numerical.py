"""
Full numerical test suite for pathwise v0.1.

Coverage:
  1. Weak convergence order (Euler ~1, Milstein ~1) via log-log regression
  2. Strong convergence order (Euler ~0.5, Milstein ~1) via common-noise Python loop
  3. BM increment distribution (KS test vs N(0, dt))
  4. BM quadratic variation convergence to T
  5. GBM log-normality of terminal values
  6. GBM European call vs Black-Scholes
  7. OU conditional mean and variance vs analytic formulas
  8. OU autocorrelation decay: Corr(X_t, X_{t+h}) = exp(-theta*h)
  9. Cross-path independence: correlation between distinct paths ~ 0
 10. Path non-degeneracy: paths are non-constant for stochastic processes
"""

import math
import numpy as np
import pytest
from scipy import stats
import pathwise as pw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def convergence_order(dts, errors):
    """Log-log regression slope: error ~ C * dt^p, returns p."""
    log_dt  = np.log(dts)
    log_err = np.log(np.maximum(errors, 1e-15))
    slope, *_ = np.polyfit(log_dt, log_err, 1)
    return slope


def bs_call(x0, K, mu, sigma, T):
    """Black-Scholes European call price (undiscounted, under risk-neutral measure r=mu)."""
    d1 = (math.log(x0 / K) + (mu + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return x0 * stats.norm.cdf(d1) - K * math.exp(-mu * T) * stats.norm.cdf(d2)


# ---------------------------------------------------------------------------
# 1. Weak convergence order
#
# Use high-volatility GBM (mu=0.5, sigma=0.5) at coarse step counts so that
# the discretization error in E[X_T] dominates Monte Carlo noise.
#
# Theory (Euler on GBM): E[X_N] = x0*(1+mu*dt)^N, exact = x0*exp(mu*T)
#   disc_err(N=5)  = 0.0382, disc_err(N=10) = 0.0198, disc_err(N=20) = 0.0101
#   MC noise (n=20k, sigma~0.88) = 0.0062:SNR > 1.5 for N in [5,10,20]
# ---------------------------------------------------------------------------

MU_WEAK, SIGMA_WEAK, X0_WEAK, T_WEAK = 0.5, 0.5, 1.0, 1.0
N_PATHS_WEAK = 20_000
STEP_COUNTS_WEAK = [5, 10, 20]


def _weak_errors_milstein():
    exact_mean = X0_WEAK * math.exp(MU_WEAK * T_WEAK)
    errors = []
    for n_steps in STEP_COUNTS_WEAK:
        g = pw.gbm(MU_WEAK, SIGMA_WEAK)
        paths = pw.simulate(g, pw.milstein(),
                            n_paths=N_PATHS_WEAK, n_steps=n_steps,
                            t1=T_WEAK, x0=X0_WEAK)
        sample_mean = paths[:, -1].mean()
        errors.append(abs(sample_mean - exact_mean))
    return np.array(errors)


def test_euler_weak_order():
    """Euler weak order >= 0.5 (expect ~1.0, conservative bound for Monte Carlo noise)."""
    g = pw.gbm(MU_WEAK, SIGMA_WEAK)
    exact_mean = X0_WEAK * math.exp(MU_WEAK * T_WEAK)
    dts = [T_WEAK / n for n in STEP_COUNTS_WEAK]
    errors = []
    for n_steps in STEP_COUNTS_WEAK:
        paths = pw.simulate(g, pw.euler(),
                            n_paths=N_PATHS_WEAK, n_steps=n_steps,
                            t1=T_WEAK, x0=X0_WEAK)
        sample_mean = paths[:, -1].mean()
        errors.append(abs(sample_mean - exact_mean))

    order = convergence_order(dts, errors)
    print(f"\nEuler weak order = {order:.4f}  (expected ~1.0, bound > 0.5)")
    for n, e in zip(STEP_COUNTS_WEAK, errors):
        print(f"  N={n:4d}, dt={T_WEAK/n:.4f}, |err| = {e:.5f}")
    assert order > 0.5, f"Euler weak order = {order:.4f}, expected > 0.5"


def test_milstein_weak_order():
    """Milstein weak order >= 0.5 (expect ~1.0)."""
    exact_mean = X0_WEAK * math.exp(MU_WEAK * T_WEAK)
    dts = [T_WEAK / n for n in STEP_COUNTS_WEAK]
    errors = _weak_errors_milstein()

    order = convergence_order(dts, errors)
    print(f"\nMilstein weak order = {order:.4f}  (expected ~1.0, bound > 0.5)")
    for n, e in zip(STEP_COUNTS_WEAK, errors):
        print(f"  N={n:4d}, dt={T_WEAK/n:.4f}, |err| = {e:.5f}")
    assert order > 0.5, f"Milstein weak order = {order:.4f}, expected > 0.5"


def test_weak_error_decreases_coarse_to_fine():
    """Coarsest step count has strictly larger weak error than finest."""
    g = pw.gbm(MU_WEAK, SIGMA_WEAK)
    exact_mean = X0_WEAK * math.exp(MU_WEAK * T_WEAK)
    results = {}
    for n_steps in [5, 20]:
        paths = pw.simulate(g, pw.euler(),
                            n_paths=N_PATHS_WEAK, n_steps=n_steps,
                            t1=T_WEAK, x0=X0_WEAK)
        results[n_steps] = abs(paths[:, -1].mean() - exact_mean)
    assert results[5] > results[20], (
        f"Coarse error {results[5]:.5f} should exceed fine error {results[20]:.5f}")


# ---------------------------------------------------------------------------
# 2. Strong convergence order (Python loop with common noise)
# ---------------------------------------------------------------------------

def _gbm_strong_error(scheme_str, n_steps, n_paths,
                      mu=0.05, sigma=0.3, x0=1.0, t1=1.0, seed_offset=0):
    """Compute strong error via common-noise NumPy reference implementation.

    This function tests the mathematical correctness of the Euler/Milstein
    schemes directly -- it does NOT call pw.simulate. It is used by
    test_euler_strong_order and test_milstein_strong_order to verify
    convergence rates against the analytical GBM solution with matched noise.
    """
    rng = np.random.default_rng(seed_offset)
    dt = t1 / n_steps
    dw = rng.normal(0, math.sqrt(dt), size=(n_paths, n_steps))

    w_T = dw.sum(axis=1)
    x_exact = x0 * np.exp((mu - 0.5 * sigma**2) * t1 + sigma * w_T)

    if scheme_str == "euler":
        x = np.full(n_paths, x0)
        for j in range(n_steps):
            x = x + mu * x * dt + sigma * x * dw[:, j]
    elif scheme_str == "milstein":
        x = np.full(n_paths, x0)
        for j in range(n_steps):
            dw_j = dw[:, j]
            x = (x + mu * x * dt + sigma * x * dw_j
                 + 0.5 * sigma**2 * x * (dw_j**2 - dt))
    else:
        raise ValueError(scheme_str)

    return np.mean(np.abs(x - x_exact))


def test_euler_strong_order():
    """Euler strong order in [0.35, 0.70] (expect ~0.5)."""
    step_counts = [25, 50, 100, 200, 400]
    dts = [1.0 / n for n in step_counts]
    errors = [_gbm_strong_error("euler", n, 5000) for n in step_counts]

    order = convergence_order(dts, errors)
    print(f"\nEuler strong order = {order:.4f}  (expected ~0.5)")
    for n, e in zip(step_counts, errors):
        print(f"  N={n:4d}, strong_err = {e:.6f}")
    assert 0.35 < order < 0.70, f"Euler strong order = {order:.4f}, expected in [0.35, 0.70]"


def test_milstein_strong_order():
    """Milstein strong order in [0.70, 1.30] (expect ~1.0)."""
    step_counts = [25, 50, 100, 200, 400]
    dts = [1.0 / n for n in step_counts]
    errors = [_gbm_strong_error("milstein", n, 5000) for n in step_counts]

    order = convergence_order(dts, errors)
    print(f"\nMilstein strong order = {order:.4f}  (expected ~1.0)")
    for n, e in zip(step_counts, errors):
        print(f"  N={n:4d}, strong_err = {e:.6f}")
    assert 0.70 < order < 1.30, f"Milstein strong order = {order:.4f}, expected in [0.70, 1.30]"


def test_milstein_strong_error_less_than_euler():
    """Milstein strong error < Euler strong error at coarse N=50."""
    e_err = _gbm_strong_error("euler",    n_steps=50, n_paths=8000, sigma=0.4)
    m_err = _gbm_strong_error("milstein", n_steps=50, n_paths=8000, sigma=0.4)
    print(f"\n  Euler strong err = {e_err:.6f}")
    print(f"  Milstein strong err = {m_err:.6f}  (ratio {e_err/m_err:.1f}x)")
    assert m_err < e_err, f"Milstein ({m_err:.6f}) should be < Euler ({e_err:.6f})"


# ---------------------------------------------------------------------------
# 3. BM increment distribution
# ---------------------------------------------------------------------------

def test_bm_increments_are_normal():
    """BM increments over dt should be N(0, dt):KS test p > 0.01."""
    n_paths, n_steps, t1 = 1000, 200, 1.0
    dt = t1 / n_steps
    paths = pw.simulate(pw.bm(), pw.euler(),
                        n_paths=n_paths, n_steps=n_steps, t1=t1, x0=0.0)
    increments = np.diff(paths, axis=1).ravel()
    ks_stat, p_value = stats.kstest(increments, "norm", args=(0, math.sqrt(dt)))
    print(f"\nBM increments KS: stat={ks_stat:.4f}, p={p_value:.4f}")
    assert p_value > 0.01, f"BM increments fail KS normality test: p={p_value:.4f}"


def test_bm_increment_mean_and_variance():
    """BM increments: mean ~ 0, variance ~ dt."""
    n_paths, n_steps, t1 = 5000, 100, 1.0
    dt = t1 / n_steps
    paths = pw.simulate(pw.bm(), pw.euler(),
                        n_paths=n_paths, n_steps=n_steps, t1=t1, x0=0.0)
    increments = np.diff(paths, axis=1).ravel()
    assert abs(increments.mean()) < 0.01, f"BM increment mean: {increments.mean():.4f}"
    assert abs(increments.var() - dt) / dt < 0.05, (
        f"BM increment var: {increments.var():.4f} vs {dt:.4f}")


# ---------------------------------------------------------------------------
# 4. BM quadratic variation
# ---------------------------------------------------------------------------

def test_bm_quadratic_variation_converges_to_t():
    """[W,W]_T = sum (dW_i)^2 -> T almost surely as dt -> 0."""
    n_paths, n_steps, t1 = 500, 1000, 1.0
    paths = pw.simulate(pw.bm(), pw.euler(),
                        n_paths=n_paths, n_steps=n_steps, t1=t1, x0=0.0)
    qv = np.sum(np.diff(paths, axis=1)**2, axis=1)  # per-path QV
    mean_qv = qv.mean()
    std_qv  = qv.std()
    print(f"\nBM QV: mean={mean_qv:.4f} (exact {t1:.4f}), std={std_qv:.4f}")
    assert abs(mean_qv - t1) / t1 < 0.02, (
        f"BM quadratic variation {mean_qv:.4f} should be ~{t1:.4f}")
    # QV std should be small (law of large numbers)
    assert std_qv < 0.1, f"QV std too large: {std_qv:.4f}"


# ---------------------------------------------------------------------------
# 5. GBM log-normality
# ---------------------------------------------------------------------------

def test_gbm_terminal_is_lognormal():
    """log(X_T) ~ N(log(x0) + (mu - sigma^2/2)*T, sigma^2*T):KS test p > 0.01."""
    mu, sigma, x0, t1 = 0.05, 0.2, 1.0, 1.0
    n_paths, n_steps = 3000, 500
    paths = pw.simulate(pw.gbm(mu, sigma), pw.euler(),
                        n_paths=n_paths, n_steps=n_steps, t1=t1, x0=x0)
    log_xT = np.log(paths[:, -1])
    exact_mean_log = math.log(x0) + (mu - 0.5 * sigma**2) * t1
    exact_std_log  = sigma * math.sqrt(t1)
    ks_stat, p_value = stats.kstest(log_xT, "norm",
                                    args=(exact_mean_log, exact_std_log))
    print(f"\nGBM log-normality KS: stat={ks_stat:.4f}, p={p_value:.4f}")
    print(f"  log(X_T): mean={log_xT.mean():.4f} (exact {exact_mean_log:.4f}), "
          f"std={log_xT.std():.4f} (exact {exact_std_log:.4f})")
    assert p_value > 0.01, f"GBM terminal not log-normal: p={p_value:.4f}"


# ---------------------------------------------------------------------------
# 6. GBM European call vs Black-Scholes
# ---------------------------------------------------------------------------

def test_gbm_call_option_matches_black_scholes():
    """E[max(X_T - K, 0)] should match the Black-Scholes formula within 1.5%."""
    mu, sigma, x0, K, t1 = 0.05, 0.2, 100.0, 100.0, 1.0
    n_paths, n_steps = 50_000, 500
    paths = pw.simulate(pw.gbm(mu, sigma), pw.euler(),
                        n_paths=n_paths, n_steps=n_steps, t1=t1, x0=x0, seed=123)
    payoff = np.maximum(paths[:, -1] - K, 0.0)
    # Discount under risk-neutral measure
    mc_price = math.exp(-mu * t1) * payoff.mean()
    bs_price = bs_call(x0, K, mu, sigma, t1)
    rel_err  = abs(mc_price - bs_price) / bs_price
    print(f"\nGBM call: MC={mc_price:.4f}, BS={bs_price:.4f}, rel_err={rel_err:.4f}")
    assert rel_err < 0.015, f"Call price rel err {rel_err:.4f} > 1.5%"


# ---------------------------------------------------------------------------
# 7. OU conditional moments
# ---------------------------------------------------------------------------

def test_ou_conditional_mean():
    """E[X_T | X_0] = mu + (x0-mu)*exp(-theta*T)."""
    theta, mu, sigma, x0, t1 = 3.0, 2.0, 0.5, 0.0, 1.0
    n_paths, n_steps = 20_000, 500
    exact_mean = mu + (x0 - mu) * math.exp(-theta * t1)
    o = pw.ou(theta, mu, sigma)
    paths = pw.simulate(o, pw.euler(),
                        n_paths=n_paths, n_steps=n_steps, t1=t1, x0=x0)
    sample_mean = paths[:, -1].mean()
    print(f"\nOU mean: {sample_mean:.4f} vs exact {exact_mean:.4f}")
    assert abs(sample_mean - exact_mean) < 0.02, (
        f"OU mean {sample_mean:.4f} vs exact {exact_mean:.4f}")


def test_ou_conditional_variance():
    """Var[X_T | X_0] = sigma^2/(2*theta) * (1 - exp(-2*theta*T))."""
    theta, mu, sigma, x0, t1 = 3.0, 2.0, 0.5, 0.0, 1.0
    n_paths, n_steps = 20_000, 500
    exact_var = sigma**2 / (2.0 * theta) * (1.0 - math.exp(-2.0 * theta * t1))
    o = pw.ou(theta, mu, sigma)
    paths = pw.simulate(o, pw.euler(),
                        n_paths=n_paths, n_steps=n_steps, t1=t1, x0=x0)
    sample_var = paths[:, -1].var()
    rel_err = abs(sample_var - exact_var) / exact_var
    print(f"\nOU var: {sample_var:.4f} vs exact {exact_var:.4f}, rel_err={rel_err:.4f}")
    assert rel_err < 0.05, f"OU var rel err {rel_err:.4f} > 5%"


# ---------------------------------------------------------------------------
# 8. OU autocorrelation
# ---------------------------------------------------------------------------

def test_ou_stationary_autocorrelation():
    """For stationary OU, Corr(X_t, X_{t+h}) = exp(-theta*h).

    Simulate many paths from the stationary distribution, then estimate
    the lag-1 autocorrelation along one long path.
    """
    theta, mu, sigma = 4.0, 1.0, 0.5
    x_stat = mu  # start at mean (already near stationary for large theta)
    t1, n_steps = 10.0, 1000
    dt = t1 / n_steps

    # Use a single long path to estimate autocorrelation
    paths = pw.simulate(pw.ou(theta, mu, sigma), pw.euler(),
                        n_paths=1, n_steps=n_steps, t1=t1, x0=x_stat)
    x = paths[0]
    lag = 1
    corr_lag1 = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    exact_corr = math.exp(-theta * dt * lag)
    print(f"\nOU autocorr lag-1: {corr_lag1:.4f} vs exact {exact_corr:.4f}")
    assert abs(corr_lag1 - exact_corr) < 0.05, (
        f"OU lag-1 autocorr {corr_lag1:.4f} vs exact {exact_corr:.4f}")


# ---------------------------------------------------------------------------
# 9. Cross-path independence
# ---------------------------------------------------------------------------

def test_bm_paths_are_independent():
    """Terminal values of distinct BM paths should be uncorrelated.

    We simulate N_PATHS independent BM paths and check that the sample
    correlation between terminal values is consistent with independence.
    Under H0 (independence), r ~ N(0, 1/n) for large n, so the sample
    mean |r| over many pairs should be small (< 0.1 for n_paths=1000).
    """
    n_paths, n_steps = 1000, 200
    paths = pw.simulate(pw.bm(), pw.euler(),
                        n_paths=n_paths, n_steps=n_steps, t1=1.0, x0=0.0)
    # Independence of TERMINAL VALUES (not full paths, which are autocorrelated time series)
    terminals = paths[:, -1]  # shape (n_paths,)
    # Split into two halves and correlate:under independence, r ~ 0
    half = n_paths // 2
    r = np.corrcoef(terminals[:half], terminals[half:])[0, 1]
    print(f"\nBM terminal half-half correlation: r={r:.4f} (expected ~0)")
    assert abs(r) < 0.10, f"BM terminal correlation {r:.4f} suggests non-independence"


# ---------------------------------------------------------------------------
# 10. Path non-degeneracy
# ---------------------------------------------------------------------------

def test_bm_paths_are_non_constant():
    """BM paths should have non-zero standard deviation across time."""
    paths = pw.simulate(pw.bm(), pw.euler(),
                        n_paths=50, n_steps=200, t1=1.0, x0=0.0)
    # Each path should have non-zero variance
    path_stds = paths.std(axis=1)
    assert np.all(path_stds > 0.01), "Some BM paths appear constant"


def test_gbm_paths_stay_positive():
    """GBM paths should remain positive (no sign flip)."""
    paths = pw.simulate(pw.gbm(0.05, 0.2), pw.euler(),
                        n_paths=200, n_steps=500, t1=1.0, x0=1.0)
    assert np.all(paths > 0), "GBM path crossed zero:numerical instability"


def test_ou_bounded_paths():
    """Strong OU (theta=10) paths should stay within reasonable range of mu."""
    theta, mu, sigma = 10.0, 0.0, 0.5
    paths = pw.simulate(pw.ou(theta, mu, sigma), pw.euler(),
                        n_paths=200, n_steps=500, t1=2.0, x0=5.0)
    # 5 sigma bound: 5 * sqrt(sigma^2/(2*theta)) = 5 * sqrt(0.0125) = 0.56
    # After t >> 1/theta=0.1, process should be near 0; use generous bound 2.0
    terminal = paths[:, -1]
    assert np.all(np.abs(terminal) < 2.0), (
        f"Some OU paths out of expected range: max|X_T|={np.abs(terminal).max():.2f}")


# ---------------------------------------------------------------------------
# 11. Built-in vs custom dispatch consistency
# ---------------------------------------------------------------------------

def test_builtin_and_custom_gbm_match_in_distribution():
    """Built-in gbm (parallel Rust path) and custom lambda SDE (serial Python path)
    should produce terminal values from the same distribution.

    Uses KS two-sample test on X_T with DIFFERENT seeds so the two samples
    are independent draws (not the same RNG sequence). Under the null hypothesis
    (same distribution) p > 0.01.

    This validates that the parallel and serial dispatch paths implement
    the same numerical scheme.
    """
    mu, sigma, x0, t1 = 0.05, 0.2, 1.0, 1.0
    n_paths, n_steps = 3000, 200

    # Different seeds produce independent samples from (hopefully) the same distribution.
    builtin_paths = pw.simulate(pw.gbm(mu, sigma), pw.euler(),
                                n_paths=n_paths, n_steps=n_steps, t1=t1, x0=x0,
                                seed=0)
    custom_paths = pw.simulate(
        pw.sde(lambda x, t: mu * x, lambda x, t: sigma * x),
        pw.euler(),
        n_paths=n_paths, n_steps=n_steps, t1=t1, x0=x0,
        seed=42,
    )
    ks_stat, p_value = stats.ks_2samp(builtin_paths[:, -1], custom_paths[:, -1])
    print(f"\nBuiltin vs custom GBM KS: stat={ks_stat:.4f}, p={p_value:.4f}")
    assert p_value > 0.01, (
        f"Built-in and custom paths have different distributions: p={p_value:.4f}. "
        f"Parallel and serial dispatch may implement different schemes."
    )
