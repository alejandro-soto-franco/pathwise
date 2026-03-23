import pytest
import numpy as np
import pathwise as pw


def test_bm_paths_shape():
    paths = pw.simulate(pw.bm(), pw.euler(), n_paths=50, n_steps=100, t1=1.0)
    assert paths.shape == (50, 101)


def test_bm_starts_at_x0():
    paths = pw.simulate(pw.bm(), pw.euler(), n_paths=20, n_steps=50, t1=1.0, x0=3.14)
    assert np.allclose(paths[:, 0], 3.14)


def test_gbm_mean_is_approximately_correct():
    # GBM: E[X_T] = x0 * exp(mu * T)
    mu, sigma, x0, T = 0.05, 0.2, 1.0, 1.0
    paths = pw.simulate(pw.gbm(mu, sigma), pw.euler(), n_paths=3000, n_steps=500, t1=T, x0=x0)
    expected_mean = x0 * np.exp(mu * T)
    sample_mean = paths[:, -1].mean()
    assert abs(sample_mean - expected_mean) < 0.05, (
        f"GBM mean={sample_mean:.4f}, expected~{expected_mean:.4f}"
    )


def test_ou_mean_reverts():
    o = pw.ou(theta=5.0, mu=3.0, sigma=0.1)
    paths = pw.simulate(o, pw.euler(), n_paths=2000, n_steps=500, t1=1.0, x0=0.0)
    final_mean = paths[:, -1].mean()
    assert abs(final_mean - 3.0) < 0.1, f"OU should revert to mu=3.0, got {final_mean}"


def test_custom_sde_with_lambda():
    theta, mu, sigma = 2.0, 1.0, 0.3
    custom = pw.sde(
        drift=lambda x, t: theta * (mu - x),
        diffusion=lambda x, t: sigma,
    )
    paths = pw.simulate(custom, pw.euler(), n_paths=500, n_steps=200, t1=1.0, x0=0.0)
    assert paths.shape == (500, 201)


def test_milstein_produces_different_paths_than_euler_for_gbm():
    # For GBM (state-dependent diffusion), Milstein != Euler
    g = pw.gbm(0.0, 0.5)
    paths_euler = pw.simulate(g, pw.euler(), n_paths=100, n_steps=50, t1=1.0, x0=1.0)
    paths_milstein = pw.simulate(g, pw.milstein(), n_paths=100, n_steps=50, t1=1.0, x0=1.0)
    assert not np.allclose(paths_euler, paths_milstein), "Euler and Milstein should differ on GBM"


def test_simulate_raises_on_invalid_params():
    with pytest.raises(ValueError):
        pw.simulate(pw.bm(), pw.euler(), n_paths=0, n_steps=100, t1=1.0)


def test_simulate_raises_on_gpu_device():
    with pytest.raises(NotImplementedError):
        pw.simulate(pw.bm(), pw.euler(), n_paths=10, n_steps=10, t1=1.0, device="cuda")


def test_numerical_divergence_exception_is_catchable():
    assert issubclass(pw.NumericalDivergence, ArithmeticError)
    assert issubclass(pw.ConvergenceError, RuntimeError)


def test_builtin_repr():
    assert repr(pw.bm()) == "SDE(bm)"
    assert repr(pw.gbm(0.05, 0.2)) == "SDE(gbm, mu=0.05, sigma=0.2)"
    assert repr(pw.ou(2.0, 1.0, 0.3)) == "SDE(ou, theta=2.0, mu=1.0, sigma=0.3)"
    assert "custom" in repr(pw.sde(lambda x, t: x, lambda x, t: 1.0))


def test_scheme_repr():
    assert repr(pw.euler()) == "euler()"
    assert repr(pw.milstein()) == "milstein()"


def test_simulate_seed_reproducibility():
    """Same seed produces identical paths."""
    r1 = pw.simulate(pw.bm(), pw.euler(), n_paths=10, n_steps=50, t1=1.0, seed=42)
    r2 = pw.simulate(pw.bm(), pw.euler(), n_paths=10, n_steps=50, t1=1.0, seed=42)
    np.testing.assert_array_equal(r1, r2)


def test_simulate_different_seeds_differ():
    """Different seeds produce different paths."""
    r1 = pw.simulate(pw.bm(), pw.euler(), n_paths=10, n_steps=50, t1=1.0, seed=0)
    r2 = pw.simulate(pw.bm(), pw.euler(), n_paths=10, n_steps=50, t1=1.0, seed=1)
    assert not np.array_equal(r1, r2)


def test_custom_sde_seed_reproducibility():
    """Seed also works for custom (serial) SDEs."""
    custom = pw.sde(lambda x, t: 0.0, lambda x, t: 1.0)
    r1 = pw.simulate(custom, pw.euler(), n_paths=5, n_steps=20, t1=1.0, seed=7)
    r2 = pw.simulate(custom, pw.euler(), n_paths=5, n_steps=20, t1=1.0, seed=7)
    np.testing.assert_array_equal(r1, r2)
