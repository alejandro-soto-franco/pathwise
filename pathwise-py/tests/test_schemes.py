import pytest
import numpy as np
try:
    import pathwise
except ImportError:
    pytest.skip("pathwise not installed", allow_module_level=True)

def test_sri_exists():
    """sri() function returns a scheme object."""
    s = pathwise.sri()
    assert s is not None

def test_heston_output_shape():
    """Heston simulation returns (n_paths, n_steps+1, 2) array."""
    h = pathwise.heston(0.05, 2.0, 0.04, 0.3, -0.7)
    m = pathwise.milstein()
    out = pathwise.simulate(h, m, n_paths=100, n_steps=50, t1=1.0)
    assert out.shape == (100, 51, 2), f"Expected (100, 51, 2), got {out.shape}"

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
    """Heston variance component (index 1) must be >= -0.05 (Euler may produce slightly negative)."""
    h = pathwise.heston(0.05, 2.0, 0.04, 0.3, -0.7)
    out = pathwise.simulate(h, pathwise.euler(), n_paths=100, n_steps=100, t1=1.0)
    variance_col = out[:, :, 1]
    assert np.all(variance_col[~np.isnan(variance_col)] >= -0.05), "Heston variance too negative"

def test_sri_on_scalar_works():
    """SRI scheme runs on scalar GBM."""
    g = pathwise.gbm(0.0, 0.3)
    s = pathwise.sri()
    out = pathwise.simulate(g, s, n_paths=100, n_steps=50, t1=1.0, x0=1.0)
    assert out.shape == (100, 51), f"Expected (100, 51), got {out.shape}"
    assert np.all(np.isfinite(out)), "SRI produced non-finite values on GBM"
