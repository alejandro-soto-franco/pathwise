use pyo3::prelude::*;

/// Format an f64 so it always contains a decimal point, matching Python float repr.
/// e.g. 2.0 -> "2.0", 0.05 -> "0.05"
/// Non-finite values (NaN, inf, -inf) pass through unchanged.
fn fmt_f64(v: f64) -> String {
    if !v.is_finite() {
        return format!("{v}"); // NaN, inf, -inf pass through as-is
    }
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{v}.0")
    }
}

/// Tagged union distinguishing built-in Rust processes from custom Python callables.
/// Built-in variants enable GIL-free parallel dispatch in simulate().
pub(crate) enum SDEKind {
    Bm,
    Gbm {
        mu: f64,
        sigma: f64,
    },
    Ou {
        theta: f64,
        mu: f64,
        sigma: f64,
    },
    Cir {
        kappa: f64,
        theta: f64,
        sigma: f64,
    },
    Heston {
        mu: f64,
        kappa: f64,
        theta: f64,
        xi: f64,
        rho: f64,
    },
    CorrOu {
        theta: f64,
        mu: Vec<f64>,
        sigma_flat: Vec<f64>,
        n: usize,
    },
    Custom {
        drift: PyObject,
        diffusion: PyObject,
    },
}

impl std::fmt::Debug for SDEKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SDEKind::Bm => write!(f, "SDEKind::Bm"),
            SDEKind::Gbm { mu, sigma } => write!(f, "SDEKind::Gbm {{ mu: {mu}, sigma: {sigma} }}"),
            SDEKind::Ou { theta, mu, sigma } => {
                write!(
                    f,
                    "SDEKind::Ou {{ theta: {theta}, mu: {mu}, sigma: {sigma} }}"
                )
            }
            SDEKind::Cir { kappa, theta, sigma } => {
                write!(f, "SDEKind::Cir {{ kappa: {kappa}, theta: {theta}, sigma: {sigma} }}")
            }
            SDEKind::Heston { mu, kappa, theta, xi, rho } => {
                write!(f, "SDEKind::Heston {{ mu: {mu}, kappa: {kappa}, theta: {theta}, xi: {xi}, rho: {rho} }}")
            }
            SDEKind::CorrOu { theta, mu, n, .. } => {
                write!(f, "SDEKind::CorrOu {{ theta: {theta}, n: {n}, mu: {mu:?} }}")
            }
            SDEKind::Custom { .. } => write!(f, "SDEKind::Custom(<PyObject>)"),
        }
    }
}

/// Python-facing SDE. Stores either Rust parameters (built-ins) or Python callables (custom).
#[pyclass(name = "SDE")]
pub struct PySDE {
    pub(crate) kind: SDEKind,
}

#[pymethods]
impl PySDE {
    /// Construct a custom SDE directly. Prefer pw.sde(drift, diffusion) at the Python level.
    #[new]
    pub fn new(drift: PyObject, diffusion: PyObject) -> Self {
        Self {
            kind: SDEKind::Custom { drift, diffusion },
        }
    }

    fn __repr__(&self) -> String {
        match &self.kind {
            SDEKind::Bm => "SDE(bm)".to_string(),
            SDEKind::Gbm { mu, sigma } => {
                format!("SDE(gbm, mu={}, sigma={})", fmt_f64(*mu), fmt_f64(*sigma))
            }
            SDEKind::Ou { theta, mu, sigma } => {
                format!(
                    "SDE(ou, theta={}, mu={}, sigma={})",
                    fmt_f64(*theta),
                    fmt_f64(*mu),
                    fmt_f64(*sigma)
                )
            }
            SDEKind::Cir { kappa, theta, sigma } => {
                format!(
                    "SDE(cir, kappa={}, theta={}, sigma={})",
                    fmt_f64(*kappa),
                    fmt_f64(*theta),
                    fmt_f64(*sigma)
                )
            }
            SDEKind::Heston { mu, kappa, theta, xi, rho } => {
                format!(
                    "SDE(heston, mu={}, kappa={}, theta={}, xi={}, rho={})",
                    fmt_f64(*mu),
                    fmt_f64(*kappa),
                    fmt_f64(*theta),
                    fmt_f64(*xi),
                    fmt_f64(*rho)
                )
            }
            SDEKind::CorrOu { theta, mu, n, .. } => {
                format!("SDE(corr_ou, theta={}, n={}, mu={:?})", fmt_f64(*theta), n, mu)
            }
            SDEKind::Custom { .. } => "SDE(custom)".to_string(),
        }
    }
}

/// Standard Brownian motion: dX = dW
#[pyfunction]
pub fn bm(_py: Python<'_>) -> PySDE {
    PySDE { kind: SDEKind::Bm }
}

/// Geometric Brownian motion: dX = mu*X dt + sigma*X dW
#[pyfunction]
pub fn gbm(_py: Python<'_>, mu: f64, sigma: f64) -> PySDE {
    PySDE {
        kind: SDEKind::Gbm { mu, sigma },
    }
}

/// Ornstein-Uhlenbeck: dX = theta*(mu - X) dt + sigma dW
#[pyfunction]
pub fn ou(_py: Python<'_>, theta: f64, mu: f64, sigma: f64) -> PySDE {
    PySDE {
        kind: SDEKind::Ou { theta, mu, sigma },
    }
}

/// Construct a custom SDE from Python drift and diffusion callables.
#[pyfunction]
pub fn sde(_py: Python<'_>, drift: PyObject, diffusion: PyObject) -> PySDE {
    PySDE {
        kind: SDEKind::Custom { drift, diffusion },
    }
}

/// Cox-Ingersoll-Ross: dX = kappa*(theta - X) dt + sigma*sqrt(X) dW
///
/// Requires strict Feller condition: 2*kappa*theta > sigma^2.
/// Raises ValueError if the condition is violated.
#[pyfunction]
pub fn cir(kappa: f64, theta: f64, sigma: f64) -> PyResult<PySDE> {
    // Delegate to pathwise_core which checks the Feller condition.
    pathwise_core::cir(kappa, theta, sigma)
        .map_err(crate::py_error::to_py_err)?;
    Ok(PySDE {
        kind: SDEKind::Cir { kappa, theta, sigma },
    })
}

/// Heston stochastic volatility model.
/// State: [log S, V]; simulate returns shape (n_paths, n_steps+1, 2).
///
/// d(log S) = (mu - V/2) dt + sqrt(V) dW1
/// dV       = kappa * (theta - V) dt + xi * sqrt(V) * (rho * dW1 + sqrt(1-rho^2) * dW2)
#[pyfunction]
pub fn heston(mu: f64, kappa: f64, theta: f64, xi: f64, rho: f64) -> PySDE {
    PySDE {
        kind: SDEKind::Heston { mu, kappa, theta, xi, rho },
    }
}

/// Correlated Ornstein-Uhlenbeck: dX = theta*(mu - X) dt + Sigma^(1/2) dW
///
/// `mu` must be a list of length N; `sigma_flat` must be a flattened N×N covariance matrix
/// (row-major, length N*N).
#[pyfunction]
pub fn corr_ou(theta: f64, mu: Vec<f64>, sigma_flat: Vec<f64>) -> PyResult<PySDE> {
    let n = mu.len();
    if sigma_flat.len() != n * n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "sigma_flat must have {} elements for {}x{} matrix, got {}",
            n * n, n, n, sigma_flat.len()
        )));
    }
    Ok(PySDE {
        kind: SDEKind::CorrOu { theta, mu, sigma_flat, n },
    })
}
