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
