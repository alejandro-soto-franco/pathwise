use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyTuple};

/// Python-facing SDE wrapper. Stores drift/diffusion as Python callables.
/// Simulation loop calls them via the GIL (see py_simulate.rs).
#[pyclass(name = "SDE")]
pub struct PySDE {
    pub drift_fn: PyObject,
    pub diffusion_fn: PyObject,
}

#[pymethods]
impl PySDE {
    #[new]
    pub fn new(drift: PyObject, diffusion: PyObject) -> Self {
        Self { drift_fn: drift, diffusion_fn: diffusion }
    }
}

/// Standard Brownian motion: drift=0, diffusion=1
#[pyfunction]
pub fn bm(py: Python<'_>) -> PyResult<PySDE> {
    let drift = PyCFunction::new_closure_bound(py, None, None,
        |_args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<f64> { Ok(0.0) }
    )?;
    let diff = PyCFunction::new_closure_bound(py, None, None,
        |_args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<f64> { Ok(1.0) }
    )?;
    Ok(PySDE::new(drift.into_any().unbind(), diff.into_any().unbind()))
}

/// Geometric Brownian motion: drift=mu*x, diffusion=sigma*x
#[pyfunction]
pub fn gbm(py: Python<'_>, mu: f64, sigma: f64) -> PyResult<PySDE> {
    let drift = PyCFunction::new_closure_bound(py, None, None,
        move |args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<f64> {
            let x: f64 = args.get_item(0)?.extract()?;
            Ok(mu * x)
        }
    )?;
    let diff = PyCFunction::new_closure_bound(py, None, None,
        move |args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<f64> {
            let x: f64 = args.get_item(0)?.extract()?;
            Ok(sigma * x)
        }
    )?;
    Ok(PySDE::new(drift.into_any().unbind(), diff.into_any().unbind()))
}

/// Ornstein-Uhlenbeck: drift=theta*(mu-x), diffusion=sigma
#[pyfunction]
pub fn ou(py: Python<'_>, theta: f64, mu: f64, sigma: f64) -> PyResult<PySDE> {
    let drift = PyCFunction::new_closure_bound(py, None, None,
        move |args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<f64> {
            let x: f64 = args.get_item(0)?.extract()?;
            Ok(theta * (mu - x))
        }
    )?;
    let diff = PyCFunction::new_closure_bound(py, None, None,
        move |_args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<f64> { Ok(sigma) }
    )?;
    Ok(PySDE::new(drift.into_any().unbind(), diff.into_any().unbind()))
}

/// Construct a custom SDE from Python drift and diffusion callables.
#[pyfunction]
pub fn sde(_py: Python<'_>, drift: PyObject, diffusion: PyObject) -> PySDE {
    PySDE::new(drift, diffusion)
}
