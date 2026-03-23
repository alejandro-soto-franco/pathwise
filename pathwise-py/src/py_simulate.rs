use crate::py_process::{PySDE, SDEKind};
use crate::py_scheme::{PyEuler, PyMilstein};
use ndarray::Array2;
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyTuple};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Simulate SDE paths. Returns np.ndarray of shape (n_paths, n_steps + 1).
///
/// Note: simulation is serial per path (GIL required for Python callables).
/// For throughput-critical workloads, use built-in bm/gbm/ou processes.
///
/// Non-finite values are stored as NaN; check output if stability is a concern.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (sde, scheme, n_paths, n_steps, t1, x0=0.0, t0=0.0, device="cpu"))]
pub fn simulate<'py>(
    py: Python<'py>,
    sde: &PySDE,
    scheme: &Bound<'_, PyAny>,
    n_paths: usize,
    n_steps: usize,
    t1: f64,
    x0: f64,
    t0: f64,
    device: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if device != "cpu" {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "GPU device support is planned for v0.3; use device='cpu'",
        ));
    }
    if n_paths == 0 || n_steps == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_paths and n_steps must be > 0",
        ));
    }
    if t1 <= t0 {
        return Err(pyo3::exceptions::PyValueError::new_err("t1 must be > t0"));
    }

    let use_milstein = scheme.is_instance_of::<PyMilstein>();
    if !use_milstein && !scheme.is_instance_of::<PyEuler>() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "scheme must be euler() or milstein()",
        ));
    }

    // Extract drift and diffusion as Python callables, converting built-ins on the fly.
    // Task 3 will replace this block with GIL-free parallel dispatch for built-in variants.
    let (drift_fn, diffusion_fn): (PyObject, PyObject) = match &sde.kind {
        SDEKind::Bm => {
            let d = PyCFunction::new_closure_bound(py, None, None, |_args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<f64> {
                Ok(0.0)
            })?;
            let g = PyCFunction::new_closure_bound(py, None, None, |_args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<f64> {
                Ok(1.0)
            })?;
            (d.into_any().unbind(), g.into_any().unbind())
        }
        SDEKind::Gbm { mu, sigma } => {
            let mu = *mu;
            let sigma = *sigma;
            let d = PyCFunction::new_closure_bound(py, None, None, move |args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<f64> {
                let x: f64 = args.get_item(0)?.extract()?;
                Ok(mu * x)
            })?;
            let g = PyCFunction::new_closure_bound(py, None, None, move |args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<f64> {
                let x: f64 = args.get_item(0)?.extract()?;
                Ok(sigma * x)
            })?;
            (d.into_any().unbind(), g.into_any().unbind())
        }
        SDEKind::Ou { theta, mu, sigma } => {
            let (theta, mu, sigma) = (*theta, *mu, *sigma);
            let d = PyCFunction::new_closure_bound(py, None, None, move |args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<f64> {
                let x: f64 = args.get_item(0)?.extract()?;
                Ok(theta * (mu - x))
            })?;
            let g = PyCFunction::new_closure_bound(py, None, None, move |_args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<f64> {
                Ok(sigma)
            })?;
            (d.into_any().unbind(), g.into_any().unbind())
        }
        SDEKind::Custom { drift, diffusion } => {
            (drift.clone_ref(py), diffusion.clone_ref(py))
        }
    };

    let dt = (t1 - t0) / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let h = 1e-5_f64;
    let mut result = Array2::<f64>::zeros((n_paths, n_steps + 1));

    for i in 0..n_paths {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(i as u64);
        let normal = Normal::new(0.0_f64, sqrt_dt).unwrap();
        let mut x = x0;
        result[[i, 0]] = x;

        for step in 0..n_steps {
            let t = t0 + step as f64 * dt;
            let dw = normal.sample(&mut rng);

            let f: f64 = drift_fn.call1(py, (x, t))?.extract(py)?;
            let g: f64 = diffusion_fn.call1(py, (x, t))?.extract(py)?;

            x = if use_milstein {
                let g_plus: f64 = diffusion_fn.call1(py, (x + h, t))?.extract(py)?;
                let g_minus: f64 = diffusion_fn.call1(py, (x - h, t))?.extract(py)?;
                let dg_dx = (g_plus - g_minus) / (2.0 * h);
                x + f * dt + g * dw + 0.5 * g * dg_dx * (dw * dw - dt)
            } else {
                x + f * dt + g * dw
            };

            if !x.is_finite() {
                x = f64::NAN;
            }
            result[[i, step + 1]] = x;
        }
    }

    Ok(numpy::PyArray2::from_owned_array_bound(py, result))
}
