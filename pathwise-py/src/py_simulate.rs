use crate::py_error::to_py_err;
use crate::py_process::{PySDE, SDEKind};
use crate::py_scheme::{PyEuler, PyMilstein};
use ndarray::Array2;
use numpy::PyArray2;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

// Must match splitmix64 in pathwise-core/src/simulate.rs exactly.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Simulate SDE paths. Returns np.ndarray of shape (n_paths, n_steps + 1).
///
/// Built-in processes (bm, gbm, ou) run in parallel via Rayon (GIL released).
/// Custom Python-callable SDEs run serially (GIL required per step).
///
/// Non-finite values are stored as NaN; check output if stability is a concern.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (sde, scheme, n_paths, n_steps, t1, x0=0.0, t0=0.0, device="cpu", seed=0))]
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
    seed: u64,
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

    match &sde.kind {
        SDEKind::Bm => {
            let sde_rust = pathwise_core::process::markov::bm();
            let result = if use_milstein {
                py.allow_threads(|| {
                    pathwise_core::simulate(
                        &sde_rust.drift,
                        &sde_rust.diffusion,
                        &pathwise_core::scheme::milstein(),
                        x0,
                        t0,
                        t1,
                        n_paths,
                        n_steps,
                        seed,
                    )
                })
            } else {
                py.allow_threads(|| {
                    pathwise_core::simulate(
                        &sde_rust.drift,
                        &sde_rust.diffusion,
                        &pathwise_core::scheme::euler(),
                        x0,
                        t0,
                        t1,
                        n_paths,
                        n_steps,
                        seed,
                    )
                })
            }
            .map_err(to_py_err)?;
            Ok(numpy::PyArray2::from_owned_array_bound(py, result))
        }
        SDEKind::Gbm { mu, sigma } => {
            let sde_rust = pathwise_core::process::markov::gbm(*mu, *sigma);
            let result = if use_milstein {
                py.allow_threads(|| {
                    pathwise_core::simulate(
                        &sde_rust.drift,
                        &sde_rust.diffusion,
                        &pathwise_core::scheme::milstein(),
                        x0,
                        t0,
                        t1,
                        n_paths,
                        n_steps,
                        seed,
                    )
                })
            } else {
                py.allow_threads(|| {
                    pathwise_core::simulate(
                        &sde_rust.drift,
                        &sde_rust.diffusion,
                        &pathwise_core::scheme::euler(),
                        x0,
                        t0,
                        t1,
                        n_paths,
                        n_steps,
                        seed,
                    )
                })
            }
            .map_err(to_py_err)?;
            Ok(numpy::PyArray2::from_owned_array_bound(py, result))
        }
        SDEKind::Ou { theta, mu, sigma } => {
            let sde_rust = pathwise_core::process::markov::ou(*theta, *mu, *sigma);
            let result = if use_milstein {
                py.allow_threads(|| {
                    pathwise_core::simulate(
                        &sde_rust.drift,
                        &sde_rust.diffusion,
                        &pathwise_core::scheme::milstein(),
                        x0,
                        t0,
                        t1,
                        n_paths,
                        n_steps,
                        seed,
                    )
                })
            } else {
                py.allow_threads(|| {
                    pathwise_core::simulate(
                        &sde_rust.drift,
                        &sde_rust.diffusion,
                        &pathwise_core::scheme::euler(),
                        x0,
                        t0,
                        t1,
                        n_paths,
                        n_steps,
                        seed,
                    )
                })
            }
            .map_err(to_py_err)?;
            Ok(numpy::PyArray2::from_owned_array_bound(py, result))
        }
        SDEKind::Custom { drift, diffusion } => simulate_serial(
            py,
            drift,
            diffusion,
            use_milstein,
            x0,
            t0,
            t1,
            n_paths,
            n_steps,
            seed,
        ),
    }
}

/// GIL-bound serial simulation for custom Python-callable SDEs.
#[allow(clippy::too_many_arguments)]
fn simulate_serial<'py>(
    py: Python<'py>,
    drift_fn: &PyObject,
    diffusion_fn: &PyObject,
    use_milstein: bool,
    x0: f64,
    t0: f64,
    t1: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dt = (t1 - t0) / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    // central-difference step for dg/dx in Milstein correction
    let h = 1e-5_f64;
    let base_seed = splitmix64(seed);
    let mut result = Array2::<f64>::zeros((n_paths, n_steps + 1));

    for i in 0..n_paths {
        let path_seed = splitmix64(base_seed.wrapping_add(i as u64));
        let mut rng = rand::rngs::SmallRng::seed_from_u64(path_seed);
        let normal = Normal::new(0.0_f64, 1.0_f64).unwrap();
        let mut x = x0;
        result[[i, 0]] = x;

        for step in 0..n_steps {
            // Once a path diverges, freeze it at NaN without calling user functions.
            if x.is_nan() {
                result[[i, step + 1]] = f64::NAN;
                continue;
            }
            let t = t0 + step as f64 * dt;
            let dw = normal.sample(&mut rng) * sqrt_dt;

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
