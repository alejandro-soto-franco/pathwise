use pyo3::prelude::*;

mod py_error;
mod py_process;
mod py_scheme;
mod py_simulate;

use py_process::{bm, cir, corr_ou, gbm, heston, ou, sde, PySDE};
use py_scheme::{euler, milstein, sri, PyEuler, PyMilstein, PySri};
use py_simulate::simulate;

#[pymodule]
fn _pathwise(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "NumericalDivergence",
        py.get_type_bound::<py_error::NumericalDivergence>(),
    )?;
    m.add(
        "ConvergenceError",
        py.get_type_bound::<py_error::ConvergenceError>(),
    )?;
    m.add_class::<PySDE>()?;
    m.add_function(wrap_pyfunction!(bm, m)?)?;
    m.add_function(wrap_pyfunction!(gbm, m)?)?;
    m.add_function(wrap_pyfunction!(ou, m)?)?;
    m.add_function(wrap_pyfunction!(cir, m)?)?;
    m.add_function(wrap_pyfunction!(heston, m)?)?;
    m.add_function(wrap_pyfunction!(corr_ou, m)?)?;
    m.add_function(wrap_pyfunction!(sde, m)?)?;
    m.add_class::<PyEuler>()?;
    m.add_class::<PyMilstein>()?;
    m.add_class::<PySri>()?;
    m.add_function(wrap_pyfunction!(euler, m)?)?;
    m.add_function(wrap_pyfunction!(milstein, m)?)?;
    m.add_function(wrap_pyfunction!(sri, m)?)?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    Ok(())
}
