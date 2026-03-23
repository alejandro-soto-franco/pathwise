use pyo3::prelude::*;

mod py_error;
mod py_process;
mod py_scheme;
mod py_simulate;

use py_process::{PySDE, bm, gbm, ou, sde};
use py_scheme::{PyEuler, PyMilstein, euler, milstein};
use py_simulate::simulate;

#[pymodule]
fn _pathwise(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NumericalDivergence", py.get_type::<py_error::NumericalDivergence>())?;
    m.add("ConvergenceError", py.get_type::<py_error::ConvergenceError>())?;
    m.add_class::<PySDE>()?;
    m.add_function(wrap_pyfunction!(bm, m)?)?;
    m.add_function(wrap_pyfunction!(gbm, m)?)?;
    m.add_function(wrap_pyfunction!(ou, m)?)?;
    m.add_function(wrap_pyfunction!(sde, m)?)?;
    m.add_class::<PyEuler>()?;
    m.add_class::<PyMilstein>()?;
    m.add_function(wrap_pyfunction!(euler, m)?)?;
    m.add_function(wrap_pyfunction!(milstein, m)?)?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    Ok(())
}
