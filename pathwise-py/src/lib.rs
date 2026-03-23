use pyo3::prelude::*;

mod py_error;

#[pymodule]
fn _pathwise(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NumericalDivergence", py.get_type::<py_error::NumericalDivergence>())?;
    m.add("ConvergenceError", py.get_type::<py_error::ConvergenceError>())?;
    Ok(())
}
