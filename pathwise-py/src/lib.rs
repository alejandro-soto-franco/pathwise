use pyo3::prelude::*;

#[pymodule]
fn _pathwise(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
