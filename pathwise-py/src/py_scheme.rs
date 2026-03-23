use pyo3::prelude::*;

#[pyclass(name = "Euler")]
pub struct PyEuler;

#[pyclass(name = "Milstein")]
pub struct PyMilstein;

#[pymethods]
impl PyEuler {
    #[new]
    fn new() -> Self { PyEuler }
}

#[pymethods]
impl PyMilstein {
    #[new]
    fn new() -> Self { PyMilstein }
}

#[pyfunction]
pub fn euler() -> PyEuler { PyEuler }

#[pyfunction]
pub fn milstein() -> PyMilstein { PyMilstein }
