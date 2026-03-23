use pyo3::prelude::*;

#[pyclass(name = "Euler")]
pub struct PyEuler;

#[pyclass(name = "Milstein")]
pub struct PyMilstein;

#[pymethods]
impl PyEuler {
    #[new]
    fn new() -> Self {
        PyEuler
    }

    fn __repr__(&self) -> &str {
        "euler()"
    }
}

#[pymethods]
impl PyMilstein {
    #[new]
    fn new() -> Self {
        PyMilstein
    }

    fn __repr__(&self) -> &str {
        "milstein()"
    }
}

#[pyfunction]
pub fn euler() -> PyEuler {
    PyEuler
}

#[pyfunction]
pub fn milstein() -> PyMilstein {
    PyMilstein
}
