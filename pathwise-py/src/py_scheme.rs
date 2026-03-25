use pyo3::prelude::*;

#[pyclass(name = "Euler")]
pub struct PyEuler;

#[pyclass(name = "Sri")]
#[derive(Clone)]
pub struct PySri;

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

#[pymethods]
impl PySri {
    #[new]
    pub fn new() -> Self {
        PySri
    }

    fn __repr__(&self) -> &str {
        "sri()"
    }
}

#[pyfunction]
pub fn sri() -> PySri {
    PySri
}
