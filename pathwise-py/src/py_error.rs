use pyo3::exceptions::{PyValueError, PyArithmeticError, PyRuntimeError};
use pyo3::{PyErr, Python, create_exception};
use pathwise_core::error::PathwiseError;

create_exception!(pathwise, NumericalDivergence, PyArithmeticError);
create_exception!(pathwise, ConvergenceError, PyRuntimeError);

pub fn to_py_err(e: PathwiseError) -> PyErr {
    match e {
        PathwiseError::InvalidParameters(msg) => PyValueError::new_err(msg),
        PathwiseError::NumericalDivergence { step, value } => {
            NumericalDivergence::new_err(format!("diverged at step {}: value={}", step, value))
        }
        PathwiseError::ConvergenceFailure(msg) => ConvergenceError::new_err(msg),
    }
}
