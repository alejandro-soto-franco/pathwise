use pathwise_core::error::PathwiseError;
use pyo3::exceptions::{PyArithmeticError, PyRuntimeError, PyValueError};
use pyo3::{create_exception, PyErr};

create_exception!(pathwise, NumericalDivergence, PyArithmeticError);
create_exception!(pathwise, ConvergenceError, PyRuntimeError);

pub fn to_py_err(e: PathwiseError) -> PyErr {
    match e {
        PathwiseError::InvalidParameters(msg) => PyValueError::new_err(msg),
        PathwiseError::NumericalDivergence { step, value } => {
            NumericalDivergence::new_err(format!("diverged at step {step}: value={value}"))
        }
        PathwiseError::ConvergenceFailure(msg) => ConvergenceError::new_err(msg),
        PathwiseError::FellerViolation(msg) => PyValueError::new_err(format!("Feller condition violated: {msg}")),
        PathwiseError::DimensionMismatch(msg) => PyValueError::new_err(msg),
    }
}
