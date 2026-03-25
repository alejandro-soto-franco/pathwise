use pathwise_core::error::PathwiseError;
use pyo3::exceptions::{PyArithmeticError, PyRuntimeError, PyValueError};
use pyo3::{create_exception, PyErr};

// Design note: non-finite values are stored as NaN rather than raising NumericalDivergence.
// This allows vectorized post-processing (e.g. np.isnan masking) and lets callers decide
// how to handle divergent paths. NumericalDivergence is reserved for cases where the
// entire simulation is structurally unsound (future: adaptive step rejection).
create_exception!(pathwise, NumericalDivergence, PyArithmeticError);
create_exception!(pathwise, ConvergenceError, PyRuntimeError);

pub fn to_py_err(e: PathwiseError) -> PyErr {
    match e {
        PathwiseError::InvalidParameters(msg) => PyValueError::new_err(msg),
        PathwiseError::NumericalDivergence { step, value } => {
            NumericalDivergence::new_err(format!("diverged at step {step}: value={value}"))
        }
        PathwiseError::ConvergenceFailure(msg) => ConvergenceError::new_err(msg),
        PathwiseError::FellerViolation(msg) => {
            PyValueError::new_err(format!("Feller condition violated: {msg}"))
        }
        PathwiseError::DimensionMismatch(msg) => PyValueError::new_err(msg),
    }
}
