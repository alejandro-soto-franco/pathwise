use thiserror::Error;

#[derive(Error, Debug)]
pub enum PathwiseError {
    #[error("invalid SDE parameters: {0}")]
    InvalidParameters(String),

    #[error("numerical divergence at step {step}: value={value}")]
    NumericalDivergence { step: usize, value: f64 },

    #[error("inference failed to converge: {0}")]
    ConvergenceFailure(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_are_human_readable() {
        let e = PathwiseError::InvalidParameters("H must be in (0,1)".into());
        assert!(e.to_string().contains("H must be in (0,1)"));

        let e = PathwiseError::NumericalDivergence { step: 42, value: f64::NAN };
        assert!(e.to_string().contains("42"));

        let e = PathwiseError::ConvergenceFailure("max iterations reached".into());
        assert!(e.to_string().contains("max iterations"));
    }
}
