use thiserror::Error;

#[derive(Error, Debug)]
pub enum PathwiseError {
    #[error("invalid SDE parameters: {0}")]
    InvalidParameters(String),

    #[error("numerical divergence at step {step}: value={value}")]
    NumericalDivergence { step: usize, value: f64 },

    #[error("inference failed to converge: {0}")]
    ConvergenceFailure(String),

    /// CIR Feller condition 2*kappa*theta > sigma^2 not strictly satisfied.
    /// Simulation continues with zero-clipping but accuracy near zero is reduced.
    #[error("Feller condition violated: {0}")]
    FellerViolation(String),

    /// Diffusion matrix shape incompatible with noise or state dimensions.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_are_human_readable() {
        let e = PathwiseError::InvalidParameters("H must be in (0,1)".into());
        assert!(e.to_string().contains("H must be in (0,1)"));
        let e = PathwiseError::FellerViolation("2*1*0.02 <= 0.04".into());
        assert!(e.to_string().contains("Feller"));
        let e = PathwiseError::DimensionMismatch("expected 2x2, got 3x3".into());
        assert!(e.to_string().contains("dimension mismatch"));
    }
}
