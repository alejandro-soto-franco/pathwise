// pathwise-core: pure Rust SDE simulation library
pub mod error;
pub use error::PathwiseError;

pub mod process;
pub use process::{SDE, State, Drift, Diffusion};
