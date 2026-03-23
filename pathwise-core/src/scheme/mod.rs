use crate::process::markov::{Diffusion, Drift};

/// One-step advance of a scalar SDE.
pub trait Scheme: Send + Sync {
    fn step<D: Drift<f64>, G: Diffusion<f64>>(
        &self,
        drift: &D,
        diffusion: &G,
        x: f64,
        t: f64,
        dt: f64,
        dw: f64,
    ) -> f64;
}

pub mod euler;
pub mod milstein;
pub use euler::{euler, EulerMaruyama};
pub use milstein::{milstein, Milstein};
