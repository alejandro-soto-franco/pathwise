use crate::process::markov::Drift;
use crate::state::{Diffusion, Increment, NoiseIncrement, State};

/// One-step advance of an SDE of type S.
///
/// The `type Noise` associated type fixes the noise increment type for this scheme.
/// For all current schemes on scalar or nD-diagonal processes, `Noise = S`.
pub trait Scheme<S: State>: Send + Sync {
    type Noise: NoiseIncrement;

    fn step<D, G>(
        &self,
        drift: &D,
        diffusion: &G,
        x: &S,
        t: f64,
        dt: f64,
        inc: &Increment<Self::Noise>,
    ) -> S
    where
        D: Drift<S>,
        G: Diffusion<S, Self::Noise>;
}

pub mod euler;
pub mod milstein;
pub mod milstein_nd;
pub mod sri;
pub use euler::{euler, EulerMaruyama};
pub use milstein::{milstein, Milstein};
pub use milstein_nd::{milstein_nd, MilsteinNd};
pub use sri::{sri, Sri};
