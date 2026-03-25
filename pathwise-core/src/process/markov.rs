use std::marker::PhantomData;
use crate::state::{Diffusion, NoiseIncrement, State};

pub trait Drift<S: State>: Fn(&S, f64) -> S + Send + Sync {}
impl<S: State, F: Fn(&S, f64) -> S + Send + Sync> Drift<S> for F {}

pub struct SDE<S: State + NoiseIncrement, D: Drift<S>, G: Diffusion<S, S>> {
    pub drift: D,
    pub diffusion: G,
    _s: PhantomData<S>,
}

impl<S: State + NoiseIncrement, D: Drift<S>, G: Diffusion<S, S>> SDE<S, D, G> {
    pub fn new(drift: D, diffusion: G) -> Self {
        Self {
            drift,
            diffusion,
            _s: PhantomData,
        }
    }

    pub fn eval_drift(&self, x: &S, t: f64) -> S {
        (self.drift)(x, t)
    }
}

/// Standard Brownian motion: dX = dW
///
/// # Example
///
/// ```
/// use pathwise_core::{simulate, bm, euler};
///
/// let b = bm();
/// let scheme = euler();
/// let paths = simulate(
///     &b.drift,
///     &b.diffusion,
///     &scheme,
///     0.0,   // x0
///     0.0,   // t0
///     1.0,   // t1
///     5,     // n_paths
///     100,   // n_steps
///     0,     // seed
/// ).expect("simulate failed");
/// assert_eq!(paths.shape(), &[5, 101]);
/// ```
pub fn bm() -> SDE<f64, impl Drift<f64>, impl Diffusion<f64, f64>> {
    SDE::new(|_x: &f64, _t: f64| 0.0_f64, |_x: f64, _t: f64| 1.0_f64)
}

/// Geometric Brownian motion: dX = mu*X dt + sigma*X dW
///
/// # Example
///
/// ```
/// use pathwise_core::{simulate, gbm, euler};
///
/// let g = gbm(0.05, 0.2);
/// let scheme = euler();
/// let paths = simulate(&g.drift, &g.diffusion, &scheme, 100.0, 0.0, 1.0, 5, 100, 0).expect("simulate failed");
/// assert_eq!(paths.shape(), &[5, 101]);
/// // All starting values equal x0 = 100.0
/// for i in 0..5 {
///     assert!((paths[[i, 0]] - 100.0).abs() < 1e-12);
/// }
/// ```
pub fn gbm(mu: f64, sigma: f64) -> SDE<f64, impl Drift<f64>, impl Diffusion<f64, f64>> {
    SDE::new(
        move |x: &f64, _t: f64| mu * x,
        move |x: f64, _t: f64| sigma * x,
    )
}

/// Ornstein-Uhlenbeck: dX = theta*(mu - X) dt + sigma dW
///
/// # Example
///
/// ```
/// use pathwise_core::{simulate, ou, euler};
///
/// // Mean-reverting process: theta=2.0, long-run mean=1.0, vol=0.3
/// let o = ou(2.0, 1.0, 0.3);
/// let scheme = euler();
/// let paths = simulate(&o.drift, &o.diffusion, &scheme, 0.0, 0.0, 1.0, 5, 100, 0).expect("simulate failed");
/// assert_eq!(paths.shape(), &[5, 101]);
/// ```
pub fn ou(theta: f64, mu: f64, sigma: f64) -> SDE<f64, impl Drift<f64>, impl Diffusion<f64, f64>> {
    SDE::new(
        move |x: &f64, _t: f64| theta * (mu - x),
        move |_x: f64, _t: f64| sigma,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Increment;

    #[test]
    fn sde_evaluates_drift_and_diffusion() {
        let sde = SDE::new(|x: &f64, _t: f64| -0.5 * x, |_x: f64, _t: f64| 1.0_f64);
        assert_eq!(sde.eval_drift(&2.0, 0.0), -1.0);
        // diffusion.apply with unit dw=1.0 gives g(x)*1.0 = 1.0
        assert_eq!(sde.diffusion.apply(&2.0, 0.0, &1.0_f64), 1.0);
    }

    #[test]
    fn sde_closures_capture_parameters() {
        let theta = 0.7_f64;
        let sde = SDE::new(
            move |x: &f64, _t: f64| -theta * x,
            |_x: f64, _t: f64| 1.0_f64,
        );
        assert!((sde.eval_drift(&1.0, 0.0) - (-0.7)).abs() < 1e-12);
    }

    #[test]
    fn bm_has_zero_drift_unit_diffusion() {
        let b = bm();
        assert_eq!(b.eval_drift(&1.5, 0.5), 0.0);
        // apply with unit dw=1.0 gives g(x,t)*1.0 = 1.0
        assert_eq!(b.diffusion.apply(&1.5, 0.5, &1.0_f64), 1.0);
    }

    #[test]
    fn gbm_drift_and_diffusion() {
        let g = gbm(0.05, 0.2);
        assert!((g.eval_drift(&2.0, 0.0) - 0.1).abs() < 1e-12);
        // sigma*x = 0.2*2.0 = 0.4; apply with dw=1.0
        assert!((g.diffusion.apply(&2.0, 0.0, &1.0_f64) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn ou_drift_and_diffusion() {
        let o = ou(1.0, 0.0, 0.5);
        assert!((o.eval_drift(&1.0, 0.0) - (-1.0)).abs() < 1e-12);
        assert!((o.diffusion.apply(&1.0, 0.0, &1.0_f64) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn increment_roundtrip() {
        let inc = Increment { dw: 0.3_f64, dz: 0.0_f64 };
        assert!((inc.dw - 0.3).abs() < 1e-12);
    }
}
