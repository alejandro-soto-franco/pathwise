use std::marker::PhantomData;

pub trait State: Clone + Send + Sync + 'static {}
impl<T: Clone + Send + Sync + 'static> State for T {}

pub trait Drift<S: State>: Fn(&S, f64) -> S + Send + Sync {}
impl<S: State, F: Fn(&S, f64) -> S + Send + Sync> Drift<S> for F {}

pub trait Diffusion<S: State>: Fn(&S, f64) -> S + Send + Sync {}
impl<S: State, F: Fn(&S, f64) -> S + Send + Sync> Diffusion<S> for F {}

pub struct SDE<S: State, D: Drift<S>, G: Diffusion<S>> {
    pub drift: D,
    pub diffusion: G,
    _s: PhantomData<S>,
}

impl<S: State, D: Drift<S>, G: Diffusion<S>> SDE<S, D, G> {
    pub fn new(drift: D, diffusion: G) -> Self {
        Self { drift, diffusion, _s: PhantomData }
    }

    pub fn eval_drift(&self, x: &S, t: f64) -> S {
        (self.drift)(x, t)
    }

    pub fn eval_diffusion(&self, x: &S, t: f64) -> S {
        (self.diffusion)(x, t)
    }
}

/// Standard Brownian motion: dX = dW
pub fn bm() -> SDE<f64, impl Drift<f64>, impl Diffusion<f64>> {
    SDE::new(|_x: &f64, _t: f64| 0.0_f64, |_x: &f64, _t: f64| 1.0_f64)
}

/// Geometric Brownian motion: dX = mu*X dt + sigma*X dW
pub fn gbm(mu: f64, sigma: f64) -> SDE<f64, impl Drift<f64>, impl Diffusion<f64>> {
    SDE::new(
        move |x: &f64, _t: f64| mu * x,
        move |x: &f64, _t: f64| sigma * x,
    )
}

/// Ornstein-Uhlenbeck: dX = theta*(mu - X) dt + sigma dW
pub fn ou(theta: f64, mu: f64, sigma: f64) -> SDE<f64, impl Drift<f64>, impl Diffusion<f64>> {
    SDE::new(
        move |x: &f64, _t: f64| theta * (mu - x),
        move |_x: &f64, _t: f64| sigma,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sde_evaluates_drift_and_diffusion() {
        let sde = SDE::new(
            |x: &f64, _t: f64| -0.5 * x,
            |_x: &f64, _t: f64| 1.0_f64,
        );
        assert_eq!(sde.eval_drift(&2.0, 0.0), -1.0);
        assert_eq!(sde.eval_diffusion(&2.0, 0.0), 1.0);
    }

    #[test]
    fn sde_closures_capture_parameters() {
        let theta = 0.7_f64;
        let sde = SDE::new(
            move |x: &f64, _t: f64| -theta * x,
            |_x: &f64, _t: f64| 1.0_f64,
        );
        assert!((sde.eval_drift(&1.0, 0.0) - (-0.7)).abs() < 1e-12);
    }

    #[test]
    fn bm_has_zero_drift_unit_diffusion() {
        let b = bm();
        assert_eq!(b.eval_drift(&1.5, 0.5), 0.0);
        assert_eq!(b.eval_diffusion(&1.5, 0.5), 1.0);
    }

    #[test]
    fn gbm_drift_and_diffusion() {
        let g = gbm(0.05, 0.2);
        assert!((g.eval_drift(&2.0, 0.0) - 0.1).abs() < 1e-12);
        assert!((g.eval_diffusion(&2.0, 0.0) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn ou_drift_and_diffusion() {
        let o = ou(1.0, 0.0, 0.5);
        assert!((o.eval_drift(&1.0, 0.0) - (-1.0)).abs() < 1e-12);
        assert!((o.eval_diffusion(&1.0, 0.0) - 0.5).abs() < 1e-12);
    }
}
