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
}
