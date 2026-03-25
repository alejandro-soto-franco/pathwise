use std::marker::PhantomData;
use nalgebra::SVector;
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

/// Cox-Ingersoll-Ross: dX = kappa*(theta - X) dt + sigma*sqrt(X) dW
///
/// Requires `kappa > 0`, `theta > 0`, `sigma > 0`.
/// Strict Feller condition: `2*kappa*theta > sigma^2`. Returns `Err(FellerViolation)` if not met.
/// Simulation clips X to 0.0 when discretization produces negative values (full truncation).
pub fn cir(
    kappa: f64,
    theta: f64,
    sigma: f64,
) -> Result<SDE<f64, impl Drift<f64>, impl Fn(f64, f64) -> f64 + Send + Sync>, crate::error::PathwiseError> {
    if kappa <= 0.0 || theta <= 0.0 || sigma <= 0.0 {
        return Err(crate::error::PathwiseError::InvalidParameters(
            format!("CIR requires kappa, theta, sigma > 0; got kappa={}, theta={}, sigma={}", kappa, theta, sigma)
        ));
    }
    if 2.0 * kappa * theta <= sigma * sigma {
        return Err(crate::error::PathwiseError::FellerViolation(
            format!("2*kappa*theta = {:.4} <= sigma^2 = {:.4}; boundary is reflecting in continuous time but clipping may introduce bias under discretization",
                2.0 * kappa * theta, sigma * sigma)
        ));
    }
    Ok(SDE::new(
        move |x: &f64, _t: f64| kappa * (theta - x),
        move |x: f64, _t: f64| sigma * x.max(0.0).sqrt(),
    ))
}

/// NdSDE: N-dimensional SDE with vector state and vector noise.
pub struct NdSDE<const N: usize, D, G> {
    pub drift: D,
    pub diffusion: G,
}

impl<const N: usize, D, G> NdSDE<N, D, G>
where
    D: Fn(&SVector<f64, N>, f64) -> SVector<f64, N> + Send + Sync,
    G: Diffusion<SVector<f64, N>, SVector<f64, N>>,
{
    pub fn new(drift: D, diffusion: G) -> Self {
        Self { drift, diffusion }
    }
}

/// Diffusion term for the Heston model (log-price, variance).
/// State: [log S, V]. Noise: [dW1, dW2] (independent).
///
/// Applies the lower-triangular Cholesky matrix:
///   d(log S) += sqrt(V) * dW1
///   dV       += xi * sqrt(V) * (rho * dW1 + sqrt(1-rho^2) * dW2)
///
/// Full truncation: V is clipped to 0 in diffusion computation.
pub struct HestonDiffusion {
    xi: f64,
    rho: f64,
    rho_perp: f64,  // sqrt(1 - rho^2)
}

impl HestonDiffusion {
    pub fn new(xi: f64, rho: f64) -> Self {
        Self { xi, rho, rho_perp: (1.0 - rho * rho).sqrt() }
    }
}

impl Diffusion<SVector<f64, 2>, SVector<f64, 2>> for HestonDiffusion {
    fn apply(&self, x: &SVector<f64, 2>, _t: f64, dw: &SVector<f64, 2>) -> SVector<f64, 2> {
        let v = x[1].max(0.0);  // full truncation
        let sv = v.sqrt();
        SVector::from([
            sv * dw[0],
            sv * self.xi * (self.rho * dw[0] + self.rho_perp * dw[1]),
        ])
    }
}

/// Heston stochastic volatility model.
/// State: [log S, V]; use exp(paths[.., .., 0]) to recover S.
///
/// d(log S) = (mu - V/2) dt + sqrt(V) dW1
/// dV       = kappa * (theta - V) dt + xi * sqrt(V) * (rho * dW1 + sqrt(1-rho^2) * dW2)
///
/// Parameters:
/// - `mu`: risk-neutral drift of log-price
/// - `kappa`: variance mean-reversion speed
/// - `theta`: long-run variance
/// - `xi`: volatility of variance (vol of vol)
/// - `rho`: correlation between price and variance Brownian motions (typically -0.7)
pub fn heston(
    mu: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
) -> NdSDE<2, impl Fn(&SVector<f64, 2>, f64) -> SVector<f64, 2> + Send + Sync, HestonDiffusion> {
    NdSDE::new(
        move |x: &SVector<f64, 2>, _t: f64| -> SVector<f64, 2> {
            let v = x[1].max(0.0);
            SVector::from([mu - v / 2.0, kappa * (theta - x[1])])
        },
        HestonDiffusion::new(xi, rho),
    )
}

/// Correlated Ornstein-Uhlenbeck diffusion via Cholesky factor.
/// apply: L * dW where L = chol(Sigma)
pub struct CorrOuDiffusion<const N: usize> {
    l: nalgebra::SMatrix<f64, N, N>,
}

impl<const N: usize> crate::state::Diffusion<nalgebra::SVector<f64, N>, nalgebra::SVector<f64, N>> for CorrOuDiffusion<N> {
    fn apply(&self, _x: &nalgebra::SVector<f64, N>, _t: f64, dw: &nalgebra::SVector<f64, N>) -> nalgebra::SVector<f64, N> {
        self.l * dw
    }
}

/// N-dimensional correlated Ornstein-Uhlenbeck process.
/// dX = theta*(mu - X) dt + L dW  where L = chol(Sigma)
///
/// Returns `Err(DimensionMismatch)` if sigma_mat Cholesky fails (not positive-definite).
pub fn corr_ou<const N: usize>(
    theta: f64,
    mu: nalgebra::SVector<f64, N>,
    sigma_mat: nalgebra::SMatrix<f64, N, N>,
) -> Result<
    NdSDE<N, impl Fn(&nalgebra::SVector<f64, N>, f64) -> nalgebra::SVector<f64, N> + Send + Sync, CorrOuDiffusion<N>>,
    crate::error::PathwiseError,
> {
    let chol = nalgebra::Cholesky::new(sigma_mat).ok_or_else(|| {
        crate::error::PathwiseError::DimensionMismatch(
            "sigma_mat is not positive-definite (Cholesky failed)".into()
        )
    })?;
    let l = chol.l();
    Ok(NdSDE::new(
        move |x: &nalgebra::SVector<f64, N>, _t: f64| -> nalgebra::SVector<f64, N> {
            (mu - x) * theta
        },
        CorrOuDiffusion { l },
    ))
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
