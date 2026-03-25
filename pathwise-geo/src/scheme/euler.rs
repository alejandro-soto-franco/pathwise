use crate::sde::ManifoldSDE;
use cartan_core::Manifold;
use pathwise_core::state::Increment;

/// Geodesic Euler-Maruyama on a Riemannian manifold.
/// x_{n+1} = Exp_{x_n}( f(x_n,t)*dt + g(x_n,t)*dW )
/// Strong order 0.5. Keeps state exactly on the manifold by using the
/// Riemannian exponential map after each step.
pub struct GeodesicEuler;

impl GeodesicEuler {
    /// Advance x by one step of geodesic Euler-Maruyama.
    ///
    /// Computes the tangent displacement f(x,t)*dt + g(x,t)*dW and
    /// maps it back to the manifold via Exp_x. Since all v0.1 cartan
    /// manifolds are complete, Exp is total and always succeeds.
    pub fn step<M, D, G>(
        &self,
        sde: &ManifoldSDE<M, D, G>,
        x: &M::Point,
        t: f64,
        dt: f64,
        inc: &Increment<f64>,
    ) -> M::Point
    where
        M: Manifold,
        D: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
        G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    {
        let f = (sde.drift)(x, t);
        let g = (sde.diffusion)(x, t);
        // f * dt + g * dW, using cartan-core Tangent's Mul<Real> and Add bounds.
        let tangent = f * dt + g * inc.dw;
        sde.manifold.exp(x, &tangent)
    }
}
