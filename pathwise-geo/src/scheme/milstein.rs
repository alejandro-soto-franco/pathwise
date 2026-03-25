use crate::sde::ManifoldSDE;
use cartan_core::{Manifold, ParallelTransport};
use pathwise_core::state::Increment;

/// Geodesic Milstein: strong order 1.0 scheme via finite-difference covariant derivative.
///
/// Correction via finite-difference approximation of nabla_g(g):
///   nabla_g g(x) ≈ (1/eps) * [PT_{y->x}(g(y)) - g(x)]
///   where y = exp_x(eps * g(x))
///
/// Full step:
///   v = f(x)*dt + g(x)*dW + 0.5 * nabla_g(g) * (dW^2 - dt)
///   x_new = exp_x(v)
///
/// Requires M: ParallelTransport to compute the covariant derivative via
/// transporting g(y) back to T_x(M) along the geodesic from y to x.
///
/// References:
///   - Milstein (1974), Platen & Wagner (1982) for the scalar correction.
///   - Said & Manton (2012) for geodesic extension to Lie groups.
pub struct GeodesicMilstein {
    /// Finite-difference step size for covariant derivative approximation.
    pub eps: f64,
}

impl GeodesicMilstein {
    /// Create with default eps = 1e-4.
    pub fn new() -> Self {
        Self { eps: 1e-4 }
    }

    /// Advance x by one Milstein step on the manifold.
    ///
    /// Computes the Milstein correction via finite-difference parallel transport:
    ///   1. Walk eps along g(x) to get y = exp_x(eps * g(x)).
    ///   2. Evaluate g at y.
    ///   3. Transport g(y) back from y to x via ParallelTransport.
    ///   4. Approx covariant deriv: nabla_g g ≈ (PT(g(y)) - g(x)) / eps.
    ///   5. Add Milstein correction: 0.5 * nabla_g(g) * (dW^2 - dt).
    ///   6. Apply exp to the full tangent displacement.
    ///
    /// If transport fails (cut locus), falls back to Euler step (no correction).
    pub fn step<M, D, G>(
        &self,
        sde: &ManifoldSDE<M, D, G>,
        x: &M::Point,
        t: f64,
        dt: f64,
        inc: &Increment<f64>,
    ) -> M::Point
    where
        M: Manifold + ParallelTransport,
        D: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
        G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    {
        let dw = inc.dw;
        let f = (sde.drift)(x, t);
        let g = (sde.diffusion)(x, t);
        let eps = self.eps;

        // Walk eps along g(x) to get perturbed point y.
        let eps_g = g.clone() * eps;
        let y = sde.manifold.exp(x, &eps_g);

        // Evaluate diffusion at y.
        let g_at_y = (sde.diffusion)(&y, t);

        // Transport g(y) from y back to T_x(M). Falls back to Euler if it fails.
        let tangent = match sde.manifold.transport(&y, x, &g_at_y) {
            Ok(g_transported) => {
                // Finite-difference covariant derivative: nabla_g g ≈ (PT(g(y)) - g(x)) / eps.
                let nabla_g_g = (g_transported - g.clone()) * (1.0 / eps);
                let correction = nabla_g_g * (0.5 * (dw * dw - dt));
                f * dt + g * dw + correction
            }
            Err(_) => {
                // Degenerate geometry (cut locus): fall back to Euler step.
                f * dt + g * dw
            }
        };

        sde.manifold.exp(x, &tangent)
    }
}

impl Default for GeodesicMilstein {
    fn default() -> Self {
        Self::new()
    }
}
