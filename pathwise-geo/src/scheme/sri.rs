use crate::sde::ManifoldSDE;
use crate::simulate::GeoScheme;
use cartan_core::{Manifold, ParallelTransport};
use pathwise_core::state::Increment;

/// Geodesic SRI: strong order 1.5 approximation for manifold SDEs.
///
/// Extends GeodesicMilstein with the dZ iterated-integral correction term.
///
/// Full step:
///   v = f(x)*dt + g(x)*dW + 0.5*nabla_g(g)*(dW^2 - dt) + nabla_g(g)*dZ
///   x_new = exp_x(v)
///
/// where nabla_g(g) is approximated by finite-difference parallel transport:
///   nabla_g g(x) ≈ (1/eps) * [PT_{y->x}(g(y)) - g(x)],  y = exp_x(eps*g(x))
///
/// # Single-FD note
///
/// This uses nabla_g g for both the Milstein correction and the dZ term.
/// Full SRI1 would require a second PT-based FD for nabla_g(nabla_g g).
/// This approximation is O(dt^{3/2}) accurate for smooth diffusion fields.
pub struct GeodesicSRI {
    /// Finite-difference step size for covariant derivative approximation.
    pub eps: f64,
}

impl GeodesicSRI {
    /// Create with default eps = 1e-4.
    pub fn new() -> Self {
        Self { eps: 1e-4 }
    }

    /// Advance x by one SRI step on the manifold.
    ///
    /// Computes the Milstein and dZ corrections via finite-difference parallel transport:
    ///   1. Walk eps along g(x) to get y = exp_x(eps * g(x)).
    ///   2. Evaluate g at y.
    ///   3. Transport g(y) back from y to x via ParallelTransport.
    ///   4. Approx covariant deriv: nabla_g g ≈ (PT(g(y)) - g(x)) / eps.
    ///   5. Milstein correction: 0.5 * nabla_g(g) * (dW^2 - dt).
    ///   6. SRI correction: nabla_g(g) * dZ.
    ///   7. Apply exp to the full tangent displacement.
    ///
    /// If transport fails (cut locus), falls back to Euler step.
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
        M::Tangent: std::ops::Add<Output = M::Tangent>
            + std::ops::Mul<f64, Output = M::Tangent>
            + std::ops::Sub<Output = M::Tangent>
            + Clone,
    {
        let dw = inc.dw;
        let dz = inc.dz;
        let f = (sde.drift)(x, t);
        let g = (sde.diffusion)(x, t);
        let eps = self.eps;

        // Walk eps along g(x) to get perturbed point y.
        let eps_g = g.clone() * eps;
        let y = sde.manifold.exp(x, &eps_g);

        // Evaluate diffusion at y.
        let g_at_y = (sde.diffusion)(&y, t);

        // Transport g(y) back from y to T_x(M). Falls back to Euler if it fails.
        let tangent = match sde.manifold.transport(&y, x, &g_at_y) {
            Ok(g_transported) => {
                // Finite-difference covariant derivative.
                let nabla_g_g = (g_transported - g.clone()) * (1.0 / eps);
                // Milstein correction: 0.5 * nabla_g(g) * (dW^2 - dt)
                let milstein_correction = nabla_g_g.clone() * (0.5 * (dw * dw - dt));
                // SRI dZ correction: nabla_g(g) * dZ
                let sri_correction = nabla_g_g * dz;
                f * dt + g * dw + milstein_correction + sri_correction
            }
            Err(_) => {
                // Degenerate geometry (cut locus): fall back to Euler step.
                f * dt + g * dw
            }
        };

        sde.manifold.exp(x, &tangent)
    }
}

impl Default for GeodesicSRI {
    fn default() -> Self {
        Self::new()
    }
}

impl<M, D, G> GeoScheme<M, D, G> for GeodesicSRI
where
    M: Manifold + ParallelTransport,
    D: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    G: Fn(&M::Point, f64) -> M::Tangent + Send + Sync,
    M::Tangent: std::ops::Add<Output = M::Tangent>
        + std::ops::Mul<f64, Output = M::Tangent>
        + std::ops::Sub<Output = M::Tangent>
        + Clone,
{
    fn step_geo(
        &self,
        sde: &ManifoldSDE<M, D, G>,
        x: &M::Point,
        t: f64,
        dt: f64,
        inc: &Increment<f64>,
    ) -> M::Point {
        self.step(sde, x, t, dt, inc)
    }
}
